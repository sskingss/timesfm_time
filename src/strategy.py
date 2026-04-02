"""
量化交易策略模块
实现三种基于 TimesFM 预测的交易策略：
1. 趋势预测策略 - 基于点预测的趋势方向，宏观因子调节阈值
2. 分位数风控策略 - 利用分位数预测做风险控制，宏观因子调节容忍度
3. 多 Horizon 集成策略 - 综合多个预测窗口加权投票 + 宏观投票
"""
import numpy as np
from . import config

SIGNAL_BUY = 1
SIGNAL_SELL = -1
SIGNAL_HOLD = 0


def _safe_macro(features_snapshot: dict) -> float:
    """从特征快照中安全读取宏观评分"""
    v = features_snapshot.get("macro_score", 0.0)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    return float(v)


class BaseStrategy:
    """策略基类"""

    def __init__(self, name):
        self.name = name
        self.macro_influence = config.STRATEGY_CONFIG.get("macro_influence", 0.3)

    def generate_signal(self, current_price, point_forecast, quantile_forecast, features_snapshot):
        raise NotImplementedError


class TrendStrategy(BaseStrategy):
    """
    策略 1：趋势预测策略

    宏观影响:
    - macro > 0（乐观）→ 降低买入阈值 / 提高卖出阈值 → 更容易做多
    - macro < 0（悲观）→ 提高买入阈值 / 降低卖出阈值 → 更容易做空
    """

    def __init__(self, threshold=None):
        super().__init__("趋势预测策略")
        self.threshold = threshold or config.STRATEGY_CONFIG["trend_threshold"]

    def generate_signal(self, current_price, point_forecast, quantile_forecast, features_snapshot):
        if point_forecast is None or len(point_forecast) == 0:
            return SIGNAL_HOLD, 0.0, "无预测数据"

        macro = _safe_macro(features_snapshot)

        forecast_mean = np.mean(point_forecast)
        expected_return = (forecast_mean - current_price) / current_price
        end_return = (point_forecast[-1] - current_price) / current_price
        combined_return = 0.6 * expected_return + 0.4 * end_return

        # 宏观调节：乐观时降低买入门槛，悲观时提高
        buy_threshold = self.threshold * (1 - macro * self.macro_influence)
        sell_threshold = self.threshold * (1 + macro * self.macro_influence)

        rsi = features_snapshot.get("rsi", 50)
        if isinstance(rsi, float) and np.isnan(rsi):
            rsi = 50

        if combined_return > buy_threshold and rsi < 75:
            strength = min(abs(combined_return) / self.threshold, 1.0)
            return (SIGNAL_BUY, strength,
                    f"预测上涨 {combined_return:.2%}, RSI={rsi:.0f}, macro={macro:+.2f}")
        elif combined_return < -sell_threshold and rsi > 25:
            strength = min(abs(combined_return) / self.threshold, 1.0)
            return (SIGNAL_SELL, strength,
                    f"预测下跌 {combined_return:.2%}, RSI={rsi:.0f}, macro={macro:+.2f}")
        else:
            return SIGNAL_HOLD, 0.0, f"预测变化 {combined_return:.2%} 未达阈值"


class QuantileRiskStrategy(BaseStrategy):
    """
    策略 2：分位数风控策略

    宏观影响:
    - macro > 0 → 稍微放宽下行风险容忍度（更愿意承担风险）
    - macro < 0 → 收紧风险容忍度（更保守）
    """

    def __init__(self, risk_threshold=None):
        super().__init__("分位数风控策略")
        self.risk_threshold = risk_threshold or config.STRATEGY_CONFIG["quantile_risk_threshold"]

    def generate_signal(self, current_price, point_forecast, quantile_forecast, features_snapshot):
        if quantile_forecast is None or len(quantile_forecast) == 0:
            return SIGNAL_HOLD, 0.0, "无分位数预测"

        macro = _safe_macro(features_snapshot)

        # 宏观调节风险阈值
        adj_risk = self.risk_threshold * (1 + macro * self.macro_influence)

        horizon_end = min(len(quantile_forecast) - 1, 9)
        q_end = quantile_forecast[horizon_end]

        median_price = q_end[5]
        low_10 = q_end[1]
        high_90 = q_end[9]

        upside = (high_90 - current_price) / current_price
        downside = (current_price - low_10) / current_price
        median_return = (median_price - current_price) / current_price
        uncertainty = (high_90 - low_10) / current_price
        risk_reward = upside / max(downside, 0.001)

        if (downside < adj_risk and risk_reward > 2.0
                and median_return > 0.005 and uncertainty < 0.15):
            strength = min(risk_reward / 4.0, 1.0)
            return (SIGNAL_BUY, strength,
                    f"RR={risk_reward:.1f}, down={downside:.2%}, macro={macro:+.2f}")

        short_upside = (current_price - low_10) / current_price
        short_downside = (high_90 - current_price) / current_price
        short_rr = short_upside / max(short_downside, 0.001)

        # 悲观宏观下更容易触发做空
        adj_risk_short = self.risk_threshold * (1 - macro * self.macro_influence)
        if (short_downside < adj_risk_short and short_rr > 2.0
                and median_return < -0.005 and uncertainty < 0.15):
            strength = min(short_rr / 4.0, 1.0)
            return (SIGNAL_SELL, strength,
                    f"Short RR={short_rr:.1f}, median={median_return:.2%}, macro={macro:+.2f}")

        return SIGNAL_HOLD, 0.0, f"风控未通过: RR={risk_reward:.1f}, down={downside:.2%}"


class EnsembleStrategy(BaseStrategy):
    """
    策略 3：多 Horizon 集成策略

    宏观影响:
    - 宏观评分作为额外投票者，直接加减到技术确认分上
    """

    def __init__(self):
        super().__init__("多Horizon集成策略")
        self.horizons = config.STRATEGY_CONFIG["ensemble_horizons"]
        self.weights = config.STRATEGY_CONFIG["ensemble_weights"]

    def generate_signal(self, current_price, point_forecast, quantile_forecast, features_snapshot):
        if point_forecast is None or len(point_forecast) == 0:
            return SIGNAL_HOLD, 0.0, "无预测数据"

        macro = _safe_macro(features_snapshot)

        votes = []
        for i, h in enumerate(self.horizons):
            if h <= len(point_forecast):
                sub_forecast = point_forecast[:h]
                sub_return = (np.mean(sub_forecast) - current_price) / current_price
                if sub_return > 0.005:
                    votes.append((SIGNAL_BUY, self.weights[i], h, sub_return))
                elif sub_return < -0.005:
                    votes.append((SIGNAL_SELL, self.weights[i], h, sub_return))
                else:
                    votes.append((SIGNAL_HOLD, self.weights[i], h, sub_return))

        if not votes:
            return SIGNAL_HOLD, 0.0, "无有效投票"

        buy_weight = sum(w for s, w, h, r in votes if s == SIGNAL_BUY)
        sell_weight = sum(w for s, w, h, r in votes if s == SIGNAL_SELL)

        # 技术指标确认
        macd_hist = features_snapshot.get("macd_hist", 0)
        bb_upper = features_snapshot.get("bb_upper", current_price * 1.1)
        bb_lower = features_snapshot.get("bb_lower", current_price * 0.9)
        for val_name in ("macd_hist", "bb_upper", "bb_lower"):
            v = features_snapshot.get(val_name)
            if v is not None and isinstance(v, float) and np.isnan(v):
                if val_name == "macd_hist":
                    macd_hist = 0
                elif val_name == "bb_upper":
                    bb_upper = current_price * 1.1
                else:
                    bb_lower = current_price * 0.9

        tech_boost = 0.0
        if macd_hist > 0:
            tech_boost += 0.1
        elif macd_hist < 0:
            tech_boost -= 0.1
        if current_price < bb_lower:
            tech_boost += 0.1
        elif current_price > bb_upper:
            tech_boost -= 0.1

        # 宏观因子作为额外投票
        tech_boost += macro * self.macro_influence * 0.5

        total_weight = buy_weight + sell_weight + sum(
            w for s, w, h, r in votes if s == SIGNAL_HOLD)
        if total_weight == 0:
            return SIGNAL_HOLD, 0.0, "总权重为零"

        buy_ratio = buy_weight / total_weight + tech_boost
        sell_ratio = sell_weight / total_weight - tech_boost

        if buy_ratio > 0.55:
            strength = min(buy_ratio, 1.0)
            detail = ", ".join([f"H{h}:{r:.2%}" for s, w, h, r in votes])
            return (SIGNAL_BUY, strength,
                    f"集成看多({buy_ratio:.0%}): {detail}, macro={macro:+.2f}")
        elif sell_ratio > 0.55:
            strength = min(sell_ratio, 1.0)
            detail = ", ".join([f"H{h}:{r:.2%}" for s, w, h, r in votes])
            return (SIGNAL_SELL, strength,
                    f"集成看空({sell_ratio:.0%}): {detail}, macro={macro:+.2f}")
        else:
            return SIGNAL_HOLD, 0.0, f"集成未达一致: buy={buy_ratio:.0%}, sell={sell_ratio:.0%}"


def create_strategies():
    """创建所有策略实例"""
    return [
        TrendStrategy(),
        QuantileRiskStrategy(),
        EnsembleStrategy(),
    ]
