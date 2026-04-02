"""
量化交易策略模块
================
实现三种基于 TimesFM 预测的交易策略，融合扩展因子库进行信号确认和过滤：

1. 趋势预测策略 - 基于点预测的趋势方向 + ADX 趋势强度确认 + 宏观因子调节阈值
2. 分位数风控策略 - 利用分位数预测做风险控制 + 波动率自适应 + 量价确认
3. 多 Horizon 集成策略 - 综合多个预测窗口加权投票 + 技术/量价/宏观多维投票

因子使用映射 (因子 → 策略):
────────────────────────────
  RSI, StochRSI, Williams%R  → TrendStrategy (超买超卖过滤)
  ADX                        → TrendStrategy (趋势强度置信)
  volatility, ATR            → QuantileRiskStrategy (自适应风控阈值)
  OBV, volume_ratio          → QuantileRiskStrategy / EnsembleStrategy (量价确认)
  MACD hist, 布林带           → EnsembleStrategy (技术确认投票)
  price_position             → EnsembleStrategy (位置风险评估)
  CCI                        → EnsembleStrategy (趋势转折确认)
  macro_score                → 全部策略 (宏观环境调节)
"""
import numpy as np
from . import config

SIGNAL_BUY = 1
SIGNAL_SELL = -1
SIGNAL_HOLD = 0


def _safe_feat(features_snapshot: dict, key: str, default=0.0) -> float:
    """从特征快照中安全读取数值，处理 None/NaN"""
    v = features_snapshot.get(key, default)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return float(v)


def _safe_macro(features_snapshot: dict) -> float:
    """从特征快照中安全读取宏观评分"""
    return _safe_feat(features_snapshot, "macro_score", 0.0)


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

        rsi = _safe_feat(features_snapshot, "rsi", 50)
        adx = _safe_feat(features_snapshot, "adx", 25)
        stoch_rsi = _safe_feat(features_snapshot, "stoch_rsi", 50)

        # ADX 趋势强度置信度: ADX > 25 趋势明确 → 信号更可信
        # ADX < 20 震荡市 → 提高阈值避免频繁交易
        adx_factor = 1.0
        if adx > 30:
            adx_factor = 0.8   # 强趋势: 降低阈值, 更容易触发
        elif adx < 20:
            adx_factor = 1.3   # 震荡市: 提高阈值, 减少假信号

        adj_buy_threshold = buy_threshold * adx_factor
        adj_sell_threshold = sell_threshold * adx_factor

        # StochRSI 增强超买超卖判断 (比 RSI 更灵敏)
        overbought = rsi > 75 or stoch_rsi > 80
        oversold = rsi < 25 or stoch_rsi < 20

        if combined_return > adj_buy_threshold and not overbought:
            strength = min(abs(combined_return) / self.threshold, 1.0)
            if adx > 25:
                strength = min(strength * 1.2, 1.0)
            return (SIGNAL_BUY, strength,
                    f"预测上涨 {combined_return:.2%}, RSI={rsi:.0f}, ADX={adx:.0f}, macro={macro:+.2f}")
        elif combined_return < -adj_sell_threshold and not oversold:
            strength = min(abs(combined_return) / self.threshold, 1.0)
            if adx > 25:
                strength = min(strength * 1.2, 1.0)
            return (SIGNAL_SELL, strength,
                    f"预测下跌 {combined_return:.2%}, RSI={rsi:.0f}, ADX={adx:.0f}, macro={macro:+.2f}")
        else:
            return SIGNAL_HOLD, 0.0, f"预测变化 {combined_return:.2%} 未达阈值 (ADX={adx:.0f})"


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
        vol = _safe_feat(features_snapshot, "volatility", 0.2)
        vol_ratio = _safe_feat(features_snapshot, "volume_ratio", 1.0)
        obv = _safe_feat(features_snapshot, "obv", 0)

        # 波动率自适应: 高波动期收紧风控, 低波动期适度放宽
        vol_adj = 1.0
        if vol > 0.4:
            vol_adj = 0.7    # 高波动: 更严格
        elif vol < 0.15:
            vol_adj = 1.2    # 低波动: 可放宽

        # 宏观 + 波动率联合调节风险阈值
        adj_risk = self.risk_threshold * (1 + macro * self.macro_influence) * vol_adj

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

        # 量价确认: 放量 (vol_ratio > 1.3) 增加信号可信度
        vol_boost = 0.0
        if vol_ratio > 1.5:
            vol_boost = 0.15
        elif vol_ratio > 1.2:
            vol_boost = 0.08

        if (downside < adj_risk and risk_reward > 2.0
                and median_return > 0.005 and uncertainty < 0.15):
            strength = min((risk_reward + vol_boost) / 4.0, 1.0)
            return (SIGNAL_BUY, strength,
                    f"RR={risk_reward:.1f}, down={downside:.2%}, vol={vol:.0%}, vratio={vol_ratio:.1f}, macro={macro:+.2f}")

        short_upside = (current_price - low_10) / current_price
        short_downside = (high_90 - current_price) / current_price
        short_rr = short_upside / max(short_downside, 0.001)

        adj_risk_short = self.risk_threshold * (1 - macro * self.macro_influence) * vol_adj
        if (short_downside < adj_risk_short and short_rr > 2.0
                and median_return < -0.005 and uncertainty < 0.15):
            strength = min((short_rr + vol_boost) / 4.0, 1.0)
            return (SIGNAL_SELL, strength,
                    f"Short RR={short_rr:.1f}, median={median_return:.2%}, vol={vol:.0%}, macro={macro:+.2f}")

        return SIGNAL_HOLD, 0.0, f"风控未通过: RR={risk_reward:.1f}, down={downside:.2%}, vol={vol:.0%}"


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

        # --- 第一维: 多 horizon 预测投票 ---
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

        # --- 第二维: 技术指标确认 (多因子综合) ---
        macd_hist = _safe_feat(features_snapshot, "macd_hist", 0)
        bb_upper = _safe_feat(features_snapshot, "bb_upper", current_price * 1.1)
        bb_lower = _safe_feat(features_snapshot, "bb_lower", current_price * 0.9)
        cci = _safe_feat(features_snapshot, "cci", 0)
        price_pos = _safe_feat(features_snapshot, "price_position", 0.5)
        vol_ratio = _safe_feat(features_snapshot, "volume_ratio", 1.0)
        adx = _safe_feat(features_snapshot, "adx", 25)

        tech_boost = 0.0

        # MACD 动能方向
        if macd_hist > 0:
            tech_boost += 0.08
        elif macd_hist < 0:
            tech_boost -= 0.08

        # 布林带位置
        if current_price < bb_lower:
            tech_boost += 0.08
        elif current_price > bb_upper:
            tech_boost -= 0.08

        # CCI 趋势转折确认
        if cci > 100:
            tech_boost += 0.05
        elif cci < -100:
            tech_boost -= 0.05

        # 量比确认: 放量方向更可信
        if vol_ratio > 1.3:
            tech_boost *= 1.3

        # 价格位置风险: 在顶部区域做多需更谨慎
        if price_pos > 0.9:
            tech_boost -= 0.05
        elif price_pos < 0.1:
            tech_boost += 0.05

        # ADX 趋势强度: 强趋势时 tech_boost 加权放大
        if adx > 30:
            tech_boost *= 1.2

        # --- 第三维: 宏观因子投票 ---
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
                    f"集成看多({buy_ratio:.0%}): {detail}, CCI={cci:.0f}, ADX={adx:.0f}, macro={macro:+.2f}")
        elif sell_ratio > 0.55:
            strength = min(sell_ratio, 1.0)
            detail = ", ".join([f"H{h}:{r:.2%}" for s, w, h, r in votes])
            return (SIGNAL_SELL, strength,
                    f"集成看空({sell_ratio:.0%}): {detail}, CCI={cci:.0f}, ADX={adx:.0f}, macro={macro:+.2f}")
        else:
            return SIGNAL_HOLD, 0.0, f"集成未达一致: buy={buy_ratio:.0%}, sell={sell_ratio:.0%}"


def create_strategies():
    """创建所有策略实例"""
    return [
        TrendStrategy(),
        QuantileRiskStrategy(),
        EnsembleStrategy(),
    ]
