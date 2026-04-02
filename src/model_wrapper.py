"""
TimesFM 模型封装模块
- 真实 TimesFM 模型调用（需 GPU 环境）
- 模拟预测器（基于统计方法模拟 TimesFM 行为）
- 统一推理接口
"""
import numpy as np
from . import config


class TimesFMPredictor:
    """
    真实 TimesFM 预测器
    需要安装 timesfm 包 + GPU 环境

    使用方法：
    1. pip install timesfm[torch]
    2. 设置 config.MODEL_CONFIG["use_real_timesfm"] = True
    """

    def __init__(self):
        import torch
        import timesfm

        torch.set_float32_matmul_precision("high")

        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            config.MODEL_CONFIG["hf_repo_id"]
        )
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=config.MODEL_CONFIG["max_context"],
                max_horizon=config.MODEL_CONFIG["max_horizon"],
                normalize_inputs=config.MODEL_CONFIG["normalize_inputs"],
                use_continuous_quantile_head=config.MODEL_CONFIG["use_continuous_quantile_head"],
                force_flip_invariance=config.MODEL_CONFIG["force_flip_invariance"],
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        print("[MODEL] TimesFM 2.5 loaded successfully (GPU)")

    def predict(self, history, horizon):
        """
        使用 TimesFM 进行预测

        Args:
            history: np.ndarray, 历史价格序列
            horizon: int, 预测步长

        Returns:
            point_forecast: np.ndarray, shape (horizon,), 点预测
            quantile_forecast: np.ndarray, shape (horizon, 10), 分位数预测
        """
        point_forecast, quantile_forecast = self.model.forecast(
            horizon=horizon,
            inputs=[history],
        )
        return point_forecast[0], quantile_forecast[0]


class SimulatedPredictor:
    """
    模拟预测器
    使用统计方法模拟 TimesFM 的预测行为，用于无 GPU 环境的回测演示。

    原理：
    1. 使用历史数据的加权趋势外推作为 point forecast
    2. 使用历史波动率生成分位数区间
    3. 加入均值回复倾向和动量因子以模拟 foundation model 的行为
    """

    def __init__(self, trend_weight=0.4, mean_revert_weight=0.3, noise_scale=0.2):
        self.trend_weight = trend_weight
        self.mean_revert_weight = mean_revert_weight
        self.noise_scale = noise_scale
        print("[MODEL] Simulated predictor initialized (CPU mode)")

    def predict(self, history, horizon):
        """
        模拟 TimesFM 预测

        Args:
            history: np.ndarray, 历史价格序列
            horizon: int, 预测步长

        Returns:
            point_forecast: np.ndarray, shape (horizon,), 点预测
            quantile_forecast: np.ndarray, shape (horizon, 10), 分位数预测
                channels: [mean, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%]
        """
        n = len(history)
        last_price = history[-1]

        # 1. 趋势成分：加权最近收益率（指数衰减权重）
        lookback = min(60, n - 1)
        returns = np.diff(history[-lookback - 1:]) / history[-lookback - 1:-1]
        weights = np.exp(np.linspace(-2, 0, len(returns)))
        weights /= weights.sum()
        trend_return = np.sum(returns * weights)

        # 2. 均值回复成分
        long_ma = np.mean(history[-min(200, n):])
        revert_signal = (long_ma - last_price) / last_price

        # 3. 动量因子
        short_ma = np.mean(history[-min(10, n):])
        momentum = (short_ma - long_ma) / long_ma

        # 4. 组合预测日收益率
        daily_return = (
            self.trend_weight * trend_return +
            self.mean_revert_weight * revert_signal +
            (1 - self.trend_weight - self.mean_revert_weight) * momentum
        )

        # 5. 历史波动率
        hist_vol = np.std(returns) if len(returns) > 1 else 0.02

        # 6. 生成点预测路径
        point_forecast = np.zeros(horizon)
        cumulative_return = 0.0
        for t in range(horizon):
            # 收益率随预测步数衰减
            decay = np.exp(-0.05 * t)
            step_return = daily_return * decay
            cumulative_return += step_return
            point_forecast[t] = last_price * np.exp(cumulative_return)

        # 7. 生成分位数预测
        # channels: [mean, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%]
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        from scipy.stats import norm as _norm

        quantile_forecast = np.zeros((horizon, 10))
        quantile_forecast[:, 0] = point_forecast  # mean

        for t in range(horizon):
            t_vol = hist_vol * np.sqrt(t + 1) * self.noise_scale
            for qi, q in enumerate(quantile_levels):
                z = _norm_ppf(q)
                quantile_forecast[t, qi + 1] = point_forecast[t] * np.exp(z * t_vol)

        return point_forecast, quantile_forecast


def _norm_ppf(q):
    """正态分布分位数函数的近似（避免依赖 scipy）"""
    # Beasley-Springer-Moro 算法的简化近似
    if q <= 0:
        return -5.0
    if q >= 1:
        return 5.0
    if q == 0.5:
        return 0.0

    # 使用有理近似
    t = np.sqrt(-2 * np.log(min(q, 1 - q)))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    result = t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)
    if q < 0.5:
        return -result
    return result


class MultiHorizonPredictor:
    """
    多 horizon 预测器
    对同一历史数据以多个不同 horizon 进行预测，用于集成策略
    """

    def __init__(self, base_predictor):
        self.predictor = base_predictor

    def predict_multi(self, history, horizons):
        """
        多 horizon 预测

        Args:
            history: np.ndarray
            horizons: list[int]

        Returns:
            dict: {horizon: (point_forecast, quantile_forecast)}
        """
        results = {}
        max_h = max(horizons)
        point, quantile = self.predictor.predict(history, max_h)
        for h in horizons:
            results[h] = (point[:h], quantile[:h])
        return results


def create_predictor():
    """
    工厂方法：根据配置创建预测器

    Returns:
        predictor: 预测器实例
    """
    if config.MODEL_CONFIG["use_real_timesfm"]:
        try:
            return TimesFMPredictor()
        except Exception as e:
            print(f"[WARN] Failed to load TimesFM: {e}")
            print("[WARN] Falling back to simulated predictor")
            return SimulatedPredictor()
    else:
        return SimulatedPredictor()
