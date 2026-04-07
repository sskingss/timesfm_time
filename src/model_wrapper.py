"""
TimesFM 模型封装模块
====================
封装 Google Research TimesFM 2.5 时间序列基础模型的推理接口。

模型架构概要（ICML 2024, Decoder-only Foundation Model for Time Series）:
────────────────────────────────────────────────────────────────────────
TimesFM 是一个 **patch 化的 decoder-only Transformer**:

  原始时序 [B, context_len]
       │  按 patch_len=32 切割
       ▼
  [B, N_patches, 32]  (200 个时间点 → ~6 个 patch)
       │  拼接 padding mask → [B, N, 64]
       │  ResidualBlock 投影 → [B, N, 1280]
       │  + PositionalEmbedding + FreqEmbedding
       ▼
  20 层 Transformer Decoder (causal self-attention, RMSNorm, SiLU-MLP)
       │  隐藏层维度 1280, 注意力头 16, head_dim 80
       ▼
  [B, N, 1280]  取最后一个 patch 的输出
       │  horizon_ff_layer (ResidualBlock) 投影
       ▼
  [B, horizon_len, 1+9]  输出: 1 个均值 + 9 个分位数 (10%~90%)
       │  反归一化还原到原始价格尺度
       ▼
  最终预测

快速推理的核心原因:
─────────────────
1. **Patch 压缩**: 200 个时间点仅产生 ~6 个 token (context/32),
   Transformer 自注意力复杂度从 O(200²)=40000 降到 O(6²)=36
2. **单步自回归**: horizon=3 < output_patch_len=128,
   一次前向传播就输出全部预测, 无需多步 decode
3. **纯推理模式**: torch.no_grad() + model.eval(), 不计算梯度
4. **单变量输入**: 仅输入收盘价一条序列, 输入维度极低
5. **轻量级模型**: 200M 参数 (20层×1280维), CPU 也能秒级推理

模块组成:
  - TimesFMPredictor: 真实模型调用 (需 pip install timesfm[torch])
  - SimulatedPredictor: 基于统计方法的 CPU 降级方案
  - MultiHorizonPredictor: 复用单次预测支持多 horizon 查询
  - create_predictor(): 工厂方法, 按 config 自动选择
"""
import numpy as np
from . import config


class TimesFMPredictor:
    """
    真实 TimesFM 预测器 —— 封装 Google TimesFM 2.5 (200M) PyTorch 版本

    内部推理流程 (对应 timesfm 库源码):
    ───────────────────────────────────
    1. forecast()                     [timesfm_base.py]
       ├─ 清洗 NaN / 线性插值
       ├─ 可选 normalize (z-score)
       └─ 调用 _forecast()
    2. _forecast()                    [timesfm_torch.py]
       ├─ _preprocess(): 截断到 context_len, padding 对齐到 batch
       ├─ torch.no_grad() 推理循环:
       │   └─ model.decode():         [pytorch_patched_decoder.py]
       │       ├─ 切 patch: [B, C] → [B, N, 32]
       │       ├─ _forward_transform: per-patch 归一化 (mu, sigma)
       │       ├─ input_ff_layer: [B, N, 64] → [B, N, 1280]
       │       ├─ + PositionalEmbedding + FreqEmbedding
       │       ├─ 20 层 StackedDecoder (causal attention + MLP)
       │       ├─ horizon_ff_layer: [B, N, 1280] → [B, N, H, 10]
       │       ├─ _reverse_transform: 反归一化
       │       └─ 自回归拼接 (通常仅 1 步, 因 H=3 << output_patch_len=128)
       └─ 返回 (mean_forecast, full_forecast)

    使用方法：
    1. pip install timesfm[torch]
    2. 设置 config.MODEL_CONFIG["use_real_timesfm"] = True
    """

    def __init__(self):
        import torch
        import timesfm

        # 使用 TF32 加速矩阵乘法 (A100/H100 GPU 上约 3x 提速)
        torch.set_float32_matmul_precision("high")

        # 从 HuggingFace Hub 下载并加载预训练权重
        # 模型参数: 20层 Transformer, 1280维隐藏层, 16头注意力, 总计 200M 参数
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            config.MODEL_CONFIG["hf_repo_id"]
        )

        # 编译推理配置:
        #   max_context=1024  — 最大可接受的历史窗口长度
        #   max_horizon=128   — 单次 decode 最大输出步数 (也是 output_patch_len)
        #   normalize_inputs  — 对输入做 z-score 归一化, 消除量纲影响
        #   use_continuous_quantile_head — 使用连续分位数回归头 (比离散 bin 更精确)
        #   force_flip_invariance — 翻转不变性: 对序列取负再预测, 取两次结果均值
        #                           消除模型对上升/下降趋势的系统性偏差
        #   infer_is_positive     — 推断输出是否为正 (如价格序列)
        #   fix_quantile_crossing — 修正分位数交叉 (确保 q10 < q50 < q90)
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
        使用 TimesFM 进行时序预测

        调用链: predict → model.forecast → _forecast → model.decode
        每次调用对一条价格序列做一次前向推理, 输出点预测和分位数预测。

        Args:
            history: np.ndarray, shape (T,), 历史收盘价序列
                     T ≤ context_length (默认 200), 超长会被截断到 max_context
            horizon: int, 预测未来几步 (默认 3, 即 3 个交易日)

        Returns:
            point_forecast: np.ndarray, shape (horizon,)
                预测的未来 horizon 天收盘价 (中位数或均值, 取决于 point_forecast_mode)
            quantile_forecast: np.ndarray, shape (horizon, 1+9)
                第 0 列为均值, 第 1~9 列为 10%~90% 分位数预测
                用于衡量预测的不确定性区间

        性能参考:
            context=200, horizon=3 → ~6 个 patch token → 1 步 decode
            GPU: ~50-100ms | CPU: ~0.5-2s
        """
        # model.forecast 接受 list[array], 支持批量推理
        # 这里传入单条序列 [history], 返回结果取 [0] 即第一条
        point_forecast, quantile_forecast = self.model.forecast(
            horizon=horizon,
            inputs=[history],
        )
        return point_forecast[0], quantile_forecast[0]

    def predict_batch(self, histories, horizon):
        """
        批量预测: 一次前向传播处理多条序列。

        Args:
            histories: list[np.ndarray], 每条为 shape (T,) 的收盘价序列
            horizon: int, 预测步长

        Returns:
            point_forecasts: list[np.ndarray], 每条 shape (horizon,)
            quantile_forecasts: list[np.ndarray], 每条 shape (horizon, 10)
        """
        point_forecasts, quantile_forecasts = self.model.forecast(
            horizon=horizon,
            inputs=histories,
        )
        return (
            [point_forecasts[i] for i in range(len(histories))],
            [quantile_forecasts[i] for i in range(len(histories))],
        )


class SimulatedPredictor:
    """
    模拟预测器 —— 无 GPU 环境下的降级方案
    用统计方法近似 TimesFM 的预测行为, 用于回测框架的功能验证。

    预测公式:
    ─────────
    daily_return = trend_weight × 加权趋势
                 + mean_revert_weight × 均值回复信号
                 + (1 - tw - mrw) × 短期动量

    point_forecast[t] = last_price × exp(Σ daily_return × decay^t)

    分位数通过历史波动率 × √t × noise_scale 生成正态分布区间。

    注意: 这不是真正的 foundation model, 预测精度远不如 TimesFM,
    仅用于验证回测框架在无 GPU 环境下能正常运行。
    """

    def __init__(self, trend_weight=0.4, mean_revert_weight=0.3, noise_scale=0.2):
        self.trend_weight = trend_weight          # 趋势跟随权重
        self.mean_revert_weight = mean_revert_weight  # 均值回归权重
        self.noise_scale = noise_scale            # 分位数散度缩放因子
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

        # 1. 趋势成分: 最近 60 天日收益率, 用指数衰减权重加权
        #    近期收益权重大, 远期权重小, 捕捉近期趋势方向
        lookback = min(60, n - 1)
        returns = np.diff(history[-lookback - 1:]) / history[-lookback - 1:-1]
        weights = np.exp(np.linspace(-2, 0, len(returns)))
        weights /= weights.sum()
        trend_return = np.sum(returns * weights)

        # 2. 均值回复成分: 当前价偏离长期均线 → 预期回归
        long_ma = np.mean(history[-min(200, n):])
        revert_signal = (long_ma - last_price) / last_price

        # 3. 动量因子: 10 日均线 vs 200 日均线的偏离度
        short_ma = np.mean(history[-min(10, n):])
        momentum = (short_ma - long_ma) / long_ma

        # 4. 三因子加权组合得到预测日收益率
        daily_return = (
            self.trend_weight * trend_return +
            self.mean_revert_weight * revert_signal +
            (1 - self.trend_weight - self.mean_revert_weight) * momentum
        )

        # 5. 历史波动率 (用于分位数区间宽度)
        hist_vol = np.std(returns) if len(returns) > 1 else 0.02

        # 6. 生成点预测路径: 收益率随预测步数指数衰减, 避免远期过度外推
        point_forecast = np.zeros(horizon)
        cumulative_return = 0.0
        for t in range(horizon):
            decay = np.exp(-0.05 * t)
            step_return = daily_return * decay
            cumulative_return += step_return
            point_forecast[t] = last_price * np.exp(cumulative_return)

        # 7. 生成分位数预测: 基于正态分布 N(0, σ√t) 展开不确定性锥
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        from scipy.stats import norm as _norm

        quantile_forecast = np.zeros((horizon, 10))
        quantile_forecast[:, 0] = point_forecast  # 第 0 列 = mean

        for t in range(horizon):
            # 波动率随时间√t增长 (布朗运动假设)
            t_vol = hist_vol * np.sqrt(t + 1) * self.noise_scale
            for qi, q in enumerate(quantile_levels):
                z = _norm_ppf(q)
                quantile_forecast[t, qi + 1] = point_forecast[t] * np.exp(z * t_vol)

        return point_forecast, quantile_forecast

    def predict_batch(self, histories, horizon):
        """批量模拟预测, 逐条调用 predict()"""
        points, quantiles = [], []
        for h in histories:
            p, q = self.predict(h, horizon)
            points.append(p)
            quantiles.append(q)
        return points, quantiles


def _norm_ppf(q):
    """
    正态分布分位数函数 (Percent Point Function) 的有理近似。
    避免在 SimulatedPredictor 中硬依赖 scipy。

    使用 Abramowitz & Stegun (1964) 的有理逼近公式:
      t = √(-2 ln(min(q, 1-q)))
      ppf ≈ t - (c0 + c1*t + c2*t²) / (1 + d1*t + d2*t² + d3*t³)

    精度: |误差| < 4.5e-4, 对分位数预测足够。
    """
    if q <= 0:
        return -5.0
    if q >= 1:
        return 5.0
    if q == 0.5:
        return 0.0

    t = np.sqrt(-2 * np.log(min(q, 1 - q)))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    result = t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)
    if q < 0.5:
        return -result
    return result


class MultiHorizonPredictor:
    """
    多 horizon 预测器 —— 用于 EnsembleStrategy

    优化策略: 只调用一次 predict(max_horizon), 然后对结果切片,
    避免对 [2, 3, 5] 三个 horizon 分别调用模型 (节省 2/3 推理时间)。
    """

    def __init__(self, base_predictor):
        self.predictor = base_predictor

    def predict_multi(self, history, horizons):
        """
        Args:
            history: np.ndarray, 历史价格序列
            horizons: list[int], 如 [2, 3, 5]

        Returns:
            dict: {horizon: (point_forecast[:h], quantile_forecast[:h])}
        """
        results = {}
        max_h = max(horizons)
        # 只做一次推理, 取 max(horizons) 步
        point, quantile = self.predictor.predict(history, max_h)
        for h in horizons:
            results[h] = (point[:h], quantile[:h])
        return results


def create_predictor():
    """
    工厂方法: 根据 MODEL_CONFIG 配置自动选择预测器。

    优先级: TimesFMPredictor (真实模型) → SimulatedPredictor (降级方案)
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
