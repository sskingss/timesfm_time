"""
全局配置模块
包含模型参数、交易参数、回测参数、宏观分析参数等
"""

# ============ 模型配置 ============
MODEL_CONFIG = {
    "use_real_timesfm": True,        # 使用真实 TimesFM 模型（需 GPU + pip install timesfm）
    "hf_repo_id": "google/timesfm-2.5-200m-pytorch",
    "context_length": 200,            # 历史窗口长度（原 512，缩短以减少滞后）
    "horizon": 3,                     # 预测步长（原 10，短线更准）
    "max_context": 1024,
    "max_horizon": 128,
    "normalize_inputs": True,
    "use_continuous_quantile_head": True,
    "force_flip_invariance": True,
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}

# ============ 交易配置 ============
TRADING_CONFIG = {
    "initial_capital": 1_000_000,     # 初始资金 100 万
    "commission_rate": 0.00075,       # A 股综合费率（佣金万2.5 + 印花税万5）
    "slippage": 0.001,                # 滑点 0.1%（A 股散户优先级低）
    "position_size": 0.7,             # 单次仓位 70%（原 30%，提高资金利用率）
    "stop_loss": 0.08,                # 止损 8%（原 5%，给更多容错）
    "take_profit": 0.20,              # 止盈 20%（原 10%，让利润充分奔跑）
    "max_positions": 1,               # 最大同时持仓数
}

# ============ 回测配置 ============
BACKTEST_CONFIG = {
    "start_date": "2022-01-01",
    "end_date": "2025-12-31",
    "rebalance_freq": 2,              # 每 2 天重新预测一次（原 5，提高反应速度）
    "warmup_period": 150,             # 预热期（原 512，释放更多交易日）
}

# ============ 数据配置 ============
# 股票代码格式：
#   美股  — 字母代码，如 "AAPL", "TSLA"
#   A 股  — 6 位数字代码，如 "600519"（茅台）, "000858"（五粮液）
#            也可带交易所后缀 "600519.SH", "000858.SZ"
DATA_CONFIG = {
    "symbols": ["601766"],       # 默认美股；可改为 ["600519", "000858"] 回测 A 股
    "start_date": "2022-01-01",
    "end_date": "2025-12-31",
}

# ============ 宏观分析配置 ============
MACRO_CONFIG = {
    "enabled": True,                   # 是否启用宏观因子
    "weights": {
        "vix": 0.35,                   # VIX 恐慌指数权重
        "yield": 0.30,                 # 国债收益率权重
        "momentum": 0.35,             # 市场指数动量权重
    },
    "us_indicators": {
        "vix": "^VIX",                 # CBOE 波动率指数
        "yield_10y": "^TNX",           # 美国 10 年期国债收益率
        "market_index": "^GSPC",       # 标普 500
    },
    "cn_indicators": {
        "market_index_ak": "sh000001", # 上证综指（akshare 格式）
        "market_index_yf": "000001.SS",# 上证综指（yfinance 格式）
    },
}

# ============ 策略配置 ============
STRATEGY_CONFIG = {
    "trend_threshold": 0.005,         # 趋势策略门槛（原 0.01，更灵敏）
    "quantile_risk_threshold": 0.05,  # 分位数策略风险阈值（原 0.03，适度放宽）
    "ensemble_horizons": [2, 3, 5],   # 集成策略窗口（原 [3,5,10]，更短线）
    "ensemble_weights": [0.3, 0.4, 0.3],
    "macro_influence": 0.15,          # 宏观调节幅度（原 0.3，降低宏观干扰）
}

# ============ 可视化配置 ============
VIS_CONFIG = {
    "output_dir": "output",
    "figsize": (14, 7),
    "dpi": 120,
    "style": "seaborn-v0_8-whitegrid",
}
