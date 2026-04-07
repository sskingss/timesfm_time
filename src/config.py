"""
全局配置模块
============
配置加载优先级:
  1. settings.toml（项目根目录，不提交到 git）
  2. 代码内默认值（兜底）

首次使用:
  cp settings.example.toml settings.toml
  # 编辑 settings.toml 填入你的参数
"""
from __future__ import annotations

import os
import tomllib
from typing import Any

# ── 定位配置文件 ──────────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SETTINGS_PATH = os.path.join(_PROJECT_ROOT, "settings.toml")


def _load_toml() -> dict[str, Any]:
    if not os.path.exists(_SETTINGS_PATH):
        return {}
    try:
        with open(_SETTINGS_PATH, "rb") as f:
            data = tomllib.load(f)
        print(f"[CONFIG] Loaded: {_SETTINGS_PATH}")
        return data
    except Exception as e:
        print(f"[CONFIG] Failed to load {_SETTINGS_PATH}: {e}, using defaults")
        return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并，override 中的值覆盖 base，嵌套 dict 递归处理"""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


_TOML = _load_toml()


def _section(key: str, defaults: dict) -> dict:
    """从 TOML 中取 [key] 段并与 defaults 合并"""
    return _deep_merge(defaults, _TOML.get(key, {}))


# ── 默认值 + TOML 覆盖 ───────────────────────────────────────

MODEL_CONFIG = _section("model", {
    "use_real_timesfm": True,
    "hf_repo_id": "google/timesfm-2.5-200m-pytorch",
    "context_length": 200,
    "horizon": 3,
    "max_context": 1024,
    "max_horizon": 128,
    "normalize_inputs": True,
    "use_continuous_quantile_head": True,
    "force_flip_invariance": True,
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
})

TRADING_CONFIG = _section("trading", {
    "initial_capital": 1_000_000,
    "commission_rate": 0.00075,
    "slippage": 0.001,
    "position_size": 0.7,
    "stop_loss": 0.08,
    "take_profit": 0.20,
    "max_positions": 1,
})

BACKTEST_CONFIG = _section("backtest", {
    "start_date": "2022-01-01",
    "end_date": "2025-12-31",
    "rebalance_freq": 2,
    "warmup_period": 150,
})

DATA_CONFIG = _section("data", {
    "symbols": ["601766"],
    "start_date": "2022-01-01",
    "end_date": "2025-12-31",
})

MACRO_CONFIG = _section("macro", {
    "enabled": True,
    "weights": {
        "vix": 0.35,
        "yield": 0.30,
        "momentum": 0.35,
    },
    "us_indicators": {
        "vix": "^VIX",
        "yield_10y": "^TNX",
        "market_index": "^GSPC",
    },
    "cn_indicators": {
        "market_index_ak": "sh000001",
        "market_index_yf": "000001.SS",
    },
})

STRATEGY_CONFIG = _section("strategy", {
    "trend_threshold": 0.005,
    "quantile_risk_threshold": 0.05,
    "ensemble_horizons": [2, 3, 5],
    "ensemble_weights": [0.3, 0.4, 0.3],
    "macro_influence": 0.15,
})

# ── 插件配置（含敏感信息，务必只写在 settings.toml 中）────────

_plugins_raw = _TOML.get("plugins", {})
PLUGIN_CONFIG = {
    "console": _deep_merge(
        {"enabled": True, "verbose": False},
        _plugins_raw.get("console", {}),
    ),
    "feishu": _deep_merge(
        {
            "enabled": False,
            "webhook_url": "",
            "secret": "",
            "app_id": "",
            "app_secret": "",
            "receive_id": "",
            "receive_id_type": "email",
            "use_card": True,
        },
        _plugins_raw.get("feishu", {}),
    ),
}

REPORT_CONFIG = _section("report", {
    "use_timesfm_screening": True,
    "stage1_pool_size": 50,
    "stage2_weight": 0.6,
    "stage1_weight": 0.4,
})

SIGNAL_CONFIG = _section("signal", {
    "actionable_only": False,
    "min_strength": 0.0,
})

_vis_raw = _section("visualization", {
    "output_dir": "output",
    "figsize": [14, 7],
    "dpi": 120,
    "style": "seaborn-v0_8-whitegrid",
})
VIS_CONFIG = {
    **_vis_raw,
    "figsize": tuple(_vis_raw["figsize"]),  # TOML 数组 → Python 元组
}
