"""
数据加载模块
- 美股真实行情（yfinance）
- A 股真实行情（akshare）
- 自动识别市场并路由
- 技术指标计算
- 本地缓存（按 股票代码+日期区间 去重，避免重复下载）
"""
import os
import pickle
import time
import numpy as np
import pandas as pd
from . import config

_MAX_RETRIES = 3
_RETRY_DELAY = 2
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "market_data")


# ===================== 缓存工具 =====================

def _cache_key(symbol: str, start_date: str, end_date: str) -> str:
    """生成缓存文件名: {symbol}_{start}_{end}.pkl"""
    clean = _clean_cn_code(symbol) if symbol[0].isdigit() else symbol
    return f"{clean}_{start_date}_{end_date}.pkl"


def _load_cache(symbol: str, start_date: str, end_date: str) -> dict | None:
    """尝试从本地缓存读取数据，未命中返回 None"""
    path = os.path.join(_CACHE_DIR, _cache_key(symbol, start_date, end_date))
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"[CACHE] Hit: {symbol} ({start_date} ~ {end_date})")
        return data
    except Exception as e:
        print(f"[CACHE] Read failed ({path}): {e}")
        return None


def _save_cache(symbol: str, start_date: str, end_date: str, data: dict):
    """将下载的行情数据写入本地缓存"""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, _cache_key(symbol, start_date, end_date))
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[CACHE] Saved: {path}")
    except Exception as e:
        print(f"[CACHE] Write failed ({path}): {e}")


# ===================== 市场识别 =====================

def detect_market(symbol: str) -> str:
    """
    根据股票代码格式自动判断市场。

    美股 — 纯字母（如 AAPL, TSLA）
    A 股 — 6 位数字，可带 .SH / .SZ 后缀（如 600519, 000858.SZ）
    """
    clean = symbol.replace(".SH", "").replace(".SZ", "").replace(".SS", "")
    if clean.isdigit() and len(clean) == 6:
        return "CN"
    return "US"


def _clean_cn_code(symbol: str) -> str:
    """去掉 A 股代码的交易所后缀，只保留 6 位数字"""
    return symbol.replace(".SH", "").replace(".SZ", "").replace(".SS", "")


# ===================== 美股数据 =====================

def download_us_data(symbol: str, start_date: str, end_date: str) -> dict:
    """通过 yfinance 下载美股 OHLCV 数据。"""
    import yfinance as yf

    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError(f"yfinance 未返回 {symbol} 在 {start_date}~{end_date} 的数据，请检查代码或网络")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df.dropna(subset=["Close"])

    return {
        "symbol": symbol,
        "market": "US",
        "currency": "USD",
        "dates": np.arange(len(df)),
        "date_strs": df.index.strftime("%Y-%m-%d").tolist(),
        "open": df["Open"].values.astype(float),
        "high": df["High"].values.astype(float),
        "low": df["Low"].values.astype(float),
        "close": df["Close"].values.astype(float),
        "volume": df["Volume"].values.astype(float),
    }


# ===================== A 股数据 =====================

def download_cn_data(symbol: str, start_date: str, end_date: str) -> dict:
    """
    通过 akshare 下载 A 股日线数据（前复权）。

    akshare API: stock_zh_a_hist(symbol, period, start_date, end_date, adjust)
    返回列: 日期, 开盘, 收盘, 最高, 最低, 成交量, ...
    """
    import akshare as ak

    code = _clean_cn_code(symbol)
    sd = start_date.replace("-", "")
    ed = end_date.replace("-", "")

    df = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=sd,
                end_date=ed,
                adjust="qfq",
            )
            break
        except Exception as e:
            print(f"[DATA] akshare attempt {attempt}/{_MAX_RETRIES} failed: {e}")
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY * attempt)

    if df is None or df.empty:
        raise ValueError(f"akshare 未返回 {symbol}({code}) 在 {start_date}~{end_date} 的数据")

    df = df.dropna(subset=["收盘"])

    return {
        "symbol": symbol,
        "market": "CN",
        "currency": "CNY",
        "dates": np.arange(len(df)),
        "date_strs": df["日期"].astype(str).tolist(),
        "open": df["开盘"].values.astype(float),
        "high": df["最高"].values.astype(float),
        "low": df["最低"].values.astype(float),
        "close": df["收盘"].values.astype(float),
        "volume": df["成交量"].values.astype(float),
    }


# ===================== 技术指标计算 =====================

def compute_ma(close, window):
    """简单移动平均线"""
    ma = np.full_like(close, np.nan)
    for i in range(window - 1, len(close)):
        ma[i] = np.mean(close[i - window + 1: i + 1])
    return ma


def compute_ema(close, window):
    """指数移动平均线"""
    ema = np.full_like(close, np.nan)
    k = 2.0 / (window + 1)
    ema[window - 1] = np.mean(close[:window])
    for i in range(window, len(close)):
        ema[i] = close[i] * k + ema[i - 1] * (1 - k)
    return ema


def compute_rsi(close, window=14):
    """相对强弱指标 RSI"""
    rsi = np.full_like(close, np.nan)
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])

    for i in range(window, len(deltas)):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def compute_macd(close, fast=12, slow=26, signal=9):
    """MACD 指标"""
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(close, window=20, num_std=2):
    """布林带"""
    ma = compute_ma(close, window)
    std = np.full_like(close, np.nan)
    for i in range(window - 1, len(close)):
        std[i] = np.std(close[i - window + 1: i + 1])
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, ma, lower


def compute_atr(high, low, close, window=14):
    """平均真实波幅 ATR"""
    atr = np.full_like(close, np.nan)
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    for i in range(window - 1, len(close)):
        atr[i] = np.mean(tr[i - window + 1: i + 1])
    return atr


def compute_features(data):
    """计算完整技术指标特征集"""
    close = data["close"]
    high = data["high"]
    low = data["low"]

    features = {
        "close": close,
        "returns": np.concatenate([[0], np.diff(close) / close[:-1]]),
        "log_returns": np.concatenate([[0], np.diff(np.log(close))]),
        "ma5": compute_ma(close, 5),
        "ma20": compute_ma(close, 20),
        "ma60": compute_ma(close, 60),
        "ema12": compute_ema(close, 12),
        "ema26": compute_ema(close, 26),
        "rsi": compute_rsi(close, 14),
        "bb_upper": compute_bollinger_bands(close)[0],
        "bb_mid": compute_bollinger_bands(close)[1],
        "bb_lower": compute_bollinger_bands(close)[2],
        "atr": compute_atr(high, low, close),
    }

    macd_line, signal_line, histogram = compute_macd(close)
    features["macd"] = macd_line
    features["macd_signal"] = signal_line
    features["macd_hist"] = histogram

    return features


# ===================== 统一入口 =====================

def load_data(symbols=None):
    """
    加载所有股票的真实行情数据并计算技术指标。

    自动识别市场：字母代码 → 美股（yfinance），6 位数字 → A 股（akshare）。
    """
    if symbols is None:
        symbols = config.DATA_CONFIG["symbols"]

    start_date = config.DATA_CONFIG["start_date"]
    end_date = config.DATA_CONFIG["end_date"]

    all_data = {}
    for sym in symbols:
        market = detect_market(sym)
        cur = "CNY" if market == "CN" else "USD"
        source = "akshare" if market == "CN" else "yfinance"

        cached = _load_cache(sym, start_date, end_date)
        if cached is not None:
            raw = cached
        else:
            print(f"[DATA] Downloading {sym} ({market}) from {source} ({start_date} ~ {end_date})...")
            if market == "CN":
                raw = download_cn_data(sym, start_date, end_date)
            else:
                raw = download_us_data(sym, start_date, end_date)
            _save_cache(sym, start_date, end_date, raw)

        features = compute_features(raw)
        all_data[sym] = {"raw": raw, "features": features}

        n = len(raw["close"])
        lo, hi = raw["close"].min(), raw["close"].max()
        print(f"[DATA] {sym}: {n} trading days, price range [{lo:.2f}, {hi:.2f}] {cur}")

    return all_data
