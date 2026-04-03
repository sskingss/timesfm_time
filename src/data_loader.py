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
        print(f"[DATA] akshare failed for {symbol}, falling back to yfinance...")
        return _download_cn_via_yfinance(symbol, code, start_date, end_date)

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


def _download_cn_via_yfinance(symbol: str, code: str, start_date: str, end_date: str) -> dict:
    """akshare 不可用时，通过 yfinance 拉取 A 股数据（上交所 .SS / 深交所 .SZ）"""
    import yfinance as yf

    suffix = ".SS" if code.startswith(("6", "5", "9")) else ".SZ"
    yf_symbol = f"{code}{suffix}"

    print(f"[DATA] yfinance fallback: {yf_symbol} ({start_date} ~ {end_date})")
    df = yf.download(yf_symbol, start=start_date, end=end_date, progress=False)

    if df.empty:
        raise ValueError(f"yfinance 也未返回 {symbol} ({yf_symbol}) 的数据，请检查网络")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df.dropna(subset=["Close"])

    return {
        "symbol": symbol,
        "market": "CN",
        "currency": "CNY",
        "dates": np.arange(len(df)),
        "date_strs": df.index.strftime("%Y-%m-%d").tolist(),
        "open": df["Open"].values.astype(float),
        "high": df["High"].values.astype(float),
        "low": df["Low"].values.astype(float),
        "close": df["Close"].values.astype(float),
        "volume": df["Volume"].values.astype(float),
    }


# ===================== 基础技术指标计算 =====================
# 以下指标分为两类:
#   (A) 策略决策因子: 被 strategy.py 的三大策略直接引用
#   (B) 扩展因子: 供策略增强或外部分析使用

def compute_ma(close, window):
    """简单移动平均线 (SMA) — 等权重滑动窗口均值"""
    ma = np.full_like(close, np.nan)
    for i in range(window - 1, len(close)):
        ma[i] = np.mean(close[i - window + 1: i + 1])
    return ma


def compute_ema(close, window):
    """指数移动平均线 (EMA) — 近期数据权重指数递增, k = 2/(window+1)"""
    ema = np.full_like(close, np.nan)
    k = 2.0 / (window + 1)
    ema[window - 1] = np.mean(close[:window])
    for i in range(window, len(close)):
        ema[i] = close[i] * k + ema[i - 1] * (1 - k)
    return ema


def compute_rsi(close, window=14):
    """
    RSI (Relative Strength Index) — Wilder 平滑法
    RSI = 100 - 100/(1 + RS),  RS = AvgGain / AvgLoss
    >70 超买, <30 超卖 (策略中用 75/25 作阈值)
    """
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
    """
    MACD — 趋势动量指标
    macd_line = EMA(fast) - EMA(slow)
    signal_line = EMA(macd_line, signal)
    histogram = macd_line - signal_line  (>0 多头动能, <0 空头动能)
    """
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(close, window=20, num_std=2):
    """
    布林带 — 均值 ± 2σ 通道
    价格触及上轨可能回落, 触及下轨可能反弹
    EnsembleStrategy 中用于 tech_boost 加减分
    """
    ma = compute_ma(close, window)
    std = np.full_like(close, np.nan)
    for i in range(window - 1, len(close)):
        std[i] = np.std(close[i - window + 1: i + 1])
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, ma, lower


def compute_atr(high, low, close, window=14):
    """
    ATR (Average True Range) — 平均真实波幅
    TR = max(H-L, |H-C_prev|, |L-C_prev|)
    ATR = SMA(TR, window)
    衡量市场波动强度, 用于动态止损和仓位管理
    """
    atr = np.full_like(close, np.nan)
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    for i in range(window - 1, len(close)):
        atr[i] = np.mean(tr[i - window + 1: i + 1])
    return atr


# ===================== 扩展因子计算 =====================

def compute_volatility(close, window=20):
    """
    历史波动率 (年化) — σ_annual = std(日收益率) × √252
    高波动 → 降低仓位; 低波动 → 可加仓 (波动率目标策略)
    适配 TimesFM: 波动率突然放大时模型预测可信度下降, 可据此调整策略阈值
    """
    vol = np.full_like(close, np.nan)
    returns = np.zeros(len(close))
    returns[1:] = np.diff(close) / close[:-1]
    for i in range(window, len(close)):
        vol[i] = np.std(returns[i - window + 1: i + 1]) * np.sqrt(252)
    return vol


def compute_obv(close, volume):
    """
    OBV (On-Balance Volume) — 能量潮
    价格上涨日累加成交量, 下跌日累减。
    OBV 趋势与价格趋势背离 → 可能反转。
    适配 TimesFM: OBV 可确认模型预测的趋势方向是否有资金面支撑。
    """
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    return obv


def compute_vwap(high, low, close, volume):
    """
    VWAP (Volume Weighted Average Price) — 成交量加权均价
    VWAP = Σ(典型价格 × 成交量) / Σ(成交量)
    典型价格 = (H + L + C) / 3
    机构常用基准: 价格 < VWAP → 偏空, 价格 > VWAP → 偏多
    """
    typical_price = (high + low + close) / 3.0
    cum_tp_vol = np.cumsum(typical_price * volume)
    cum_vol = np.cumsum(volume)
    cum_vol = np.where(cum_vol == 0, 1, cum_vol)
    return cum_tp_vol / cum_vol


def compute_stoch_rsi(close, rsi_period=14, stoch_period=14, k_smooth=3):
    """
    StochRSI — RSI 的随机指标化
    StochRSI = (RSI - min(RSI, N)) / (max(RSI, N) - min(RSI, N))
    比 RSI 更灵敏, 适合短线捕捉超买超卖。
    适配 TimesFM: 模型预测上涨 + StochRSI 从超卖区回升 → 高确信度做多信号
    """
    rsi = compute_rsi(close, rsi_period)
    stoch_rsi = np.full_like(close, np.nan)
    for i in range(rsi_period + stoch_period - 1, len(close)):
        window_rsi = rsi[i - stoch_period + 1: i + 1]
        valid = window_rsi[~np.isnan(window_rsi)]
        if len(valid) < 2:
            continue
        rsi_min = np.min(valid)
        rsi_max = np.max(valid)
        if rsi_max - rsi_min < 1e-10:
            stoch_rsi[i] = 50.0
        else:
            stoch_rsi[i] = (rsi[i] - rsi_min) / (rsi_max - rsi_min) * 100.0

    # K 线平滑
    k = np.full_like(close, np.nan)
    for i in range(k_smooth - 1, len(close)):
        window = stoch_rsi[i - k_smooth + 1: i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            k[i] = np.mean(valid)
    return k


def compute_williams_r(high, low, close, window=14):
    """
    Williams %R — 超买超卖振荡器
    %R = (最高价 - 收盘价) / (最高价 - 最低价) × (-100)
    范围 [-100, 0]: < -80 超卖, > -20 超买
    与 RSI 互补, 对价格突破更敏感
    """
    wr = np.full_like(close, np.nan)
    for i in range(window - 1, len(close)):
        hh = np.max(high[i - window + 1: i + 1])
        ll = np.min(low[i - window + 1: i + 1])
        if hh - ll < 1e-10:
            wr[i] = -50.0
        else:
            wr[i] = (hh - close[i]) / (hh - ll) * (-100.0)
    return wr


def compute_cci(high, low, close, window=20):
    """
    CCI (Commodity Channel Index) — 顺势指标
    CCI = (TP - SMA(TP)) / (0.015 × MeanDeviation)
    >100 超买/强势, <-100 超卖/弱势
    对趋势转折点敏感, 适合辅助 TimesFM 的趋势预测确认
    """
    tp = (high + low + close) / 3.0
    cci = np.full_like(close, np.nan)
    for i in range(window - 1, len(close)):
        tp_window = tp[i - window + 1: i + 1]
        tp_ma = np.mean(tp_window)
        mean_dev = np.mean(np.abs(tp_window - tp_ma))
        if mean_dev < 1e-10:
            cci[i] = 0.0
        else:
            cci[i] = (tp[i] - tp_ma) / (0.015 * mean_dev)
    return cci


def compute_adx(high, low, close, window=14):
    """
    ADX (Average Directional Index) — 趋势强度指标
    ADX > 25 表示存在明显趋势, < 20 表示震荡盘整。
    不区分方向, 仅衡量趋势强度。
    适配 TimesFM: ADX 高时模型的趋势预测更可信, ADX 低时应降低仓位或切换均值回归策略。
    """
    n = len(close)
    adx = np.full(n, np.nan)
    if n < window + 1:
        return adx

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    # Wilder 平滑
    atr_smooth = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)

    atr_smooth[window] = np.sum(tr[1:window + 1])
    plus_smooth = np.sum(plus_dm[1:window + 1])
    minus_smooth = np.sum(minus_dm[1:window + 1])

    for i in range(window + 1, n):
        atr_smooth[i] = atr_smooth[i - 1] - atr_smooth[i - 1] / window + tr[i]
        plus_smooth = plus_smooth - plus_smooth / window + plus_dm[i]
        minus_smooth = minus_smooth - minus_smooth / window + minus_dm[i]
        if atr_smooth[i] > 0:
            plus_di[i] = 100.0 * plus_smooth / atr_smooth[i]
            minus_di[i] = 100.0 * minus_smooth / atr_smooth[i]

    dx = np.zeros(n)
    for i in range(window, n):
        denom = plus_di[i] + minus_di[i]
        if denom > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / denom

    for i in range(2 * window, n):
        adx[i] = np.mean(dx[i - window + 1: i + 1])

    return adx


def compute_volume_ratio(volume, window=20):
    """
    量比 = 当日成交量 / MA(成交量, window)
    > 1.5 放量, < 0.7 缩量
    放量突破更可信; 缩量上涨可能是假突破
    适配 TimesFM: 配合模型预测方向, 量比确认信号强度
    """
    vol_ratio = np.full_like(volume, np.nan)
    ma_vol = compute_ma(volume, window)
    for i in range(window - 1, len(volume)):
        if ma_vol[i] > 0:
            vol_ratio[i] = volume[i] / ma_vol[i]
    return vol_ratio


def compute_price_position(close, window=60):
    """
    价格位置 = (close - min(window)) / (max(window) - min(window))
    范围 [0, 1]: 接近 0 在底部, 接近 1 在顶部
    适配 TimesFM: 模型预测上涨但已在 0.9+ → 追高风险大
    """
    pos = np.full_like(close, np.nan)
    for i in range(window - 1, len(close)):
        hh = np.max(close[i - window + 1: i + 1])
        ll = np.min(close[i - window + 1: i + 1])
        if hh - ll < 1e-10:
            pos[i] = 0.5
        else:
            pos[i] = (close[i] - ll) / (hh - ll)
    return pos


def compute_ma_dispersion(close):
    """
    均线离散度 = std(MA5, MA20, MA60) / close
    均线粘合 (低离散) → 即将变盘; 均线发散 (高离散) → 趋势确立
    适配 TimesFM: 低离散度时模型预测的方向突破更有参考价值
    """
    ma5 = compute_ma(close, 5)
    ma20 = compute_ma(close, 20)
    ma60 = compute_ma(close, 60)
    disp = np.full_like(close, np.nan)
    for i in range(59, len(close)):
        vals = [ma5[i], ma20[i], ma60[i]]
        if all(not np.isnan(v) for v in vals) and close[i] > 0:
            disp[i] = np.std(vals) / close[i]
    return disp


def compute_features(data):
    """
    计算完整技术指标特征集

    特征分为三组:
    ┌──────────────────┬──────────────────────────────────────────────┐
    │ 基础价格特征     │ close, returns, log_returns                   │
    │ 均线族           │ ma5/20/60, ema12/26                           │
    │ 动量振荡器       │ rsi, stoch_rsi, williams_r, cci               │
    │ 趋势指标         │ macd/signal/hist, adx                         │
    │ 波动率通道       │ bb_upper/mid/lower, atr, volatility           │
    │ 量价因子         │ obv, vwap, volume_ratio                       │
    │ 综合位置因子     │ price_position, ma_dispersion                 │
    └──────────────────┴──────────────────────────────────────────────┘

    注意: 这些因子不直接喂给 TimesFM (模型只看 close 序列),
    而是在策略层 (strategy.py) 中用于:
      1. 过滤/确认模型预测信号
      2. 调节买卖阈值
      3. 衡量信号可信度
    """
    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    # --- 基础价格特征 ---
    features = {
        "close": close,
        "returns": np.concatenate([[0], np.diff(close) / close[:-1]]),
        "log_returns": np.concatenate([[0], np.diff(np.log(close))]),
    }

    # --- 均线族 (趋势过滤) ---
    features["ma5"] = compute_ma(close, 5)
    features["ma20"] = compute_ma(close, 20)
    features["ma60"] = compute_ma(close, 60)
    features["ema12"] = compute_ema(close, 12)
    features["ema26"] = compute_ema(close, 26)

    # --- 动量振荡器 (超买超卖) ---
    features["rsi"] = compute_rsi(close, 14)
    features["stoch_rsi"] = compute_stoch_rsi(close)
    features["williams_r"] = compute_williams_r(high, low, close)
    features["cci"] = compute_cci(high, low, close)

    # --- 趋势强度指标 ---
    macd_line, signal_line, histogram = compute_macd(close)
    features["macd"] = macd_line
    features["macd_signal"] = signal_line
    features["macd_hist"] = histogram
    features["adx"] = compute_adx(high, low, close)

    # --- 波动率通道 ---
    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(close)
    features["bb_upper"] = bb_upper
    features["bb_mid"] = bb_mid
    features["bb_lower"] = bb_lower
    features["atr"] = compute_atr(high, low, close)
    features["volatility"] = compute_volatility(close)

    # --- 量价因子 ---
    features["obv"] = compute_obv(close, volume)
    features["vwap"] = compute_vwap(high, low, close, volume)
    features["volume_ratio"] = compute_volume_ratio(volume)

    # --- 综合位置因子 ---
    features["price_position"] = compute_price_position(close)
    features["ma_dispersion"] = compute_ma_dispersion(close)

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
