"""
宏观经济分析模块
获取宏观指标（VIX、国债收益率、市场指数），计算复合宏观环境评分，
作为交易策略的辅助决策因子。

评分区间 [-1, +1]:
  -1  极度悲观（高波动 + 利率急升 + 市场下跌）
   0  中性
  +1  极度乐观（低波动 + 宽松环境 + 市场上涨）
"""
import time
import numpy as np
import pandas as pd
from . import config
from .data_loader import detect_market

_MAX_RETRIES = 3
_RETRY_DELAY = 2


class MacroAnalyzer:
    """
    宏观环境评分器。

    根据目标股票所属市场（US / CN）抓取对应的宏观指标，
    计算逐日评分数组，注入到 data_dict[sym]["features"]["macro_score"] 中，
    使策略可直接通过 features_snapshot 读取。
    """

    def __init__(self):
        self.cfg = config.MACRO_CONFIG

    # ------------------------------------------------------------------ #
    #  公共接口                                                            #
    # ------------------------------------------------------------------ #

    def load_scores(self, data_dict: dict):
        """
        为 data_dict 中的每个 symbol 计算宏观评分并注入 features。

        对同一市场的多只股票只抓取一次宏观数据。
        """
        if not self.cfg.get("enabled", True):
            for sym in data_dict:
                n = len(data_dict[sym]["raw"]["close"])
                data_dict[sym]["features"]["macro_score"] = np.zeros(n)
            print("[MACRO] Macro analysis disabled — all scores set to 0")
            return

        start = config.DATA_CONFIG["start_date"]
        end = config.DATA_CONFIG["end_date"]

        us_syms = [s for s in data_dict if detect_market(s) == "US"]
        cn_syms = [s for s in data_dict if detect_market(s) == "CN"]

        if us_syms:
            print("[MACRO] Computing US macro scores (VIX / 10Y Yield / S&P 500)...")
            us_scores = self._compute_us_scores(start, end)
            for sym in us_syms:
                n = len(data_dict[sym]["raw"]["close"])
                data_dict[sym]["features"]["macro_score"] = _align(us_scores, n)

        if cn_syms:
            print("[MACRO] Computing CN macro scores (VIX / Shanghai Composite)...")
            cn_scores = self._compute_cn_scores(start, end)
            for sym in cn_syms:
                n = len(data_dict[sym]["raw"]["close"])
                data_dict[sym]["features"]["macro_score"] = _align(cn_scores, n)

        for sym in data_dict:
            scores = data_dict[sym]["features"]["macro_score"]
            mean_s = np.nanmean(scores)
            print(f"[MACRO] {sym}: mean score = {mean_s:+.3f} "
                  f"(range [{np.nanmin(scores):+.2f}, {np.nanmax(scores):+.2f}])")

    # ------------------------------------------------------------------ #
    #  美股宏观                                                            #
    # ------------------------------------------------------------------ #

    def _compute_us_scores(self, start: str, end: str) -> np.ndarray:
        import yfinance as yf
        w = self.cfg["weights"]
        indicators = self.cfg["us_indicators"]

        scores_parts = []

        vix_score = self._fetch_and_score(
            yf, indicators["vix"], start, end, self._score_vix)
        if vix_score is not None:
            scores_parts.append(("vix", w["vix"], vix_score))

        yield_score = self._fetch_and_score(
            yf, indicators["yield_10y"], start, end, self._score_yield)
        if yield_score is not None:
            scores_parts.append(("yield", w["yield"], yield_score))

        mom_score = self._fetch_and_score(
            yf, indicators["market_index"], start, end, self._score_momentum)
        if mom_score is not None:
            scores_parts.append(("momentum", w["momentum"], mom_score))

        return self._weighted_combine(scores_parts)

    # ------------------------------------------------------------------ #
    #  A 股宏观                                                            #
    # ------------------------------------------------------------------ #

    def _compute_cn_scores(self, start: str, end: str) -> np.ndarray:
        import yfinance as yf
        w = self.cfg["weights"]
        cn_ind = self.cfg["cn_indicators"]

        scores_parts = []

        # VIX 对 A 股同样有全球情绪传导作用，但权重减半
        vix_score = self._fetch_and_score(
            yf, self.cfg["us_indicators"]["vix"], start, end, self._score_vix)
        if vix_score is not None:
            scores_parts.append(("vix", w["vix"] * 0.5, vix_score))

        # 上证综指 — 优先 akshare，回退 yfinance
        mom_score = self._fetch_cn_index(cn_ind, start, end)
        if mom_score is None:
            mom_score = self._fetch_and_score(
                yf, cn_ind["market_index_yf"], start, end, self._score_momentum)
        if mom_score is not None:
            scores_parts.append(("momentum", w["momentum"], mom_score))

        return self._weighted_combine(scores_parts)

    def _fetch_cn_index(self, cn_ind: dict, start: str, end: str):
        import akshare as ak
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                df = ak.stock_zh_index_daily(symbol=cn_ind["market_index_ak"])
                df["date"] = pd.to_datetime(df["date"])
                mask = (df["date"] >= start) & (df["date"] <= end)
                df = df.loc[mask]
                if df.empty:
                    return None
                close = df["close"].values.astype(float)
                return self._score_momentum(close)
            except Exception as e:
                print(f"[MACRO] akshare index attempt {attempt}/{_MAX_RETRIES} failed: {e}")
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY * attempt)
        return None

    # ------------------------------------------------------------------ #
    #  通用 yfinance 获取 + 评分                                           #
    # ------------------------------------------------------------------ #

    def _fetch_and_score(self, yf, ticker, start, end, score_fn):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            close = df["Close"].dropna().values.astype(float)
            if len(close) < 30:
                return None
            return score_fn(close)
        except Exception as e:
            print(f"[MACRO] Failed to fetch {ticker}: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  评分算法                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _score_vix(vix: np.ndarray) -> np.ndarray:
        """
        VIX 评分：水平分 + 趋势分。

        水平:  <15 → +0.8,  15~20 → +0.2~−0.1,  20~30 → −0.3,  >30 → −0.8
        趋势:  20 日变化率，下降 → 看多，上升 → 看空
        """
        n = len(vix)
        out = np.zeros(n)
        for i in range(n):
            v = vix[i]
            if np.isnan(v):
                continue
            if v < 15:
                level = 0.8
            elif v < 20:
                level = 0.3 - (v - 15) * 0.1
            elif v < 30:
                level = -0.2 - (v - 20) * 0.06
            else:
                level = -0.8

            trend = 0.0
            if i >= 20 and vix[i - 20] > 0:
                trend = -np.clip((v - vix[i - 20]) / vix[i - 20] * 3, -0.5, 0.5)

            out[i] = 0.6 * level + 0.4 * trend
        return out

    @staticmethod
    def _score_yield(yld: np.ndarray) -> np.ndarray:
        """
        国债收益率评分：收益率快速上升 → 紧缩 → bearish；下降 → 宽松 → bullish。
        """
        n = len(yld)
        out = np.zeros(n)
        for i in range(5, n):
            if np.isnan(yld[i]):
                continue
            d5 = yld[i] - yld[max(i - 5, 0)]
            d20 = yld[i] - yld[max(i - 20, 0)] if i >= 20 else d5
            short = -np.clip(d5 * 2, -0.6, 0.6)
            long = -np.clip(d20 * 0.8, -0.4, 0.4)
            out[i] = 0.6 * short + 0.4 * long
        return out

    @staticmethod
    def _score_momentum(close: np.ndarray) -> np.ndarray:
        """
        市场指数动量评分：价格与 50 日均线的偏离 + 20 日均线斜率。
        """
        n = len(close)
        out = np.zeros(n)
        for i in range(50, n):
            ma50 = np.mean(close[max(0, i - 49): i + 1])
            ma20 = np.mean(close[max(0, i - 19): i + 1])

            deviation = (close[i] - ma50) / ma50
            pos_score = np.clip(deviation * 5, -0.6, 0.6)

            slope_score = 0.0
            if i >= 39:
                prev_ma20 = np.mean(close[max(0, i - 39): i - 19])
                if prev_ma20 > 0:
                    slope_score = np.clip((ma20 - prev_ma20) / prev_ma20 * 20, -0.4, 0.4)

            out[i] = 0.6 * pos_score + 0.4 * slope_score
        return out

    # ------------------------------------------------------------------ #
    #  辅助                                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _weighted_combine(parts: list) -> np.ndarray:
        """将多个评分分量加权合并并裁剪到 [-1, 1]。"""
        if not parts:
            return np.zeros(1)
        max_len = max(len(s) for _, _, s in parts)
        combined = np.zeros(max_len)
        for name, weight, scores in parts:
            aligned = _align(scores, max_len)
            combined += weight * aligned
        total_w = sum(w for _, w, _ in parts)
        if total_w > 0:
            combined /= total_w
        return np.clip(combined, -1.0, 1.0)


# ================================================================== #
#  模块级工具函数                                                       #
# ================================================================== #

def _align(scores: np.ndarray, target_len: int) -> np.ndarray:
    """将评分数组对齐到 target_len（截尾或前补零）。"""
    n = len(scores)
    if n == target_len:
        return scores
    if n > target_len:
        return scores[n - target_len:]
    result = np.zeros(target_len)
    result[target_len - n:] = scores
    return result


def create_macro_analyzer() -> MacroAnalyzer:
    """工厂方法"""
    return MacroAnalyzer()
