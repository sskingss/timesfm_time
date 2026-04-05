"""
A股短线市场风格分析与选股报告
==============================
市场风格趋势分析 → 整体资金流向 → 板块热点预测 → 个股筛选 → 推送报告

分析维度:
  1. 大盘概览: 涨跌分布、量能、情绪温度
  2. 风格轮动: 大盘/小盘、成长/价值强弱对比
  3. 板块热点: 行业+概念板块资金流入排名
  4. 个股精选: 多因子评分模型筛选上涨概率最大的标的

用法:
  python -m src.market_report
  python -m src.market_report --top 15
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import base64
import json
import os
import sys
import time
import traceback
import urllib.request
import urllib.error

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


# ================================================================
# Constants
# ================================================================

_STYLE_LABELS = {
    "large_value": "大盘价值",
    "large_growth": "大盘成长",
    "small_value": "小盘价值",
    "small_growth": "小盘成长",
    "balanced": "均衡",
}

_SENTIMENT_LABELS = {
    "极度恐慌": (-999, 20),
    "恐慌": (20, 35),
    "偏空": (35, 45),
    "中性": (45, 55),
    "偏多": (55, 65),
    "乐观": (65, 80),
    "极度乐观": (80, 999),
}

_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_RESET = "\033[0m"

_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"
)


# ================================================================
# 1. Data Fetching (akshare)
# ================================================================

def _retry(fn, *args, retries: int = 2, delay: float = 2.0, label: str = "", **kwargs):
    """带重试的 API 调用包装"""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            tag = label or fn.__name__
            if attempt < retries - 1:
                print(f"  [FETCH] {tag} attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(delay * (attempt + 1))
            else:
                print(f"  [FETCH] {tag} failed after {retries} attempts: {e}")
    return None


def _fetch_all_stocks() -> pd.DataFrame | None:
    """全A股实时行情快照 (东方财富)"""
    import akshare as ak
    df = _retry(ak.stock_zh_a_spot_em, label="全A行情")
    if df is None or df.empty:
        return None
    for col in ["最新价", "涨跌幅", "涨跌额", "成交量", "成交额", "振幅",
                "最高", "最低", "今开", "昨收", "量比", "换手率",
                "市盈率-动态", "市净率", "总市值", "流通市值",
                "60日涨跌幅", "年初至今涨跌幅"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fetch_sector_flow(sector_type: str = "行业资金流") -> pd.DataFrame | None:
    """板块资金流排名"""
    import akshare as ak
    df = _retry(
        ak.stock_sector_fund_flow_rank,
        indicator="今日", sector_type=sector_type,
        label=f"板块资金流({sector_type})",
    )
    if df is not None and not df.empty:
        for col in df.columns:
            if col not in ("序号", "名称"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fetch_stock_fund_flow() -> pd.DataFrame | None:
    """个股资金流排名"""
    import akshare as ak
    df = _retry(ak.stock_individual_fund_flow_rank, indicator="今日", label="个股资金流")
    if df is not None and not df.empty:
        for col in df.columns:
            if col not in ("序号", "代码", "名称"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fetch_index_daily(symbol: str, name: str) -> dict | None:
    """获取指数最新收盘数据"""
    import akshare as ak
    try:
        df = ak.stock_zh_index_daily_em(symbol=symbol)
        if df is not None and len(df) >= 2:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            close = float(last["close"])
            prev_close = float(prev["close"])
            change_pct = (close - prev_close) / prev_close * 100 if prev_close else 0
            return {
                "name": name,
                "close": close,
                "change_pct": round(change_pct, 2),
                "date": str(last.get("date", "")),
            }
    except Exception:
        pass
    return None


def _fetch_major_indices() -> list[dict]:
    """获取主要指数行情"""
    targets = [
        ("sh000001", "上证指数"), ("sz399001", "深证成指"),
        ("sz399006", "创业板指"), ("sh000016", "上证50"),
        ("sh000300", "沪深300"), ("sh000905", "中证500"),
        ("sh000852", "中证1000"),
    ]
    results = []
    for code, name in targets:
        data = _fetch_index_daily(code, name)
        if data:
            results.append(data)
    return results


# ================================================================
# 2. Market Overview Analysis
# ================================================================

def _analyze_overview(stocks: pd.DataFrame) -> dict:
    """从全A快照中分析大盘概览"""
    total = len(stocks)
    valid = stocks.dropna(subset=["涨跌幅"])

    change = valid["涨跌幅"]
    up = int((change > 0).sum())
    down = int((change < 0).sum())
    flat = int((change == 0).sum())

    code_col = valid["代码"].astype(str)
    pct = valid["涨跌幅"]

    is_main = code_col.str.startswith(("6", "0"))
    is_gem = code_col.str.startswith("3")
    is_star = code_col.str.startswith("68")
    is_st = valid["名称"].str.contains("ST", na=False)

    limit_up_thresh = np.where(is_gem | is_star, 19.8, np.where(is_st, 4.8, 9.8))
    limit_down_thresh = np.where(is_gem | is_star, -19.8, np.where(is_st, -4.8, -9.8))

    limit_up = int((pct >= limit_up_thresh).sum())
    limit_down = int((pct <= limit_down_thresh).sum())

    total_amount = valid["成交额"].sum() / 1e8 if "成交额" in valid.columns else 0
    avg_change = float(change.mean())
    median_change = float(change.median())

    up_ratio = up / max(total, 1) * 100
    sentiment_score = up_ratio
    if limit_up > 80:
        sentiment_score += 10
    elif limit_up < 10:
        sentiment_score -= 10
    if limit_down > 50:
        sentiment_score -= 15

    sentiment = "中性"
    for label, (lo, hi) in _SENTIMENT_LABELS.items():
        if lo <= sentiment_score < hi:
            sentiment = label
            break

    return {
        "total": total,
        "up": up, "down": down, "flat": flat,
        "limit_up": limit_up, "limit_down": limit_down,
        "total_amount_yi": round(total_amount, 1),
        "avg_change": round(avg_change, 2),
        "median_change": round(median_change, 2),
        "up_ratio": round(up_ratio, 1),
        "sentiment": sentiment,
        "sentiment_score": round(sentiment_score, 1),
    }


# ================================================================
# 3. Market Style Analysis
# ================================================================

def _analyze_style(stocks: pd.DataFrame) -> dict:
    """分析大小盘、成长价值风格强弱"""
    valid = stocks.dropna(subset=["涨跌幅", "总市值"]).copy()
    valid = valid[valid["总市值"] > 0]

    def avg_change(mask):
        s = valid.loc[mask, "涨跌幅"]
        return round(float(s.mean()), 2) if len(s) > 0 else 0

    large = valid["总市值"] >= 500e8
    mid = (valid["总市值"] >= 100e8) & (valid["总市值"] < 500e8)
    small = valid["总市值"] < 100e8

    large_chg = avg_change(large)
    mid_chg = avg_change(mid)
    small_chg = avg_change(small)

    code = valid["代码"].astype(str)
    growth_mask = code.str.startswith("3") | code.str.startswith("68")
    value_mask = ~growth_mask

    growth_chg = avg_change(growth_mask)
    value_chg = avg_change(value_mask)

    size_diff = small_chg - large_chg
    style_diff = growth_chg - value_chg

    if size_diff > 0.5 and style_diff > 0.5:
        style_label = "small_growth"
    elif size_diff > 0.5 and style_diff <= 0.5:
        style_label = "small_value"
    elif size_diff <= -0.5 and style_diff > 0.5:
        style_label = "large_growth"
    elif size_diff <= -0.5 and style_diff <= 0.5:
        style_label = "large_value"
    else:
        style_label = "balanced"

    cap_segments = {
        "超大盘(>500亿)": (int(large.sum()), large_chg),
        "中盘(100-500亿)": (int(mid.sum()), mid_chg),
        "小盘(<100亿)": (int(small.sum()), small_chg),
    }

    momentum_desc = []
    if abs(size_diff) > 1.0:
        leader = "小盘股" if size_diff > 0 else "大盘股"
        momentum_desc.append(f"{leader}显著领涨, 大小盘分化 {abs(size_diff):.1f}%")
    if abs(style_diff) > 1.0:
        leader = "成长风格" if style_diff > 0 else "价值风格"
        momentum_desc.append(f"{leader}占优, 风格差 {abs(style_diff):.1f}%")

    return {
        "large_chg": large_chg,
        "mid_chg": mid_chg,
        "small_chg": small_chg,
        "growth_chg": growth_chg,
        "value_chg": value_chg,
        "style_label": style_label,
        "style_cn": _STYLE_LABELS.get(style_label, "均衡"),
        "cap_segments": cap_segments,
        "momentum_desc": momentum_desc,
    }


# ================================================================
# 4. Sector Hotspot Analysis
# ================================================================

def _analyze_sectors(
    industry_flow: pd.DataFrame | None,
    concept_flow: pd.DataFrame | None,
) -> dict:
    """分析行业和概念板块热点"""

    def _extract_top(df: pd.DataFrame | None, n: int = 8) -> list[dict]:
        if df is None or df.empty:
            return []
        name_col = "名称" if "名称" in df.columns else df.columns[1]
        change_col = next((c for c in df.columns if "涨跌幅" in c), None)
        flow_col = next((c for c in df.columns if "主力净流入" in c and "净额" in c), None)

        if not change_col:
            return []

        sorted_df = df.sort_values(change_col, ascending=False).head(n)
        results = []
        for _, row in sorted_df.iterrows():
            item = {
                "name": str(row.get(name_col, "")),
                "change_pct": round(float(row.get(change_col, 0)), 2),
            }
            if flow_col:
                raw = row.get(flow_col, 0)
                item["net_inflow_yi"] = round(float(raw) / 1e8, 2) if abs(float(raw)) > 1e6 else round(float(raw), 2)
            results.append(item)
        return results

    def _extract_top_inflow(df: pd.DataFrame | None, n: int = 5) -> list[dict]:
        if df is None or df.empty:
            return []
        flow_col = next((c for c in df.columns if "主力净流入" in c and "净额" in c), None)
        if not flow_col:
            return []
        name_col = "名称" if "名称" in df.columns else df.columns[1]
        change_col = next((c for c in df.columns if "涨跌幅" in c), None)
        sorted_df = df.sort_values(flow_col, ascending=False).head(n)
        results = []
        for _, row in sorted_df.iterrows():
            raw = row.get(flow_col, 0)
            item = {
                "name": str(row.get(name_col, "")),
                "net_inflow_yi": round(float(raw) / 1e8, 2) if abs(float(raw)) > 1e6 else round(float(raw), 2),
            }
            if change_col:
                item["change_pct"] = round(float(row.get(change_col, 0)), 2)
            results.append(item)
        return results

    return {
        "hot_industries": _extract_top(industry_flow),
        "hot_concepts": _extract_top(concept_flow),
        "inflow_industries": _extract_top_inflow(industry_flow),
        "inflow_concepts": _extract_top_inflow(concept_flow),
    }


# ================================================================
# 5. Multi-Factor Stock Screening
# ================================================================

def _is_candidate(row: pd.Series) -> bool:
    """基础过滤: 排除不适合短线交易的标的"""
    name = str(row.get("名称", ""))
    code = str(row.get("代码", ""))
    price = row.get("最新价", 0)
    change = row.get("涨跌幅", 0)
    volume = row.get("成交额", 0)
    turnover = row.get("换手率", 0)

    if pd.isna(price) or price <= 0:
        return False
    if "ST" in name or "退" in name:
        return False
    if price < 2:
        return False
    if pd.isna(change):
        return False

    is_gem_star = code.startswith("3") or code.startswith("68")
    limit = 19.8 if is_gem_star else 9.8
    if change >= limit:
        return False
    if change <= -limit:
        return False

    if pd.isna(volume) or volume < 5000 * 1e4:
        return False
    if not pd.isna(turnover) and turnover > 25:
        return False

    return True


def _score_momentum(row: pd.Series) -> tuple[float, list[str]]:
    """动量评分 (满分 25)"""
    change = float(row.get("涨跌幅", 0) or 0)
    change_60d = float(row.get("60日涨跌幅", 0) or 0)
    reasons = []
    score = 0.0

    if 1.0 <= change <= 5.0:
        score += 18 - abs(change - 3) * 2
        reasons.append("短线涨幅适中")
    elif 0 < change < 1.0:
        score += 10
    elif 5.0 < change <= 8.0:
        score += 8
    elif change <= 0:
        score += 2

    if 5 < change_60d <= 30:
        score += 5
        reasons.append("中期趋势向好")
    elif 0 < change_60d <= 5:
        score += 3
    elif change_60d > 30:
        score += 1

    return min(score, 25), reasons


def _score_volume(row: pd.Series) -> tuple[float, list[str]]:
    """量能评分 (满分 25)"""
    vol_ratio = float(row.get("量比", 1) or 1)
    turnover = float(row.get("换手率", 0) or 0)
    amplitude = float(row.get("振幅", 0) or 0)
    reasons = []
    score = 0.0

    if 1.5 <= vol_ratio <= 4.0:
        score += 12
        reasons.append(f"温和放量(量比{vol_ratio:.1f})")
    elif 1.2 <= vol_ratio < 1.5:
        score += 8
    elif 4.0 < vol_ratio <= 8.0:
        score += 6
    elif vol_ratio > 8:
        score += 2

    if 3 <= turnover <= 10:
        score += 8
        if turnover >= 5:
            reasons.append("交投活跃")
    elif 1 <= turnover < 3:
        score += 4
    elif 10 < turnover <= 20:
        score += 5

    if 2 <= amplitude <= 6:
        score += 5
    elif amplitude < 2:
        score += 2
    elif amplitude > 6:
        score += 3

    return min(score, 25), reasons


def _score_capital_flow(code: str, flow_df: pd.DataFrame | None) -> tuple[float, list[str]]:
    """资金流评分 (满分 25)"""
    if flow_df is None or flow_df.empty:
        return 5, []

    match = flow_df[flow_df["代码"].astype(str) == str(code)]
    if match.empty:
        return 5, []

    row = match.iloc[0]
    reasons = []
    score = 0.0

    flow_ratio_col = next(
        (c for c in flow_df.columns if "主力净流入" in c and "净占比" in c), None
    )
    flow_amount_col = next(
        (c for c in flow_df.columns if "主力净流入" in c and "净额" in c), None
    )
    super_col = next(
        (c for c in flow_df.columns if "超大单" in c and "净占比" in c), None
    )

    flow_ratio = float(row.get(flow_ratio_col, 0) or 0) if flow_ratio_col else 0
    flow_amount = float(row.get(flow_amount_col, 0) or 0) if flow_amount_col else 0
    super_ratio = float(row.get(super_col, 0) or 0) if super_col else 0

    if flow_ratio > 10:
        score += 15
        reasons.append("主力大幅流入")
    elif flow_ratio > 5:
        score += 12
        reasons.append("主力持续加仓")
    elif flow_ratio > 2:
        score += 9
        reasons.append("主力小幅流入")
    elif flow_ratio > 0:
        score += 5
    else:
        score += 1

    if super_ratio > 5:
        score += 8
        reasons.append("超大单抢筹")
    elif super_ratio > 0:
        score += 4

    return min(score, 25), reasons


def _score_valuation(row: pd.Series) -> tuple[float, list[str]]:
    """估值与市值评分 (满分 15)"""
    pe = float(row.get("市盈率-动态", 0) or 0)
    pb = float(row.get("市净率", 0) or 0)
    mcap = float(row.get("总市值", 0) or 0)
    reasons = []
    score = 0.0

    if 10 <= pe <= 40:
        score += 5
        reasons.append("估值合理")
    elif 0 < pe < 10:
        score += 4
    elif 40 < pe <= 80:
        score += 3
    elif pe > 80 or pe <= 0:
        score += 1

    if 0.5 <= pb <= 5:
        score += 3
    elif pb > 5:
        score += 1

    if 80e8 <= mcap <= 500e8:
        score += 5
        if mcap >= 200e8:
            reasons.append("中大盘标的")
    elif 500e8 < mcap <= 2000e8:
        score += 4
    elif 30e8 <= mcap < 80e8:
        score += 3
    elif mcap > 2000e8:
        score += 3

    return min(score, 15), reasons


def _score_extra(row: pd.Series) -> tuple[float, list[str]]:
    """附加评分 (满分 10): 综合质量因子"""
    change_ytd = float(row.get("年初至今涨跌幅", 0) or 0)
    reasons = []
    score = 0.0

    if -10 < change_ytd < 20:
        score += 5
        if change_ytd < 0:
            reasons.append("年内滞涨有补涨空间")
    elif 20 <= change_ytd < 50:
        score += 3

    amplitude = float(row.get("振幅", 0) or 0)
    change = float(row.get("涨跌幅", 0) or 0)
    if amplitude > 0 and change > 0:
        efficiency = change / amplitude
        if efficiency > 0.6:
            score += 5
            reasons.append("上涨效率高(实体大)")
        elif efficiency > 0.4:
            score += 3

    return min(score, 10), reasons


def screen_stocks(
    stocks: pd.DataFrame,
    flow_df: pd.DataFrame | None,
    top_n: int = 10,
) -> list[dict]:
    """多因子评分模型筛选个股"""
    candidates = stocks[stocks.apply(_is_candidate, axis=1)].copy()
    print(f"[SCREEN] {len(candidates)} candidates after basic filtering (from {len(stocks)} total)")

    results = []
    for _, row in candidates.iterrows():
        code = str(row.get("代码", ""))
        name = str(row.get("名称", ""))

        s1, r1 = _score_momentum(row)
        s2, r2 = _score_volume(row)
        s3, r3 = _score_capital_flow(code, flow_df)
        s4, r4 = _score_valuation(row)
        s5, r5 = _score_extra(row)

        total = s1 + s2 + s3 + s4 + s5
        all_reasons = r1 + r2 + r3 + r4 + r5

        results.append({
            "code": code,
            "name": name,
            "price": float(row.get("最新价", 0) or 0),
            "change_pct": round(float(row.get("涨跌幅", 0) or 0), 2),
            "volume_ratio": round(float(row.get("量比", 0) or 0), 1),
            "turnover_rate": round(float(row.get("换手率", 0) or 0), 1),
            "amount_yi": round(float(row.get("成交额", 0) or 0) / 1e8, 1),
            "market_cap_yi": round(float(row.get("总市值", 0) or 0) / 1e8, 0),
            "total_score": round(total, 1),
            "scores": {
                "momentum": round(s1, 1),
                "volume": round(s2, 1),
                "capital": round(s3, 1),
                "valuation": round(s4, 1),
                "extra": round(s5, 1),
            },
            "reasons": all_reasons if all_reasons else ["综合评分靠前"],
        })

    results.sort(key=lambda x: x["total_score"], reverse=True)
    return results[:top_n]


# ================================================================
# 6. Risk Assessment
# ================================================================

def _assess_risks(overview: dict, style: dict) -> list[str]:
    """基于市场数据生成风险提示"""
    warnings = []

    if overview["sentiment_score"] > 75:
        warnings.append("市场情绪偏热，涨停家数较多，短期追高风险加大，注意控制仓位")
    if overview["sentiment_score"] < 30:
        warnings.append("市场情绪极度低迷，恐慌抛售可能加剧，建议轻仓观望")

    if overview["limit_up"] > 100:
        warnings.append(f"涨停{overview['limit_up']}家，市场情绪亢奋，谨防次日分歧回落")
    if overview["limit_down"] > 50:
        warnings.append(f"跌停{overview['limit_down']}家，市场杀跌明显，短期回避高位股")

    large = style["large_chg"]
    small = style["small_chg"]
    if abs(large - small) > 2.0:
        leader = "大盘股" if large > small else "小盘股"
        laggard = "小盘股" if large > small else "大盘股"
        warnings.append(f"大小盘严重分化({leader}强{laggard}弱)，风格切换风险较高")

    if overview["total_amount_yi"] < 6000:
        warnings.append("全市场成交不足6000亿，量能萎缩，上涨持续性存疑")
    elif overview["total_amount_yi"] > 20000:
        warnings.append("全市场成交超2万亿，天量之后常见回落，短期获利盘压力大")

    if overview["up_ratio"] > 80:
        warnings.append("上涨比例超80%，普涨行情后注意分化回调")

    if not warnings:
        warnings.append("当前无显著风险信号，保持正常仓位管理即可")

    return warnings


# ================================================================
# 7. Trend Prediction (Forward-Looking)
# ================================================================

def _predict_trend(overview: dict, style: dict, sectors: dict) -> list[str]:
    """基于当前市场状态做出短期趋势预判"""
    predictions = []

    if overview["avg_change"] > 0.5 and overview["total_amount_yi"] > 10000:
        predictions.append("量价齐升，市场做多动能充沛，短期大盘有望延续反弹")
    elif overview["avg_change"] > 0.5 and overview["total_amount_yi"] < 8000:
        predictions.append("指数上涨但量能不足，反弹持续性待验证，关注明日能否放量")
    elif overview["avg_change"] < -0.5 and overview["total_amount_yi"] > 12000:
        predictions.append("放量下跌，短期调整压力较大，预计还需1-2日消化卖盘")
    elif overview["avg_change"] < -0.5 and overview["total_amount_yi"] < 8000:
        predictions.append("缩量回调，抛压有限，调整幅度可控，关注支撑位企稳信号")

    if style["style_label"] == "small_growth":
        predictions.append("小盘成长风格占优，题材股活跃度高，可关注热点概念的龙头补涨机会")
    elif style["style_label"] == "large_value":
        predictions.append("大盘价值风格占优，权重股护盘明显，建议关注低估值蓝筹的修复行情")
    elif style["style_label"] == "balanced":
        predictions.append("市场风格均衡，无明显主线，建议分散配置，等待风格明确后加仓")

    hot = sectors.get("hot_industries", [])
    if hot:
        top_sector = hot[0]["name"]
        top_change = hot[0]["change_pct"]
        if top_change > 3:
            predictions.append(f"板块方面，{top_sector}领涨且涨幅较大，明日关注板块内滞涨个股的跟风机会")
        elif top_change > 1:
            predictions.append(f"板块方面，{top_sector}等板块走强，若资金持续流入可保持关注")

    inflow = sectors.get("inflow_industries", [])
    if inflow and inflow[0].get("net_inflow_yi", 0) > 10:
        predictions.append(
            f"资金面上，{inflow[0]['name']}获主力大幅净流入"
            f"({inflow[0]['net_inflow_yi']:.1f}亿)，后续可能延续强势"
        )

    return predictions


# ================================================================
# 8. Console Report Builder
# ================================================================

def _build_console_report(
    report_time: str,
    overview: dict,
    indices: list[dict],
    style: dict,
    sectors: dict,
    picks: list[dict],
    risks: list[str],
    predictions: list[str],
) -> str:
    """构建终端可读的完整分析报告"""
    lines = []
    w = 64

    lines.append(f"\n{'═' * w}")
    lines.append(f"  📊 A股短线市场分析报告 — {report_time}")
    lines.append(f"{'═' * w}")

    # ── 主要指数 ──
    if indices:
        lines.append(f"\n  {_BOLD}▌ 主要指数{_RESET}")
        for idx in indices:
            c = _GREEN if idx["change_pct"] > 0 else _RED if idx["change_pct"] < 0 else _YELLOW
            arrow = "▲" if idx["change_pct"] > 0 else "▼" if idx["change_pct"] < 0 else "─"
            lines.append(
                f"  ├─ {idx['name']:<8s} {idx['close']:>10.2f}  "
                f"{c}{arrow} {idx['change_pct']:+.2f}%{_RESET}"
            )

    # ── 大盘概览 ──
    lines.append(f"\n  {_BOLD}▌ 大盘概览{_RESET}")
    lines.append(
        f"  ├─ 全市场 {overview['total']} 只 | "
        f"{_GREEN}上涨 {overview['up']}{_RESET} | "
        f"{_RED}下跌 {overview['down']}{_RESET} | "
        f"平盘 {overview['flat']}"
    )
    lines.append(
        f"  ├─ {_RED}涨停 {overview['limit_up']}{_RESET} | "
        f"{_GREEN}跌停 {overview['limit_down']}{_RESET} | "
        f"上涨比例 {overview['up_ratio']:.1f}%"
    )
    lines.append(
        f"  ├─ 成交额 {overview['total_amount_yi']:.0f} 亿 | "
        f"均涨 {overview['avg_change']:+.2f}% | "
        f"中位数 {overview['median_change']:+.2f}%"
    )
    lines.append(f"  └─ 市场情绪: {_BOLD}{overview['sentiment']}{_RESET} (评分 {overview['sentiment_score']:.0f})")

    # ── 风格分析 ──
    lines.append(f"\n  {_BOLD}▌ 市场风格{_RESET}")
    lines.append(f"  ├─ 当前主导: {_CYAN}{_BOLD}{style['style_cn']}{_RESET}")
    lines.append(
        f"  ├─ 大盘股 {style['large_chg']:+.2f}% | "
        f"中盘 {style['mid_chg']:+.2f}% | "
        f"小盘 {style['small_chg']:+.2f}%"
    )
    lines.append(
        f"  ├─ 成长 {style['growth_chg']:+.2f}% | "
        f"价值 {style['value_chg']:+.2f}%"
    )
    for desc in style.get("momentum_desc", []):
        lines.append(f"  └─ 💡 {desc}")

    # ── 板块热点 ──
    lines.append(f"\n  {_BOLD}▌ 行业板块 TOP8 (按涨幅){_RESET}")
    lines.append(f"  │  {'排名':^4s}  {'板块':<10s} {'涨幅':>8s}  {'主力净流入':>10s}")
    lines.append(f"  │  {'─' * 38}")
    for i, s in enumerate(sectors.get("hot_industries", [])[:8], 1):
        medal = "🥇🥈🥉"[i - 1] if i <= 3 else f"  {i}"
        inflow_str = f"{s.get('net_inflow_yi', 0):+.1f}亿" if "net_inflow_yi" in s else "N/A"
        c = _GREEN if s["change_pct"] > 0 else _RED
        lines.append(
            f"  │  {medal}  {s['name']:<10s} {c}{s['change_pct']:>+7.2f}%{_RESET}  {inflow_str:>10s}"
        )

    if sectors.get("hot_concepts"):
        lines.append(f"\n  {_BOLD}▌ 概念板块 TOP5 (按涨幅){_RESET}")
        for i, s in enumerate(sectors["hot_concepts"][:5], 1):
            medal = "🥇🥈🥉"[i - 1] if i <= 3 else f"  {i}"
            inflow_str = f"{s.get('net_inflow_yi', 0):+.1f}亿" if "net_inflow_yi" in s else ""
            c = _GREEN if s["change_pct"] > 0 else _RED
            lines.append(
                f"  │  {medal}  {s['name']:<12s} {c}{s['change_pct']:>+7.2f}%{_RESET}  {inflow_str}"
            )

    # ── 趋势预判 ──
    if predictions:
        lines.append(f"\n  {_BOLD}▌ 短线趋势预判{_RESET}")
        for i, p in enumerate(predictions):
            prefix = "└" if i == len(predictions) - 1 else "├"
            lines.append(f"  {prefix}─ 🔮 {p}")

    # ── 个股精选 ──
    lines.append(f"\n  {_BOLD}▌ 精选个股 TOP{len(picks)}{_RESET}")
    lines.append(
        f"  │  {'排名':^4s}  {'代码':<8s} {'名称':<8s} {'现价':>8s} "
        f"{'涨幅':>7s} {'评分':>5s} {'量比':>5s} {'换手':>6s} {'成交额':>7s}"
    )
    lines.append(f"  │  {'─' * 64}")
    medals = ["🥇", "🥈", "🥉"]
    for i, p in enumerate(picks, 1):
        medal = medals[i - 1] if i <= 3 else f"  {i}"
        c = _GREEN if p["change_pct"] > 0 else _RED if p["change_pct"] < 0 else ""
        lines.append(
            f"  │  {medal}  {p['code']:<8s} {p['name']:<8s} "
            f"¥{p['price']:>7.2f} {c}{p['change_pct']:>+6.2f}%{_RESET} "
            f"{p['total_score']:>5.1f} "
            f"{p['volume_ratio']:>5.1f} "
            f"{p['turnover_rate']:>5.1f}% "
            f"{p['amount_yi']:>6.1f}亿"
        )
        if p["reasons"]:
            tags = " ".join(f"✦{r}" for r in p["reasons"][:4])
            lines.append(f"  │       {_DIM}{tags}{_RESET}")

    # ── 风险提示 ──
    lines.append(f"\n  {_BOLD}▌ 风险提示{_RESET}")
    for i, r in enumerate(risks):
        prefix = "└" if i == len(risks) - 1 else "├"
        lines.append(f"  {prefix}─ ⚠ {r}")

    lines.append(f"\n{'━' * w}")
    lines.append(f"  TimesFM Quant · 短线市场分析报告 · 仅供参考，不构成投资建议")
    lines.append(f"{'━' * w}\n")

    return "\n".join(lines)


# ================================================================
# 9. Feishu Card Builder
# ================================================================

def _build_feishu_card(
    report_time: str,
    overview: dict,
    indices: list[dict],
    style: dict,
    sectors: dict,
    picks: list[dict],
    risks: list[str],
    predictions: list[str],
) -> dict:
    """构建飞书交互卡片消息"""
    elements: list[dict] = []

    # ── 指数概览 ──
    idx_lines = []
    for idx in indices[:4]:
        arrow = "📈" if idx["change_pct"] > 0 else "📉" if idx["change_pct"] < 0 else "➡️"
        idx_lines.append(f"{arrow} {idx['name']} {idx['close']:.2f} ({idx['change_pct']:+.2f}%)")
    if idx_lines:
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": "\n".join(idx_lines)},
        })

    # ── 大盘概览 ──
    elements.append({"tag": "hr"})
    elements.append({
        "tag": "div",
        "fields": [
            {"is_short": True, "text": {"tag": "lark_md",
                "content": f"**涨跌**\n🟢{overview['up']}家 / 🔴{overview['down']}家"}},
            {"is_short": True, "text": {"tag": "lark_md",
                "content": f"**涨跌停**\n涨停{overview['limit_up']} / 跌停{overview['limit_down']}"}},
            {"is_short": True, "text": {"tag": "lark_md",
                "content": f"**成交额**\n{overview['total_amount_yi']:.0f}亿"}},
            {"is_short": True, "text": {"tag": "lark_md",
                "content": f"**情绪**\n{overview['sentiment']}({overview['sentiment_score']:.0f})"}},
        ],
    })

    # ── 风格判断 ──
    elements.append({"tag": "hr"})
    style_text = f"**🎯 市场风格: {style['style_cn']}**\n"
    style_text += f"大盘{style['large_chg']:+.2f}% | 中盘{style['mid_chg']:+.2f}% | 小盘{style['small_chg']:+.2f}%\n"
    style_text += f"成长{style['growth_chg']:+.2f}% | 价值{style['value_chg']:+.2f}%"
    elements.append({"tag": "div", "text": {"tag": "lark_md", "content": style_text}})

    # ── 热点板块 ──
    hot = sectors.get("hot_industries", [])[:5]
    if hot:
        elements.append({"tag": "hr"})
        sector_lines = ["**🔥 行业热点**"]
        for i, s in enumerate(hot, 1):
            inflow = f" | 主力{s.get('net_inflow_yi', 0):+.1f}亿" if "net_inflow_yi" in s else ""
            sector_lines.append(f"{i}. {s['name']} {s['change_pct']:+.2f}%{inflow}")
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": "\n".join(sector_lines)}})

    # ── 趋势预判 ──
    if predictions:
        elements.append({"tag": "hr"})
        pred_lines = ["**🔮 短线趋势预判**"]
        for p in predictions[:3]:
            pred_lines.append(f"• {p}")
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": "\n".join(pred_lines)}})

    # ── 精选个股 ──
    elements.append({"tag": "hr"})
    pick_lines = [f"**🏆 精选个股 TOP{min(len(picks), 8)}**"]
    medals = ["🥇", "🥈", "🥉"]
    for i, p in enumerate(picks[:8], 1):
        medal = medals[i - 1] if i <= 3 else f"{i}."
        tags = " ".join(f"✦{r}" for r in p["reasons"][:2])
        pick_lines.append(
            f"{medal} **{p['code']} {p['name']}** ¥{p['price']:.2f} "
            f"({p['change_pct']:+.2f}%) 评分:`{p['total_score']:.0f}`\n"
            f"   量比{p['volume_ratio']:.1f} | 换手{p['turnover_rate']:.1f}% "
            f"| {p['amount_yi']:.1f}亿  {tags}"
        )
    elements.append({"tag": "div", "text": {"tag": "lark_md", "content": "\n".join(pick_lines)}})

    # ── 风险提示 ──
    elements.append({"tag": "hr"})
    risk_text = "**⚠️ 风险提示**\n" + "\n".join(f"• {r}" for r in risks[:3])
    elements.append({"tag": "div", "text": {"tag": "lark_md", "content": risk_text}})

    # ── 底部 ──
    elements.append({
        "tag": "note",
        "elements": [{"tag": "plain_text",
                       "content": "TimesFM Quant · 短线市场分析报告 · 仅供参考，不构成投资建议"}],
    })

    header_color = "green" if overview["avg_change"] > 0.3 else "red" if overview["avg_change"] < -0.3 else "blue"
    card = {
        "header": {
            "title": {"tag": "plain_text",
                       "content": f"📊 A股短线市场研报 — {report_time} | {overview['sentiment']}"},
            "template": header_color,
        },
        "elements": elements,
    }
    return {"msg_type": "interactive", "card": card}


# ================================================================
# 10. Feishu Sender (reuse config)
# ================================================================

def _send_feishu(payload: dict) -> None:
    """通过飞书发送报告 (复用 PLUGIN_CONFIG 中的配置)"""
    feishu_cfg = config.PLUGIN_CONFIG.get("feishu", {})
    if not feishu_cfg.get("enabled"):
        print("  [Feishu] Not enabled, skipping report push")
        return

    webhook_url = feishu_cfg.get("webhook_url", "")
    if webhook_url:
        _send_via_webhook(webhook_url, feishu_cfg.get("secret", ""), payload)
        return

    app_id = feishu_cfg.get("app_id", "")
    app_secret = feishu_cfg.get("app_secret", "")
    if app_id and app_secret:
        _send_via_sdk(feishu_cfg, payload)
        return

    print("  [Feishu] No webhook or SDK configured, skipping")


def _send_via_webhook(url: str, secret: str, payload: dict) -> None:
    if secret:
        ts = str(int(time.time()))
        payload["timestamp"] = ts
        string_to_sign = f"{ts}\n{secret}"
        hmac_code = hmac.new(string_to_sign.encode(), digestmod=hashlib.sha256).digest()
        payload["sign"] = base64.b64encode(hmac_code).decode()

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
            if result.get("code", 0) != 0:
                print(f"  [Feishu] Webhook error: {result.get('msg', 'unknown')}")
            else:
                print("  [Feishu] ✓ Report sent via webhook")
    except urllib.error.URLError as e:
        print(f"  [Feishu] Webhook failed: {e}")


def _send_via_sdk(cfg: dict, payload: dict) -> None:
    try:
        import lark_oapi as lark
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        client = (
            lark.Client.builder()
            .app_id(cfg["app_id"])
            .app_secret(cfg["app_secret"])
            .build()
        )

        if payload["msg_type"] == "interactive":
            content_str = json.dumps(payload["card"], ensure_ascii=False)
        else:
            content_str = json.dumps(payload["content"], ensure_ascii=False)

        body = (
            CreateMessageRequestBody.builder()
            .receive_id(cfg.get("receive_id", ""))
            .msg_type(payload["msg_type"])
            .content(content_str)
            .build()
        )
        request = (
            CreateMessageRequest.builder()
            .receive_id_type(cfg.get("receive_id_type", "email"))
            .request_body(body)
            .build()
        )

        response = client.im.v1.message.create(request)
        if response.success():
            rid = cfg.get("receive_id", "")
            print(f"  [Feishu] ✓ Report sent via SDK → {rid}")
        else:
            print(f"  [Feishu] SDK error: code={response.code}, msg={response.msg}")

    except ImportError:
        print("  [Feishu] lark-oapi not installed, cannot send via SDK")
    except Exception as e:
        print(f"  [Feishu] SDK send failed: {e}")


# ================================================================
# 11. Report File Saver
# ================================================================

def _save_report_text(text: str, report_time: str) -> str | None:
    """将报告保存为文本文件"""
    try:
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        date_str = report_time.replace("-", "").replace(":", "").replace(" ", "_")
        path = os.path.join(_OUTPUT_DIR, f"market_report_{date_str}.txt")
        import re
        clean = re.sub(r"\033\[[0-9;]*m", "", text)
        with open(path, "w", encoding="utf-8") as f:
            f.write(clean)
        print(f"[REPORT] Saved: {path}")
        return path
    except Exception as e:
        print(f"[REPORT] Save failed: {e}")
        return None


# ================================================================
# 12. Main Entry
# ================================================================

def run(top_n: int = 10) -> dict:
    """执行完整的市场分析流程并推送报告"""
    print("=" * 60)
    print(f"  📊 A股短线市场分析报告 — {dt.date.today()}")
    print("=" * 60)

    start_time = time.time()
    report_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Step 1: Fetch data ──
    print("\n[REPORT] Step 1/5: Fetching market data...")
    all_stocks = _fetch_all_stocks()
    if all_stocks is None or all_stocks.empty:
        print("[REPORT] ✗ Failed to fetch stock data, aborting.")
        return {}

    print(f"[REPORT] Loaded {len(all_stocks)} stocks")

    print("[REPORT] Fetching index data...")
    indices = _fetch_major_indices()
    print(f"[REPORT] Loaded {len(indices)} indices")

    print("[REPORT] Fetching sector fund flow...")
    industry_flow = _fetch_sector_flow("行业资金流")
    concept_flow = _fetch_sector_flow("概念资金流")

    print("[REPORT] Fetching individual stock fund flow...")
    stock_flow = _fetch_stock_fund_flow()

    # ── Step 2: Analyze market ──
    print("\n[REPORT] Step 2/5: Analyzing market overview...")
    overview = _analyze_overview(all_stocks)

    print("[REPORT] Step 3/5: Analyzing market style...")
    style = _analyze_style(all_stocks)

    print("[REPORT] Analyzing sector hotspots...")
    sectors = _analyze_sectors(industry_flow, concept_flow)

    # ── Step 3: Screen stocks ──
    print(f"\n[REPORT] Step 4/5: Screening top {top_n} stocks...")
    picks = screen_stocks(all_stocks, stock_flow, top_n)

    # ── Step 4: Risk & Prediction ──
    risks = _assess_risks(overview, style)
    predictions = _predict_trend(overview, style, sectors)

    # ── Step 5: Build & push report ──
    print(f"\n[REPORT] Step 5/5: Building and sending report...")

    console_text = _build_console_report(
        report_time, overview, indices, style, sectors, picks, risks, predictions,
    )
    print(console_text)

    _save_report_text(console_text, report_time)

    feishu_card = _build_feishu_card(
        report_time, overview, indices, style, sectors, picks, risks, predictions,
    )
    _send_feishu(feishu_card)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Report completed in {elapsed:.1f}s — {len(picks)} stock(s) selected")
    print(f"{'=' * 60}")

    return {
        "overview": overview,
        "style": style,
        "sectors": sectors,
        "picks": picks,
        "risks": risks,
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser(description="A股短线市场分析报告")
    parser.add_argument("--top", type=int, default=10, help="精选个股数量 (default: 10)")
    args = parser.parse_args()

    try:
        run(top_n=args.top)
    except KeyboardInterrupt:
        print("\n[REPORT] Interrupted by user")
    except Exception as e:
        print(f"\n[REPORT] Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
