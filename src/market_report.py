"""
A股短线市场风格分析与选股报告
==============================
数据源: 腾讯实时行情 · 新浪行业分类 · AKShare 指数

分析维度:
  1. 大盘概览: 涨跌分布、量能、情绪温度
  2. 风格轮动: 大盘/小盘、成长/价值强弱对比
  3. 板块热点: 行业板块涨幅 & 量能排名
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
import re
import sys
import time
import traceback
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
import requests as _requests

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

_TENCENT_API = "http://qt.gtimg.cn/q="
_SINA_SECTOR_API = "https://vip.stock.finance.sina.com.cn/q/view/newSinaHy.php"
_EM_DATACENTER_API = "https://datacenter-web.eastmoney.com/api/data/v1/get"

_BATCH_SIZE = 800
_BATCH_DELAY = 0.15

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "report"
)


# ================================================================
# 1. Data Fetching — Tencent Real-Time Quotes
# ================================================================

def _safe_float(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _generate_all_codes() -> list[str]:
    """生成全部 A 股代码 (上海 + 深圳), 返回腾讯 API 格式"""
    codes: list[str] = []
    for start, end in [(600000, 602000), (603000, 604000), (605000, 606000)]:
        codes += [f"sh{i}" for i in range(start, end)]
    codes += [f"sh{i}" for i in range(688000, 690000)]
    codes += [f"sz{i:06d}" for i in range(1, 4500)]
    codes += [f"sz{i}" for i in range(300000, 302000)]
    return codes


def _parse_tencent_line(line: str) -> dict | None:
    """解析腾讯行情 API 返回的单行数据"""
    if "~" not in line or '=""' in line:
        return None
    try:
        raw = line.split('"')[1]
    except IndexError:
        return None
    parts = raw.split("~")
    if len(parts) < 50:
        return None

    price = _safe_float(parts[3])
    if price <= 0:
        return None

    row = {
        "代码": parts[2],
        "名称": re.sub(r"\s+", "", parts[1]),
        "最新价": price,
        "昨收": _safe_float(parts[4]),
        "今开": _safe_float(parts[5]),
        "成交量": _safe_float(parts[6]) * 100,
        "成交额": _safe_float(parts[37]) * 1e4,
        "涨跌额": _safe_float(parts[31]),
        "涨跌幅": _safe_float(parts[32]),
        "最高": _safe_float(parts[33]),
        "最低": _safe_float(parts[34]),
        "换手率": _safe_float(parts[38]),
        "量比": _safe_float(parts[49]),
        "振幅": _safe_float(parts[43]),
        "市盈率-动态": _safe_float(parts[39]),
        "市净率": _safe_float(parts[46]),
        "流通市值": _safe_float(parts[44]) * 1e8,
        "总市值": _safe_float(parts[45]) * 1e8,
        "外盘": _safe_float(parts[7]),
        "内盘": _safe_float(parts[8]),
    }
    if len(parts) > 66:
        row["60日涨跌幅"] = _safe_float(parts[65])
        row["年初至今涨跌幅"] = _safe_float(parts[66])
    return row


def _fetch_tencent_quotes(codes: list[str]) -> pd.DataFrame:
    """通过腾讯 qt.gtimg.cn 批量拉取全 A 实时行情"""
    session = _requests.Session()
    rows: list[dict] = []
    total_batches = (len(codes) + _BATCH_SIZE - 1) // _BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * _BATCH_SIZE
        batch = codes[start : start + _BATCH_SIZE]
        url = _TENCENT_API + ",".join(batch)

        resp = None
        for attempt in range(3):
            try:
                resp = session.get(url, timeout=15)
                break
            except Exception as exc:
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    print(f"  [FETCH] batch {batch_idx + 1}/{total_batches} failed: {exc}")

        if resp is None:
            continue

        for line in resp.text.split(";"):
            parsed = _parse_tencent_line(line)
            if parsed:
                rows.append(parsed)

        if batch_idx < total_batches - 1:
            time.sleep(_BATCH_DELAY)

        if (batch_idx + 1) % 5 == 0 or batch_idx + 1 == total_batches:
            print(
                f"  [FETCH] {batch_idx + 1}/{total_batches} batches "
                f"({len(rows)} stocks)"
            )

    session.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ================================================================
# 2. Data Fetching — Industry Classification
# ================================================================

def _fetch_industry_mapping() -> dict[str, str]:
    """从东方财富 datacenter 获取全 A 股行业分类映射 (代码→行业名)

    数据源: 申万行业分类, 覆盖 ~5500 只 A 股.
    结果按日缓存到 cache/report/ 下.
    """
    import pickle

    os.makedirs(_CACHE_DIR, exist_ok=True)
    today = dt.date.today().isoformat()
    cache_path = os.path.join(_CACHE_DIR, f"industry_map_{today}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        print(f"  [FETCH] Industry mapping loaded from cache ({len(cached)} codes)")
        return cached

    code_to_industry: dict[str, str] = {}
    for page in range(1, 6):
        params = {
            "reportName": "RPT_F10_CORETHEME_BOARDTYPE",
            "columns": "SECURITY_CODE,BOARD_NAME",
            "pageSize": "6000",
            "pageNumber": str(page),
            "sortTypes": "1",
            "sortColumns": "SECURITY_CODE",
            "filter": '(BOARD_TYPE="行业")',
        }
        try:
            resp = _requests.get(_EM_DATACENTER_API, params=params, timeout=30)
            result = resp.json().get("result")
            if result is None:
                break
            data = result.get("data", [])
            if not data:
                break
            for item in data:
                code = item.get("SECURITY_CODE", "")
                industry = item.get("BOARD_NAME", "")
                if code and industry:
                    code_to_industry[code] = industry
        except Exception as exc:
            print(f"  [FETCH] East Money industry page {page} failed: {exc}")
            break

    if code_to_industry:
        with open(cache_path, "wb") as f:
            pickle.dump(code_to_industry, f)

    return code_to_industry


def _fetch_sina_sector_summary() -> list[dict]:
    """从新浪获取行业板块行情摘要 (49 个粗行业的涨跌幅)"""
    try:
        resp = _requests.get(
            _SINA_SECTOR_API,
            headers={"Referer": "https://finance.sina.com.cn"},
            timeout=10,
        )
        text = resp.text.strip()
        if not text.startswith("var"):
            return []

        payload = text.split("=", 1)[1].strip().rstrip(";")
        data: dict = json.loads(payload.replace("'", '"'))

        sector_list: list[dict] = []
        for _key, info_str in data.items():
            parts = info_str.split(",")
            if len(parts) < 6:
                continue
            sector_list.append({
                "name": parts[1],
                "change_pct": round(_safe_float(parts[5]), 2),
                "count": int(parts[2]) if parts[2].isdigit() else 0,
            })

        sector_list.sort(key=lambda x: x["change_pct"], reverse=True)
        return sector_list

    except Exception as exc:
        print(f"  [FETCH] Sina sector summary failed: {exc}")
        return []


# ================================================================
# 3. Data Fetching — AKShare Index
# ================================================================

def _fetch_index_daily(symbol: str, name: str) -> dict | None:
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
# 4. Market Overview Analysis
# ================================================================

def _analyze_overview(stocks: pd.DataFrame) -> dict:
    total = len(stocks)
    valid = stocks.dropna(subset=["涨跌幅"])

    change = valid["涨跌幅"]
    up = int((change > 0).sum())
    down = int((change < 0).sum())
    flat = int((change == 0).sum())

    code_col = valid["代码"].astype(str)
    pct = valid["涨跌幅"]

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
# 5. Market Style Analysis
# ================================================================

def _analyze_style(stocks: pd.DataFrame) -> dict:
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

    momentum_desc: list[str] = []
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
# 6. Sector Hotspot Analysis
# ================================================================

def _analyze_sectors(
    stocks: pd.DataFrame,
    sector_map: dict[str, str],
    sina_sectors: list[dict],
) -> dict:
    """从全量快照 + 行业分类映射分析板块热点"""
    valid = stocks.dropna(subset=["涨跌幅"]).copy()
    valid["行业"] = valid["代码"].map(sector_map).fillna("未分类")

    classified = valid[valid["行业"] != "未分类"]
    if classified.empty:
        top8 = sina_sectors[:8]
        return {
            "hot_industries": [
                {"name": s["name"], "change_pct": s["change_pct"]} for s in top8
            ],
            "hot_concepts": [],
            "inflow_industries": [],
            "inflow_concepts": [],
        }

    grouped = classified.groupby("行业")
    stats = grouped.agg(
        count=("代码", "count"),
        avg_change=("涨跌幅", "mean"),
        total_amount=("成交额", "sum"),
        avg_turnover=("换手率", "mean"),
        up_count=("涨跌幅", lambda x: (x > 0).sum()),
        net_buy=("外盘", "sum"),
        net_sell=("内盘", "sum"),
    ).reset_index()

    stats["up_ratio"] = (stats["up_count"] / stats["count"] * 100).round(1)

    sina_map = {s["name"]: s["change_pct"] for s in sina_sectors}
    for idx, row in stats.iterrows():
        if row["行业"] in sina_map:
            stats.at[idx, "avg_change"] = sina_map[row["行业"]]

    hot_industries: list[dict] = []
    for _, row in stats.sort_values("avg_change", ascending=False).head(10).iterrows():
        hot_industries.append({
            "name": row["行业"],
            "change_pct": round(row["avg_change"], 2),
            "count": int(row["count"]),
            "up_ratio": row["up_ratio"],
            "total_amount_yi": round(row["total_amount"] / 1e8, 1),
        })

    stats["net_buy_val"] = (stats["net_buy"] - stats["net_sell"]) * stats["avg_change"]
    inflow_industries: list[dict] = []
    for _, row in stats.sort_values("net_buy_val", ascending=False).head(5).iterrows():
        net = (row["net_buy"] - row["net_sell"])
        inflow_industries.append({
            "name": row["行业"],
            "change_pct": round(row["avg_change"], 2),
            "net_inflow_yi": round(row["total_amount"] / 1e8 * 0.01, 1),
        })

    return {
        "hot_industries": hot_industries,
        "hot_concepts": [],
        "inflow_industries": inflow_industries,
        "inflow_concepts": [],
    }


# ================================================================
# 7. Multi-Factor Stock Screening
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
    if change >= limit or change <= -limit:
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
    reasons: list[str] = []
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
    reasons: list[str] = []
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


def _score_capital_proxy(row: pd.Series) -> tuple[float, list[str]]:
    """资金面评分 (满分 25) — 基于内外盘和量价配合"""
    outer = float(row.get("外盘", 0) or 0)
    inner = float(row.get("内盘", 0) or 0)
    vol_ratio = float(row.get("量比", 1) or 1)
    change = float(row.get("涨跌幅", 0) or 0)
    reasons: list[str] = []
    score = 0.0

    if inner > 0:
        oi_ratio = outer / inner
        if oi_ratio > 1.5:
            score += 12
            reasons.append(f"主动买入强势(外/内{oi_ratio:.1f})")
        elif oi_ratio > 1.2:
            score += 9
            reasons.append("买盘偏强")
        elif oi_ratio > 1.0:
            score += 6
        elif oi_ratio > 0.8:
            score += 3
        else:
            score += 1
    else:
        score += 5

    if vol_ratio > 1.2 and change > 0:
        score += 8
        if vol_ratio > 2.0:
            reasons.append("量价齐升")
    elif vol_ratio > 1.2 and change <= 0:
        score += 2
    elif vol_ratio <= 1.2 and change > 0:
        score += 5
    else:
        score += 3

    if change > 2 and vol_ratio > 1.5:
        score += 5
        if not any("量价" in r for r in reasons):
            reasons.append("资金抢筹迹象")

    return min(score, 25), reasons


def _score_valuation(row: pd.Series) -> tuple[float, list[str]]:
    """估值与市值评分 (满分 15)"""
    pe = float(row.get("市盈率-动态", 0) or 0)
    pb = float(row.get("市净率", 0) or 0)
    mcap = float(row.get("总市值", 0) or 0)
    reasons: list[str] = []
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
    reasons: list[str] = []
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


def screen_stocks(stocks: pd.DataFrame, top_n: int = 10) -> list[dict]:
    """多因子评分模型筛选个股"""
    candidates = stocks[stocks.apply(_is_candidate, axis=1)].copy()
    print(
        f"  [SCREEN] {len(candidates)} candidates after filtering "
        f"(from {len(stocks)} total)"
    )

    results: list[dict] = []
    for _, row in candidates.iterrows():
        code = str(row.get("代码", ""))
        name = str(row.get("名称", ""))

        s1, r1 = _score_momentum(row)
        s2, r2 = _score_volume(row)
        s3, r3 = _score_capital_proxy(row)
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
# 8. Stage 2 — TimesFM Deep Prediction
# ================================================================

def _score_timesfm(
    current_price: float,
    point_forecast: np.ndarray,
    quantile_forecast: np.ndarray,
    signals: list[tuple],
) -> tuple[float, dict]:
    """TimesFM 综合评分 (满分 100).

    Components:
      - 预测收益率 (30 pts)
      - 风险收益比 (25 pts)
      - 策略共识   (25 pts)
      - 预测置信度 (20 pts)
    """
    from src.strategy import SIGNAL_BUY, SIGNAL_SELL

    score = 0.0
    details: dict = {}

    forecast_mean = float(np.mean(point_forecast))
    expected_return = (forecast_mean - current_price) / current_price
    details["expected_return"] = round(expected_return * 100, 2)

    if expected_return > 0.03:
        score += 30
    elif expected_return > 0.02:
        score += 24
    elif expected_return > 0.01:
        score += 18
    elif expected_return > 0.005:
        score += 12
    elif expected_return > 0:
        score += 7
    else:
        score += 2

    q_end = quantile_forecast[-1]
    q10 = float(q_end[1])
    q90 = float(q_end[9])
    upside = (q90 - current_price) / current_price
    downside = (current_price - q10) / current_price
    rr = upside / max(downside, 0.001)

    details["q10_return"] = round((q10 - current_price) / current_price * 100, 2)
    details["q90_return"] = round(upside * 100, 2)
    details["risk_reward"] = round(rr, 2)

    if rr > 3.0:
        score += 25
    elif rr > 2.0:
        score += 20
    elif rr > 1.5:
        score += 14
    elif rr > 1.0:
        score += 8
    else:
        score += 3

    buy_count = sum(1 for s, *_ in signals if s == SIGNAL_BUY)
    sell_count = sum(1 for s, *_ in signals if s == SIGNAL_SELL)
    strengths = [st for _, st, *_ in signals if st > 0]
    avg_strength = float(np.mean(strengths)) if strengths else 0.0

    details["buy_signals"] = buy_count
    details["sell_signals"] = sell_count
    details["signal_strength"] = round(avg_strength, 2)

    if buy_count == 3:
        score += 25
    elif buy_count == 2:
        score += 18
    elif buy_count == 1 and sell_count == 0:
        score += 12
    elif sell_count >= 2:
        score += 2
    else:
        score += 7

    uncertainty = (q90 - q10) / current_price
    details["uncertainty"] = round(uncertainty * 100, 2)

    if uncertainty < 0.03:
        score += 20
    elif uncertainty < 0.05:
        score += 16
    elif uncertainty < 0.08:
        score += 10
    elif uncertainty < 0.12:
        score += 5
    else:
        score += 2

    if buy_count >= 2 or (buy_count == 1 and expected_return > 0.01):
        details["direction"] = "看多"
        details["direction_icon"] = "🟢"
    elif sell_count >= 2 or expected_return < -0.01:
        details["direction"] = "看空"
        details["direction_icon"] = "🔴"
    else:
        details["direction"] = "中性"
        details["direction_icon"] = "⚪"

    details["point_forecast"] = [round(float(v), 2) for v in point_forecast]

    return min(score, 100), details


def _deep_predict(stage1_picks: list[dict], top_n: int) -> list[dict]:
    """Stage 2: 对 Stage 1 候选股做 TimesFM 深度预测 + 综合评分."""
    from src.data_loader import download_cn_data, compute_features
    from src.model_wrapper import create_predictor
    from src.strategy import create_strategies

    context_len = config.MODEL_CONFIG["context_length"]
    horizon = config.MODEL_CONFIG["horizon"]
    w1 = config.REPORT_CONFIG["stage1_weight"]
    w2 = config.REPORT_CONFIG["stage2_weight"]

    today = dt.date.today()
    lookback_days = int(context_len / 0.65) + 60
    start_date = (today - dt.timedelta(days=lookback_days)).isoformat()
    end_date = today.isoformat()

    print(f"  [DEEP] Fetching {len(stage1_picks)} stock histories ({start_date} → {end_date})...")
    histories: dict[str, dict] = {}
    features_dict: dict[str, dict] = {}
    for pick in stage1_picks:
        code = pick["code"]
        try:
            raw = download_cn_data(code, start_date, end_date)
            if raw and len(raw.get("close", [])) >= context_len:
                features_dict[code] = compute_features(raw)
                histories[code] = raw
        except Exception as exc:
            print(f"  [DEEP] {code} fetch failed: {exc}")
        time.sleep(0.15)

    fetched = len(histories)
    print(f"  [DEEP] ✓ {fetched}/{len(stage1_picks)} histories loaded")

    if fetched == 0:
        print("  [DEEP] No history data, falling back to Stage 1 only")
        for p in stage1_picks:
            p["final_score"] = round(p["total_score"] * w1, 1)
            p["tfm_score"] = 0.0
            p["tfm_details"] = None
        return stage1_picks[:top_n]

    print("  [DEEP] Loading predictor...")
    try:
        predictor = create_predictor()
    except Exception as exc:
        print(f"  [DEEP] Predictor init failed: {exc}, falling back to Stage 1")
        for p in stage1_picks:
            p["final_score"] = round(p["total_score"] * w1, 1)
            p["tfm_score"] = 0.0
            p["tfm_details"] = None
        return stage1_picks[:top_n]

    codes_ordered = list(histories.keys())
    history_arrays = [
        histories[c]["close"][-context_len:] for c in codes_ordered
    ]

    print(f"  [DEEP] Batch predicting {len(codes_ordered)} stocks (horizon={horizon})...")
    t0 = time.time()
    try:
        point_forecasts, quantile_forecasts = predictor.predict_batch(
            history_arrays, horizon
        )
    except Exception as exc:
        print(f"  [DEEP] Batch prediction failed: {exc}")
        for p in stage1_picks:
            p["final_score"] = round(p["total_score"] * w1, 1)
            p["tfm_score"] = 0.0
            p["tfm_details"] = None
        return stage1_picks[:top_n]
    print(f"  [DEEP] ✓ Predictions done in {time.time() - t0:.1f}s")

    strategies = create_strategies()
    code_to_tfm: dict[str, tuple[float, dict]] = {}

    for i, code in enumerate(codes_ordered):
        raw = histories[code]
        feat = features_dict[code]
        close_arr = raw["close"]
        current_price = float(close_arr[-1])
        n = len(close_arr)

        feat_snapshot: dict = {}
        for k, v in feat.items():
            if isinstance(v, np.ndarray) and n - 1 < len(v):
                feat_snapshot[k] = float(v[n - 1])

        point = point_forecasts[i]
        quantile = quantile_forecasts[i]

        signals = []
        for strategy in strategies:
            sig, strength, reason = strategy.generate_signal(
                current_price, point, quantile, feat_snapshot,
            )
            signals.append((sig, strength, reason, strategy.name))

        tfm_score, tfm_details = _score_timesfm(
            current_price, point, quantile, signals,
        )
        code_to_tfm[code] = (tfm_score, tfm_details)

    for pick in stage1_picks:
        code = pick["code"]
        if code in code_to_tfm:
            tfm_score, tfm_details = code_to_tfm[code]
            pick["tfm_score"] = round(tfm_score, 1)
            pick["tfm_details"] = tfm_details
            pick["final_score"] = round(
                pick["total_score"] * w1 + tfm_score * w2, 1
            )
        else:
            pick["tfm_score"] = 0.0
            pick["tfm_details"] = None
            pick["final_score"] = round(pick["total_score"] * w1, 1)

    stage1_picks.sort(key=lambda x: x["final_score"], reverse=True)

    promoted = sum(
        1 for i, p in enumerate(stage1_picks[:top_n])
        if p.get("tfm_details") and p["tfm_details"]["direction"] == "看多"
    )
    print(f"  [DEEP] ✓ Final ranking done — {promoted}/{top_n} stocks marked bullish")

    return stage1_picks[:top_n]


# ================================================================
# 9. Risk Assessment
# ================================================================

def _assess_risks(overview: dict, style: dict) -> list[str]:
    warnings: list[str] = []

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
# 9. Trend Prediction
# ================================================================

def _predict_trend(overview: dict, style: dict, sectors: dict) -> list[str]:
    predictions: list[str] = []

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
            predictions.append(
                f"板块方面，{top_sector}领涨且涨幅较大，明日关注板块内滞涨个股的跟风机会"
            )
        elif top_change > 1:
            predictions.append(
                f"板块方面，{top_sector}等板块走强，若资金持续流入可保持关注"
            )

    inflow = sectors.get("inflow_industries", [])
    if inflow and inflow[0].get("net_inflow_yi", 0) > 5:
        predictions.append(
            f"资金面上，{inflow[0]['name']}获资金青睐"
            f"(板块成交 {inflow[0]['net_inflow_yi']:.0f}亿)，后续可能延续强势"
        )

    return predictions


# ================================================================
# 10. Console Report Builder
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
    lines: list[str] = []
    w = 64

    lines.append(f"\n{'═' * w}")
    lines.append(f"  📊 A股短线市场分析报告 — {report_time}")
    lines.append(f"{'═' * w}")

    if indices:
        lines.append(f"\n  {_BOLD}▌ 主要指数{_RESET}")
        for idx in indices:
            c = _GREEN if idx["change_pct"] > 0 else _RED if idx["change_pct"] < 0 else _YELLOW
            arrow = "▲" if idx["change_pct"] > 0 else "▼" if idx["change_pct"] < 0 else "─"
            lines.append(
                f"  ├─ {idx['name']:<8s} {idx['close']:>10.2f}  "
                f"{c}{arrow} {idx['change_pct']:+.2f}%{_RESET}"
            )

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
    lines.append(
        f"  └─ 市场情绪: {_BOLD}{overview['sentiment']}{_RESET} "
        f"(评分 {overview['sentiment_score']:.0f})"
    )

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

    lines.append(f"\n  {_BOLD}▌ 行业板块 TOP8 (按涨幅){_RESET}")
    hot = sectors.get("hot_industries", [])[:8]
    if hot and "up_ratio" in hot[0]:
        lines.append(
            f"  │  {'排名':^4s}  {'板块':<10s} {'涨幅':>8s}  "
            f"{'上涨占比':>8s}  {'成交额':>8s}"
        )
        lines.append(f"  │  {'─' * 46}")
        for i, s in enumerate(hot, 1):
            medal = "🥇🥈🥉"[i - 1] if i <= 3 else f"  {i}"
            c = _GREEN if s["change_pct"] > 0 else _RED
            amt = f"{s.get('total_amount_yi', 0):.0f}亿" if "total_amount_yi" in s else ""
            ratio = f"{s.get('up_ratio', 0):.0f}%" if "up_ratio" in s else ""
            lines.append(
                f"  │  {medal}  {s['name']:<10s} "
                f"{c}{s['change_pct']:>+7.2f}%{_RESET}  "
                f"{ratio:>8s}  {amt:>8s}"
            )
    else:
        lines.append(f"  │  {'排名':^4s}  {'板块':<10s} {'涨幅':>8s}")
        lines.append(f"  │  {'─' * 28}")
        for i, s in enumerate(hot, 1):
            medal = "🥇🥈🥉"[i - 1] if i <= 3 else f"  {i}"
            c = _GREEN if s["change_pct"] > 0 else _RED
            lines.append(
                f"  │  {medal}  {s['name']:<10s} "
                f"{c}{s['change_pct']:>+7.2f}%{_RESET}"
            )

    if predictions:
        lines.append(f"\n  {_BOLD}▌ 短线趋势预判{_RESET}")
        for i, p in enumerate(predictions):
            prefix = "└" if i == len(predictions) - 1 else "├"
            lines.append(f"  {prefix}─ 🔮 {p}")

    has_tfm = any(p.get("tfm_details") for p in picks)
    score_label = "综合" if has_tfm else "评分"

    lines.append(f"\n  {_BOLD}▌ 精选个股 TOP{len(picks)}{_RESET}")
    lines.append(
        f"  │  {'排名':^4s}  {'代码':<8s} {'名称':<8s} {'现价':>8s} "
        f"{'涨幅':>7s} {score_label:>5s} {'量比':>5s} {'换手':>6s} {'成交额':>7s}"
    )
    lines.append(f"  │  {'─' * 64}")
    medals = ["🥇", "🥈", "🥉"]
    for i, p in enumerate(picks, 1):
        medal = medals[i - 1] if i <= 3 else f"  {i}"
        c = _GREEN if p["change_pct"] > 0 else _RED if p["change_pct"] < 0 else ""
        display_score = p.get("final_score", p["total_score"])
        lines.append(
            f"  │  {medal}  {p['code']:<8s} {p['name']:<8s} "
            f"¥{p['price']:>7.2f} {c}{p['change_pct']:>+6.2f}%{_RESET} "
            f"{display_score:>5.1f} "
            f"{p['volume_ratio']:>5.1f} "
            f"{p['turnover_rate']:>5.1f}% "
            f"{p['amount_yi']:>6.1f}亿"
        )
        if p["reasons"]:
            tags = " ".join(f"✦{r}" for r in p["reasons"][:4])
            lines.append(f"  │       {_DIM}{tags}{_RESET}")
        tfm = p.get("tfm_details")
        if tfm:
            icon = tfm["direction_icon"]
            exp_r = tfm["expected_return"]
            q10r = tfm["q10_return"]
            q90r = tfm["q90_return"]
            buy_n = tfm["buy_signals"]
            total_n = buy_n + tfm["sell_signals"] + (3 - buy_n - tfm["sell_signals"])
            exp_c = _GREEN if exp_r > 0 else _RED
            lines.append(
                f"  │       🔮 TimesFM: {icon}{tfm['direction']} "
                f"预测{exp_c}{exp_r:+.1f}%{_RESET} "
                f"区间[{q10r:+.1f}%,{q90r:+.1f}%] "
                f"策略共识{buy_n}/{total_n}"
            )

    lines.append(f"\n  {_BOLD}▌ 风险提示{_RESET}")
    for i, r in enumerate(risks):
        prefix = "└" if i == len(risks) - 1 else "├"
        lines.append(f"  {prefix}─ ⚠ {r}")

    lines.append(f"\n{'━' * w}")
    lines.append("  TimesFM Quant · 短线市场分析报告 · 仅供参考，不构成投资建议")
    lines.append(f"{'━' * w}\n")

    return "\n".join(lines)


# ================================================================
# 11. Feishu Card Builder
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
    elements: list[dict] = []

    idx_lines = []
    for idx in indices[:4]:
        arrow = "📈" if idx["change_pct"] > 0 else "📉" if idx["change_pct"] < 0 else "➡️"
        idx_lines.append(
            f"{arrow} {idx['name']} {idx['close']:.2f} ({idx['change_pct']:+.2f}%)"
        )
    if idx_lines:
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": "\n".join(idx_lines)},
        })

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

    elements.append({"tag": "hr"})
    style_text = f"**🎯 市场风格: {style['style_cn']}**\n"
    style_text += (
        f"大盘{style['large_chg']:+.2f}% | "
        f"中盘{style['mid_chg']:+.2f}% | "
        f"小盘{style['small_chg']:+.2f}%\n"
    )
    style_text += f"成长{style['growth_chg']:+.2f}% | 价值{style['value_chg']:+.2f}%"
    elements.append({"tag": "div", "text": {"tag": "lark_md", "content": style_text}})

    hot = sectors.get("hot_industries", [])[:5]
    if hot:
        elements.append({"tag": "hr"})
        sector_lines = ["**🔥 行业热点**"]
        for i, s in enumerate(hot, 1):
            extra = ""
            if "up_ratio" in s:
                extra = f" | 上涨占比{s['up_ratio']:.0f}%"
            sector_lines.append(f"{i}. {s['name']} {s['change_pct']:+.2f}%{extra}")
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": "\n".join(sector_lines)},
        })

    if predictions:
        elements.append({"tag": "hr"})
        pred_lines = ["**🔮 短线趋势预判**"]
        for p in predictions[:3]:
            pred_lines.append(f"• {p}")
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": "\n".join(pred_lines)},
        })

    elements.append({"tag": "hr"})
    pick_lines = [f"**🏆 精选个股 TOP{min(len(picks), 8)}**"]
    medals_emoji = ["🥇", "🥈", "🥉"]
    for i, p in enumerate(picks[:8], 1):
        medal = medals_emoji[i - 1] if i <= 3 else f"{i}."
        tags = " ".join(f"✦{r}" for r in p["reasons"][:2])
        display_score = p.get("final_score", p["total_score"])
        line = (
            f"{medal} **{p['code']} {p['name']}** ¥{p['price']:.2f} "
            f"({p['change_pct']:+.2f}%) 评分:`{display_score:.0f}`\n"
            f"   量比{p['volume_ratio']:.1f} | 换手{p['turnover_rate']:.1f}% "
            f"| {p['amount_yi']:.1f}亿  {tags}"
        )
        tfm = p.get("tfm_details")
        if tfm:
            line += (
                f"\n   🔮 {tfm['direction_icon']}{tfm['direction']} "
                f"预测{tfm['expected_return']:+.1f}% "
                f"区间[{tfm['q10_return']:+.1f}%,{tfm['q90_return']:+.1f}%] "
                f"共识{tfm['buy_signals']}/3"
            )
        pick_lines.append(line)
    elements.append({
        "tag": "div",
        "text": {"tag": "lark_md", "content": "\n".join(pick_lines)},
    })

    elements.append({"tag": "hr"})
    risk_text = "**⚠️ 风险提示**\n" + "\n".join(f"• {r}" for r in risks[:3])
    elements.append({"tag": "div", "text": {"tag": "lark_md", "content": risk_text}})

    elements.append({
        "tag": "note",
        "elements": [{
            "tag": "plain_text",
            "content": "TimesFM Quant · 短线市场分析报告 · 仅供参考，不构成投资建议",
        }],
    })

    header_color = (
        "green" if overview["avg_change"] > 0.3
        else "red" if overview["avg_change"] < -0.3
        else "blue"
    )
    card = {
        "header": {
            "title": {
                "tag": "plain_text",
                "content": (
                    f"📊 A股短线市场研报 — {report_time} | {overview['sentiment']}"
                ),
            },
            "template": header_color,
        },
        "elements": elements,
    }
    return {"msg_type": "interactive", "card": card}


# ================================================================
# 12. Feishu Sender (reuse config)
# ================================================================

def _send_feishu(payload: dict) -> None:
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
        hmac_code = hmac.new(
            string_to_sign.encode(), digestmod=hashlib.sha256
        ).digest()
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
        from lark_oapi.api.im.v1 import (
            CreateMessageRequest,
            CreateMessageRequestBody,
        )

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
# 13. Report File Saver
# ================================================================

def _save_report_text(text: str, report_time: str) -> str | None:
    try:
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        date_str = report_time.replace("-", "").replace(":", "").replace(" ", "_")
        path = os.path.join(_OUTPUT_DIR, f"market_report_{date_str}.txt")
        clean = re.sub(r"\033\[[0-9;]*m", "", text)
        with open(path, "w", encoding="utf-8") as f:
            f.write(clean)
        print(f"  [REPORT] Saved: {path}")
        return path
    except Exception as e:
        print(f"  [REPORT] Save failed: {e}")
        return None


# ================================================================
# 14. Main Entry
# ================================================================

def run(top_n: int = 10) -> dict:
    """执行完整的市场分析流程并推送报告"""
    print("=" * 60)
    print(f"  📊 A股短线市场分析报告 — {dt.date.today()}")
    print("=" * 60)

    start_time = time.time()
    report_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Step 1: 拉取全量实时行情 ──
    print("\n[STEP 1/5] Fetching real-time quotes (Tencent) + industry mapping (East Money)...")
    codes = _generate_all_codes()
    print(f"  [FETCH] {len(codes)} candidate codes generated")

    all_stocks = _fetch_tencent_quotes(codes)
    if all_stocks.empty:
        print("[REPORT] ✗ Failed to fetch stock data, aborting.")
        return {}
    print(f"  [FETCH] ✓ {len(all_stocks)} valid stocks loaded")

    sector_map = _fetch_industry_mapping()
    mapped = sum(1 for c in all_stocks["代码"] if c in sector_map)
    print(f"  [FETCH] ✓ Industry mapping: {len(sector_map)} codes, {mapped} matched")

    sina_sectors = _fetch_sina_sector_summary()

    print("\n[STEP 1/5] Fetching index data (AKShare)...")
    indices = _fetch_major_indices()
    print(f"  [FETCH] ✓ {len(indices)} indices loaded")

    # ── Step 2: 大盘概览 & 风格 ──
    print("\n[STEP 2/5] Analyzing market overview & style...")
    overview = _analyze_overview(all_stocks)
    style = _analyze_style(all_stocks)

    # ── Step 3: 板块热点 ──
    print("[STEP 3/5] Analyzing sector hotspots...")
    sectors = _analyze_sectors(all_stocks, sector_map, sina_sectors)

    # ── Step 4: 个股筛选 (两阶段) ──
    use_tfm = config.REPORT_CONFIG["use_timesfm_screening"]
    pool_size = config.REPORT_CONFIG["stage1_pool_size"] if use_tfm else top_n

    print(f"\n[STEP 4/6] Stage 1 — Multi-factor screening (pool={pool_size})...")
    stage1_picks = screen_stocks(all_stocks, pool_size)

    if use_tfm and stage1_picks:
        print(f"\n[STEP 5/6] Stage 2 — TimesFM deep prediction...")
        picks = _deep_predict(stage1_picks, top_n)
    else:
        for p in stage1_picks:
            p["final_score"] = p["total_score"]
            p["tfm_score"] = 0.0
            p["tfm_details"] = None
        picks = stage1_picks[:top_n]

    # ── Step 6: 生成 & 推送报告 ──
    risks = _assess_risks(overview, style)
    predictions = _predict_trend(overview, style, sectors)

    print(f"\n[STEP 6/6] Building and sending report...")

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
    parser.add_argument(
        "--top", type=int, default=10, help="精选个股数量 (default: 10)"
    )
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
