"""
实时信号扫描器
==============
加载最新行情 → 模型预测 → 策略生成信号 → 通过插件分发

用法:
  # 每日扫描（自动拉取截至今天的最新行情）
  python -m src.signal_runner

  # 指定标的
  python -m src.signal_runner --symbols 601766 600519

  # 指定策略
  python -m src.signal_runner --strategy 分位数风控策略

  # 只推送买卖信号，不推送 HOLD
  python -m src.signal_runner --actionable-only
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.data_loader import load_data
from src.model_wrapper import create_predictor
from src.macro_analyzer import create_macro_analyzer
from src.strategy import create_strategies, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD
from src.plugins.base import SignalEvent
from src.plugins import create_plugins_from_config
from src.signal_dispatcher import SignalDispatcher


def _build_signal_event(
    symbol: str,
    strategy_name: str,
    signal: int,
    strength: float,
    reason: str,
    current_price: float,
    point_forecast: np.ndarray | None,
    quantile_forecast: np.ndarray | None,
    feat_snapshot: dict,
) -> SignalEvent:
    """把策略输出打包成 SignalEvent"""
    q_summary = None
    if quantile_forecast is not None and len(quantile_forecast) > 0:
        q_end = quantile_forecast[min(len(quantile_forecast) - 1, 9)]
        q_summary = {
            "q10": round(float(q_end[1]), 4),
            "q50": round(float(q_end[5]), 4),
            "q90": round(float(q_end[9]), 4),
        }

    return SignalEvent(
        timestamp=dt.datetime.now(),
        symbol=symbol,
        strategy=strategy_name,
        signal=signal,
        strength=strength,
        reason=reason,
        current_price=float(current_price),
        point_forecast=(
            [round(float(v), 4) for v in point_forecast]
            if point_forecast is not None else None
        ),
        quantile_summary=q_summary,
        features={k: float(v) for k, v in feat_snapshot.items()
                  if isinstance(v, (int, float, np.floating, np.integer))},
    )


def _prepare_live_dates():
    """
    设置实时扫描的日期范围：
    - end_date = 今天
    - start_date = 往前推足够长（保证 context_length + warmup 有足够数据）
    直接覆盖 DATA_CONFIG，让 load_data 拉取最新行情。
    """
    today = dt.date.today().isoformat()
    context_len = config.MODEL_CONFIG["context_length"]
    # 交易日约占自然日 70%，多留余量
    lookback_days = int(context_len / 0.65) + 60
    start = (dt.date.today() - dt.timedelta(days=lookback_days)).isoformat()

    config.DATA_CONFIG["start_date"] = start
    config.DATA_CONFIG["end_date"] = today
    print(f"[SCAN] Live mode: data range {start} → {today}")


def scan_signals(
    symbols: list[str] | None = None,
    strategy_filter: str | None = None,
    actionable_only: bool = False,
) -> list[SignalEvent]:
    """
    扫描最新信号。

    流程：
    1. 覆盖日期为实时范围，拉取最新行情
    2. 宏观评分注入
    3. 对每个标的取最近 context_length 窗口做预测
    4. 每个策略生成信号
    5. 返回 SignalEvent 列表
    """
    _prepare_live_dates()

    print("[SCAN] Loading latest market data...")
    data_dict = load_data(symbols)

    print("[SCAN] Analyzing macro environment...")
    macro = create_macro_analyzer()
    macro.load_scores(data_dict)

    print("[SCAN] Initializing predictor...")
    predictor = create_predictor()

    all_strategies = create_strategies()
    if strategy_filter:
        all_strategies = [s for s in all_strategies if strategy_filter in s.name]
        if not all_strategies:
            print(f"[SCAN] Warning: no strategy matches '{strategy_filter}', using all")
            all_strategies = create_strategies()

    context_len = config.MODEL_CONFIG["context_length"]
    horizon = config.MODEL_CONFIG["horizon"]

    events: list[SignalEvent] = []

    for symbol, sdata in data_dict.items():
        close = sdata["raw"]["close"]
        features = sdata["features"]
        dates = sdata["raw"].get("dates")
        n = len(close)

        if n < context_len:
            print(f"[SCAN] {symbol}: only {n} days, need {context_len}, skipping")
            continue

        current_price = close[-1]
        last_date = dates[-1] if dates is not None and len(dates) > 0 else "unknown"
        print(f"[SCAN] {symbol}: latest price ¥{current_price:.2f} ({last_date}), {n} days loaded")

        history = close[-context_len:]

        try:
            point_forecast, quantile_forecast = predictor.predict(history, horizon)
        except Exception as e:
            print(f"[SCAN] {symbol}: prediction failed: {e}")
            point_forecast, quantile_forecast = None, None

        feat_snapshot = {}
        for k, v in features.items():
            if isinstance(v, np.ndarray) and n - 1 < len(v):
                feat_snapshot[k] = v[n - 1]

        for strategy in all_strategies:
            signal, strength, reason = strategy.generate_signal(
                current_price, point_forecast, quantile_forecast, feat_snapshot,
            )

            if actionable_only and signal == SIGNAL_HOLD:
                continue

            event = _build_signal_event(
                symbol=symbol,
                strategy_name=strategy.name,
                signal=signal,
                strength=strength,
                reason=reason,
                current_price=current_price,
                point_forecast=point_forecast,
                quantile_forecast=quantile_forecast,
                feat_snapshot=feat_snapshot,
            )
            events.append(event)

    return events


def run(
    symbols: list[str] | None = None,
    strategy_filter: str | None = None,
    actionable_only: bool = False,
) -> list[SignalEvent]:
    """完整的 扫描 → 分发 流程"""
    print("=" * 60)
    print(f"  TimesFM Signal Scanner — {dt.date.today()}")
    print("=" * 60)

    start = time.time()

    plugins = create_plugins_from_config(config.PLUGIN_CONFIG)
    dispatcher = SignalDispatcher()
    dispatcher.register_many(plugins)
    dispatcher.startup()

    print(f"[SCAN] Active plugins: {dispatcher.plugin_names}")

    events = scan_signals(symbols, strategy_filter, actionable_only)

    actionable = [e for e in events if e.is_actionable]
    hold = [e for e in events if not e.is_actionable]
    symbols_count = len({e.symbol for e in events})
    print(f"\n[SCAN] {len(events)} signal(s) across {symbols_count} symbol(s): "
          f"{len(actionable)} actionable, {len(hold)} hold")
    print(f"[SCAN] Dispatching (grouped by symbol)...\n")

    dispatcher.dispatch_batch(events)

    dispatcher.shutdown()

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  Scan completed in {elapsed:.1f}s — {len(actionable)} actionable signal(s)")
    print(f"{'=' * 60}")

    return events


def main():
    parser = argparse.ArgumentParser(description="TimesFM Signal Scanner")
    parser.add_argument("--symbols", nargs="+", help="Override symbols to scan")
    parser.add_argument("--strategy", type=str, help="Filter by strategy name (partial match)")
    parser.add_argument("--actionable-only", action="store_true",
                        help="Only dispatch BUY/SELL signals, skip HOLD")
    args = parser.parse_args()

    run(
        symbols=args.symbols,
        strategy_filter=args.strategy,
        actionable_only=args.actionable_only,
    )


if __name__ == "__main__":
    main()
