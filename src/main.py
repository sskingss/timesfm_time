"""
TimesFM 量化交易回测系统 - 主程序
===================================
基于 Google Research TimesFM 2.5 时间序列基础模型的量化交易回测框架
支持美股（yfinance）和 A 股（akshare）真实行情 + 宏观因子分析

使用方法:
  python -m src.main                   # 默认使用真实 TimesFM 模型
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.data_loader import load_data
from src.model_wrapper import create_predictor, MultiHorizonPredictor
from src.macro_analyzer import create_macro_analyzer
from src.strategy import create_strategies
from src.backtester import run_backtest
from src.visualizer import generate_all_charts


def print_banner():
    print("=" * 72)
    print("  TimesFM Quantitative Trading Backtest System")
    print("  Based on Google Research TimesFM 2.5 (200M params)")
    print("  ICML 2024 | Decoder-only Foundation Model for Time Series")
    print("  Supports: US Stocks (yfinance) + China A-Shares (akshare)")
    print("=" * 72)


def print_metrics_table(all_results):
    """打印汇总绩效表格"""
    print("\n" + "=" * 100)
    print(f"{'Symbol':<12} {'Strategy':<22} {'Return':>10} {'AnnRet':>10} {'Sharpe':>8} "
          f"{'MaxDD':>8} {'WinRate':>8} {'Trades':>7} {'PF':>7}")
    print("-" * 100)

    for r in all_results:
        m = r["metrics"]
        print(f"{r['symbol']:<12} {r['strategy']:<22} {m['total_return']:>9.2%} "
              f"{m['ann_return']:>9.2%} {m['sharpe_ratio']:>7.2f} "
              f"{m['max_drawdown']:>7.2%} {m['win_rate']:>7.0%} "
              f"{m['total_trades']:>7d} {m['profit_factor']:>6.2f}")

    print("=" * 100)

    best = max(all_results, key=lambda r: r["metrics"]["sharpe_ratio"])
    print(f"\n  [BEST] Highest Sharpe: {best['strategy']} on {best['symbol']} "
          f"(Sharpe={best['metrics']['sharpe_ratio']:.2f}, "
          f"Return={best['metrics']['total_return']:.2%})")

    worst_dd = min(all_results, key=lambda r: -r["metrics"]["max_drawdown"])
    print(f"  [RISK] Lowest MaxDD: {worst_dd['strategy']} on {worst_dd['symbol']} "
          f"(MaxDD={worst_dd['metrics']['max_drawdown']:.2%})")


def main():
    start_time = time.time()
    print_banner()

    # Step 1: 加载真实行情数据
    print("\n[STEP 1] Loading real market data...")
    data_dict = load_data()

    # Step 2: 宏观经济分析
    print("\n[STEP 2] Analyzing macro environment...")
    macro = create_macro_analyzer()
    macro.load_scores(data_dict)

    # Step 3: 初始化预测模型
    print("\n[STEP 3] Initializing prediction model...")
    predictor = create_predictor()

    # Step 4: 创建策略
    print("\n[STEP 4] Creating trading strategies...")
    strategies = create_strategies()
    for s in strategies:
        print(f"  - {s.name}")

    # Step 5: 执行回测
    print("\n[STEP 5] Running backtests...")
    all_results = run_backtest(data_dict, strategies, predictor)

    # Step 6: 打印绩效报告
    print("\n[STEP 6] Performance Report")
    print_metrics_table(all_results)

    # Step 7: 生成图表
    print("\n[STEP 7] Generating charts...")
    chart_paths = generate_all_charts(all_results, data_dict, predictor)
    print(f"\n  Total charts generated: {len(chart_paths)}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 72}")
    print(f"  Backtest completed in {elapsed:.1f}s")
    print(f"  Output directory: {config.VIS_CONFIG['output_dir']}")
    print(f"{'=' * 72}")

    return all_results


if __name__ == "__main__":
    results = main()
