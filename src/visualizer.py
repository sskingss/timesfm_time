"""
可视化模块
生成回测报告所需的各类图表
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import config

import platform
if platform.system() == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _savefig(fig, filename):
    """保存图片到输出目录"""
    import os
    outdir = config.VIS_CONFIG["output_dir"]
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=config.VIS_CONFIG["dpi"], bbox_inches='tight')
    plt.close(fig)
    print(f"  [VIS] Saved: {path}")
    return path


def plot_equity_curves(results, symbol):
    """
    图1：权益曲线对比图
    多策略在同一股票上的权益曲线
    """
    fig, ax = plt.subplots(figsize=config.VIS_CONFIG["figsize"])
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']

    for i, r in enumerate(results):
        if r["symbol"] != symbol:
            continue
        warmup = r["warmup"]
        eq = r["equity_curve"][warmup:]
        label = f'{r["strategy"]} (Return: {r["metrics"]["total_return"]:.1%})'
        ax.plot(eq, label=label, color=colors[i % len(colors)], linewidth=1.5)

    # 基准：Buy & Hold
    for r in results:
        if r["symbol"] == symbol:
            warmup = r["warmup"]
            break
    # 找到对应数据
    init_cap = config.TRADING_CONFIG["initial_capital"]
    ax.axhline(y=init_cap, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')

    ax.set_title(f'Equity Curves - {symbol}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    return _savefig(fig, f'equity_curves_{symbol}.png')


def plot_signals_on_price(result):
    """
    图2：交易信号叠加在价格图上
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1], sharex=True)
    symbol = result["symbol"]
    warmup = result["warmup"]

    # 获取价格数据 (从回测区间开始)
    eq = result["equity_curve"]
    n = len(eq)
    x = np.arange(warmup, n)

    # 我们没有直接存储 close 在 result 中, 用 equity 替代展示
    ax1.plot(x, eq[warmup:], color='#333333', linewidth=1, label='Portfolio Value')

    # 标注买卖信号
    buy_signals = [s for s in result["signals"] if s["signal"] == 1]
    sell_signals = [s for s in result["signals"] if s["signal"] == -1]

    if buy_signals:
        buy_x = [s["idx"] for s in buy_signals]
        buy_y = [eq[s["idx"]] for s in buy_signals if s["idx"] < n]
        ax1.scatter(buy_x[:len(buy_y)], buy_y, marker='^', color='#4CAF50', s=80, zorder=5, label='Buy')

    if sell_signals:
        sell_x = [s["idx"] for s in sell_signals]
        sell_y = [eq[s["idx"]] for s in sell_signals if s["idx"] < n]
        ax1.scatter(sell_x[:len(sell_y)], sell_y, marker='v', color='#F44336', s=80, zorder=5, label='Sell')

    ax1.set_title(f'Trading Signals - {result["strategy"]} on {symbol}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 下图：信号强度
    all_idx = [s["idx"] for s in result["signals"]]
    all_strength = [s["strength"] * s["signal"] for s in result["signals"]]
    colors_bar = ['#4CAF50' if v > 0 else '#F44336' if v < 0 else '#999' for v in all_strength]
    ax2.bar(all_idx, all_strength, color=colors_bar, alpha=0.7, width=2)
    ax2.set_ylabel('Signal Strength', fontsize=11)
    ax2.set_xlabel('Day Index', fontsize=11)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    return _savefig(fig, f'signals_{result["strategy"].replace(" ", "_")}_{symbol}.png')


def plot_drawdown(results, symbol):
    """
    图3：回撤曲线对比
    """
    fig, ax = plt.subplots(figsize=config.VIS_CONFIG["figsize"])
    colors = ['#2196F3', '#FF5722', '#4CAF50']

    for i, r in enumerate(results):
        if r["symbol"] != symbol:
            continue
        warmup = r["warmup"]
        eq = r["equity_curve"][warmup:]
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak * 100  # 百分比
        ax.fill_between(range(len(dd)), dd, alpha=0.2, color=colors[i % len(colors)])
        ax.plot(dd, label=f'{r["strategy"]} (MaxDD: {r["metrics"]["max_drawdown"]:.1%})',
                color=colors[i % len(colors)], linewidth=1.2)

    ax.set_title(f'Drawdown Curves - {symbol}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    return _savefig(fig, f'drawdown_{symbol}.png')


def plot_performance_comparison(all_results):
    """
    图4：各策略绩效对比柱状图
    """
    # 按策略分组
    strategy_names = list(set(r["strategy"] for r in all_results))
    metrics_to_plot = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]
    metric_labels = ["Total Return", "Sharpe Ratio", "Max Drawdown", "Win Rate"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#2196F3', '#FF5722', '#4CAF50']

    for ax_idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[ax_idx // 2][ax_idx % 2]
        x = np.arange(len(strategy_names))
        width = 0.35

        symbols = list(set(r["symbol"] for r in all_results))
        for si, sym in enumerate(symbols):
            values = []
            for sname in strategy_names:
                matched = [r for r in all_results if r["strategy"] == sname and r["symbol"] == sym]
                if matched:
                    v = matched[0]["metrics"][metric]
                    if metric in ["total_return", "max_drawdown", "win_rate"]:
                        v *= 100  # 转百分比
                    values.append(v)
                else:
                    values.append(0)
            offset = (si - len(symbols) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=sym, color=colors[si % len(colors)], alpha=0.8)

        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s[:8] for s in strategy_names], fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        if metric in ["total_return", "max_drawdown", "win_rate"]:
            ax.set_ylabel('%')

    fig.suptitle('Strategy Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    return _savefig(fig, 'performance_comparison.png')


def plot_prediction_vs_actual(data, predictor, symbol):
    """
    图5：预测 vs 实际对比图
    在随机选取的时间点做预测，与真实走势对比
    """
    close = data["raw"]["close"]
    n = len(close)
    context_len = config.MODEL_CONFIG["context_length"]
    horizon = config.MODEL_CONFIG["horizon"]

    fig, ax = plt.subplots(figsize=config.VIS_CONFIG["figsize"])

    # 绘制真实价格
    ax.plot(close, color='#333', linewidth=1, label='Actual Price', alpha=0.7)

    # 选取 5 个预测点
    pred_points = np.linspace(context_len + 50, n - horizon - 10, 5, dtype=int)
    pred_colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']

    for pi, start_idx in enumerate(pred_points):
        history = close[:start_idx]
        point_forecast, quantile_forecast = predictor.predict(history, horizon)

        pred_x = np.arange(start_idx, start_idx + horizon)
        ax.plot(pred_x, point_forecast, color=pred_colors[pi], linewidth=2, linestyle='--',
                label=f'Pred @day {start_idx}' if pi < 3 else None)

        # 分位数区间
        if quantile_forecast is not None and quantile_forecast.shape[1] >= 10:
            q10 = quantile_forecast[:, 1]
            q90 = quantile_forecast[:, 9]
            ax.fill_between(pred_x, q10, q90, alpha=0.15, color=pred_colors[pi])

        # 标注预测起始点
        ax.scatter([start_idx], [close[start_idx]], color=pred_colors[pi], s=50, zorder=5)

    ax.set_title(f'Prediction vs Actual - {symbol}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Day Index', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    return _savefig(fig, f'prediction_vs_actual_{symbol}.png')


def plot_trade_distribution(all_results):
    """
    图6：交易盈亏分布图
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    strategy_names = list(set(r["strategy"] for r in all_results))

    for i, sname in enumerate(strategy_names):
        ax = axes[i] if i < 3 else axes[0]
        all_pnl = []
        for r in all_results:
            if r["strategy"] == sname:
                pnls = [t.pnl_pct * 100 for t in r["trades"]]
                all_pnl.extend(pnls)

        if all_pnl:
            colors_h = ['#4CAF50' if p > 0 else '#F44336' for p in all_pnl]
            ax.bar(range(len(all_pnl)), all_pnl, color=colors_h, alpha=0.7)
            ax.axhline(y=0, color='black', linewidth=0.8)
            avg_pnl = np.mean(all_pnl)
            ax.axhline(y=avg_pnl, color='#2196F3', linestyle='--', label=f'Avg: {avg_pnl:.1f}%')

        ax.set_title(sname[:12], fontsize=12, fontweight='bold')
        ax.set_ylabel('PnL (%)')
        ax.set_xlabel('Trade #')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Trade PnL Distribution by Strategy', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return _savefig(fig, 'trade_distribution.png')


def generate_all_charts(all_results, data_dict, predictor):
    """
    生成所有图表

    Returns:
        list[str]: 生成的图片路径列表
    """
    paths = []
    symbols = list(set(r["symbol"] for r in all_results))

    for sym in symbols:
        sym_results = [r for r in all_results if r["symbol"] == sym]

        # 图1: 权益曲线
        paths.append(plot_equity_curves(all_results, sym))

        # 图2: 交易信号 (只画第一个策略)
        if sym_results:
            paths.append(plot_signals_on_price(sym_results[0]))

        # 图3: 回撤曲线
        paths.append(plot_drawdown(all_results, sym))

        # 图5: 预测 vs 实际
        if sym in data_dict:
            paths.append(plot_prediction_vs_actual(data_dict[sym], predictor, sym))

    # 图4: 绩效对比
    paths.append(plot_performance_comparison(all_results))

    # 图6: 交易分布
    paths.append(plot_trade_distribution(all_results))

    return paths
