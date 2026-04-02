"""
回测引擎模块
- 滚动窗口回测
- 完整交易模拟（手续费、滑点、止损止盈）
- 绩效指标计算
"""
import numpy as np
from . import config
from .strategy import SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD


class Position:
    """持仓信息"""
    def __init__(self, direction, entry_price, shares, entry_idx):
        self.direction = direction     # 1 = long, -1 = short
        self.entry_price = entry_price
        self.shares = shares
        self.entry_idx = entry_idx
        self.exit_price = None
        self.exit_idx = None
        self.pnl = 0.0
        self.pnl_pct = 0.0


class BacktestEngine:
    """
    回测引擎

    核心流程：
    1. 从 warmup_period 开始，每 rebalance_freq 步执行一次预测
    2. 预测器生成 point_forecast 和 quantile_forecast
    3. 策略根据预测生成信号
    4. 引擎执行交易，管理仓位
    5. 每步更新权益曲线
    """

    def __init__(self, strategy, predictor):
        self.strategy = strategy
        self.predictor = predictor
        self.tc = config.TRADING_CONFIG
        self.bc = config.BACKTEST_CONFIG

    def run(self, data, features):
        """
        执行回测

        Args:
            data: dict, OHLCV 原始数据
            features: dict, 技术指标

        Returns:
            dict: 回测结果
        """
        close = data["close"]
        n = len(close)
        warmup = self.bc["warmup_period"]
        rebalance = self.bc["rebalance_freq"]
        context_len = config.MODEL_CONFIG["context_length"]
        horizon = config.MODEL_CONFIG["horizon"]

        # 初始化
        capital = self.tc["initial_capital"]
        cash = capital
        position = None  # 当前持仓
        equity_curve = np.zeros(n)
        equity_curve[:warmup] = capital

        # 记录
        trades = []
        signals_log = []
        daily_returns = []

        print(f"  [BACKTEST] {self.strategy.name} | {data['symbol']} | {n} days, warmup={warmup}")

        for i in range(warmup, n):
            current_price = close[i]

            # 更新持仓市值
            if position is not None:
                if position.direction == 1:
                    unrealized_pnl = (current_price - position.entry_price) * position.shares
                else:
                    unrealized_pnl = (position.entry_price - current_price) * position.shares
                portfolio_value = cash + position.entry_price * position.shares + unrealized_pnl
            else:
                portfolio_value = cash

            equity_curve[i] = portfolio_value

            # 止损止盈检查
            if position is not None:
                pnl_pct = unrealized_pnl / (position.entry_price * position.shares)
                if pnl_pct <= -self.tc["stop_loss"]:
                    # 止损
                    cash, position = self._close_position(position, current_price, cash, i, "止损")
                    trades.append(position)
                    position = None
                elif pnl_pct >= self.tc["take_profit"]:
                    # 止盈
                    cash, position = self._close_position(position, current_price, cash, i, "止盈")
                    trades.append(position)
                    position = None

            # 定期重新预测 + 生成信号
            if (i - warmup) % rebalance == 0:
                # 提取历史数据
                start_idx = max(0, i - context_len)
                history = close[start_idx:i + 1]

                # 模型预测
                try:
                    point_forecast, quantile_forecast = self.predictor.predict(history, horizon)
                except Exception as e:
                    point_forecast, quantile_forecast = None, None

                # 技术指标快照
                feat_snapshot = {}
                for k, v in features.items():
                    if isinstance(v, np.ndarray) and i < len(v):
                        feat_snapshot[k] = v[i]

                # 策略信号
                signal, strength, reason = self.strategy.generate_signal(
                    current_price, point_forecast, quantile_forecast, feat_snapshot
                )

                signals_log.append({
                    "idx": i,
                    "price": current_price,
                    "signal": signal,
                    "strength": strength,
                    "reason": reason,
                })

                # 执行交易
                if signal == SIGNAL_BUY and position is None:
                    cash, position = self._open_position(1, current_price, cash, i, portfolio_value)
                elif signal == SIGNAL_SELL and position is None:
                    cash, position = self._open_position(-1, current_price, cash, i, portfolio_value)
                elif signal == SIGNAL_SELL and position is not None and position.direction == 1:
                    cash, position = self._close_position(position, current_price, cash, i, "反转信号")
                    trades.append(position)
                    position = None
                elif signal == SIGNAL_BUY and position is not None and position.direction == -1:
                    cash, position = self._close_position(position, current_price, cash, i, "反转信号")
                    trades.append(position)
                    position = None

            # 日收益率
            if i > warmup:
                daily_ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                daily_returns.append(daily_ret)

        # 收尾：平掉剩余仓位
        if position is not None:
            cash, position = self._close_position(position, close[-1], cash, n - 1, "回测结束")
            trades.append(position)
            equity_curve[-1] = cash

        # 计算绩效指标
        metrics = self._compute_metrics(equity_curve[warmup:], daily_returns, trades, capital)

        return {
            "symbol": data["symbol"],
            "strategy": self.strategy.name,
            "equity_curve": equity_curve,
            "trades": trades,
            "signals": signals_log,
            "metrics": metrics,
            "warmup": warmup,
        }

    def _open_position(self, direction, price, cash, idx, portfolio_value):
        """开仓"""
        # 计算仓位大小
        position_capital = portfolio_value * self.tc["position_size"]
        # 考虑滑点
        exec_price = price * (1 + self.tc["slippage"] * direction)
        # 计算可买股数
        shares = int(position_capital / exec_price)
        if shares <= 0:
            return cash, None
        # 手续费
        commission = shares * exec_price * self.tc["commission_rate"]
        cost = shares * exec_price + commission
        if cost > cash:
            shares = int((cash - commission) / exec_price)
            if shares <= 0:
                return cash, None
            cost = shares * exec_price + shares * exec_price * self.tc["commission_rate"]

        cash -= cost
        pos = Position(direction, exec_price, shares, idx)
        return cash, pos

    def _close_position(self, position, price, cash, idx, reason=""):
        """平仓"""
        exec_price = price * (1 - self.tc["slippage"] * position.direction)
        proceeds = position.shares * exec_price
        commission = proceeds * self.tc["commission_rate"]

        if position.direction == 1:
            pnl = (exec_price - position.entry_price) * position.shares - commission
        else:
            pnl = (position.entry_price - exec_price) * position.shares - commission

        position.exit_price = exec_price
        position.exit_idx = idx
        position.pnl = pnl
        position.pnl_pct = pnl / (position.entry_price * position.shares)

        cash += proceeds - commission
        return cash, position

    def _compute_metrics(self, equity_curve, daily_returns, trades, initial_capital):
        """计算完整绩效指标"""
        daily_returns = np.array(daily_returns) if daily_returns else np.array([0.0])
        final_value = equity_curve[-1] if len(equity_curve) > 0 else initial_capital
        n_days = len(equity_curve)

        # 总收益率
        total_return = (final_value - initial_capital) / initial_capital

        # 年化收益率
        ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        # 最大回撤
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Sharpe Ratio (年化, 无风险利率 3%)
        rf_daily = 0.03 / 252
        excess_returns = daily_returns - rf_daily
        sharpe = np.sqrt(252) * np.mean(excess_returns) / max(np.std(daily_returns), 1e-10)

        # Sortino Ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
        sortino = np.sqrt(252) * np.mean(excess_returns) / max(downside_std, 1e-10)

        # Calmar Ratio
        calmar = ann_return / max(max_drawdown, 1e-10)

        # 交易统计
        n_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        win_rate = len(winning_trades) / max(n_trades, 1)

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 1
        profit_factor = avg_win / max(avg_loss, 1e-10)

        # 平均持仓天数
        avg_hold = np.mean([t.exit_idx - t.entry_idx for t in trades]) if trades else 0

        return {
            "total_return": total_return,
            "ann_return": ann_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "total_trades": n_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_holding_days": avg_hold,
            "final_value": final_value,
            "total_pnl": sum(t.pnl for t in trades),
        }


def run_backtest(data_dict, strategies, predictor):
    """
    对所有股票 × 所有策略执行回测

    Args:
        data_dict: {symbol: {"raw": ..., "features": ...}}
        strategies: list of strategy instances
        predictor: predictor instance

    Returns:
        list[dict]: 所有回测结果
    """
    all_results = []
    for symbol, sdata in data_dict.items():
        for strategy in strategies:
            engine = BacktestEngine(strategy, predictor)
            result = engine.run(sdata["raw"], sdata["features"])
            all_results.append(result)
            m = result["metrics"]
            print(f"    => Return: {m['total_return']:.2%} | Sharpe: {m['sharpe_ratio']:.2f} | "
                  f"MaxDD: {m['max_drawdown']:.2%} | Trades: {m['total_trades']} | WinRate: {m['win_rate']:.0%}")
    return all_results
