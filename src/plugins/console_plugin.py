"""
控制台插件 —— 将信号以可读格式打印到终端
"""
from .base import BasePlugin, SignalEvent

_COLOR = {
    1:  "\033[92m",   # green  → BUY
    -1: "\033[91m",   # red    → SELL
    0:  "\033[93m",   # yellow → HOLD
}
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_ARROW = {1: "▲", -1: "▼", 0: "─"}


class ConsolePlugin(BasePlugin):

    def __init__(self, verbose: bool = False):
        self._verbose = verbose

    @property
    def name(self) -> str:
        return "Console"

    def handle(self, event: SignalEvent) -> None:
        self._print_single(event)

    def handle_batch(self, symbol: str, events: list[SignalEvent]) -> None:
        ref = events[0]
        print(f"\n  {_BOLD}┌── {symbol}  ¥{ref.current_price:.2f}  "
              f"({ref.timestamp:%Y-%m-%d %H:%M}){_RESET}")

        if self._verbose and ref.point_forecast:
            fc = ", ".join(f"{v:.2f}" for v in ref.point_forecast)
            print(f"  │  {_DIM}forecast=[{fc}]{_RESET}")

        if self._verbose and ref.features:
            feats = ", ".join(f"{k}={v:.3f}" for k, v in ref.features.items()
                             if k in ("rsi", "adx", "volume_ratio", "macro_score"))
            if feats:
                print(f"  │  {_DIM}features: {feats}{_RESET}")

        for i, event in enumerate(events):
            is_last = (i == len(events) - 1)
            prefix = "└" if is_last else "├"
            self._print_strategy_line(event, prefix)

        print()

    def _print_single(self, event: SignalEvent) -> None:
        c = _COLOR.get(event.signal, "")
        arrow = _ARROW.get(event.signal, "?")
        label = event.signal_label

        print(
            f"  {c}{arrow} [{label}]{_RESET}  "
            f"{event.symbol} | {event.strategy} | "
            f"price={event.current_price:.2f} | "
            f"strength={event.strength:.0%} | {event.reason}"
        )

    def _print_strategy_line(self, event: SignalEvent, prefix: str) -> None:
        c = _COLOR.get(event.signal, "")
        arrow = _ARROW.get(event.signal, "?")
        label = event.signal_label

        print(
            f"  {prefix}─ {c}{arrow} [{label}]{_RESET}  "
            f"{event.strategy} | "
            f"strength={event.strength:.0%} | {event.reason}"
        )
