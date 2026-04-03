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
_ARROW = {1: "▲", -1: "▼", 0: "─"}


class ConsolePlugin(BasePlugin):

    def __init__(self, verbose: bool = False):
        self._verbose = verbose

    @property
    def name(self) -> str:
        return "Console"

    def handle(self, event: SignalEvent) -> None:
        c = _COLOR.get(event.signal, "")
        arrow = _ARROW.get(event.signal, "?")
        label = event.signal_label

        print(
            f"  {c}{arrow} [{label}]{_RESET}  "
            f"{event.symbol} | {event.strategy} | "
            f"price={event.current_price:.2f} | "
            f"strength={event.strength:.0%} | {event.reason}"
        )

        if self._verbose and event.point_forecast:
            fc = ", ".join(f"{v:.2f}" for v in event.point_forecast)
            print(f"       forecast=[{fc}]")

        if self._verbose and event.features:
            feats = ", ".join(f"{k}={v:.3f}" for k, v in event.features.items()
                             if k in ("rsi", "adx", "volume_ratio", "macro_score"))
            if feats:
                print(f"       features: {feats}")
