"""
信号事件模型与插件基类
====================
定义统一的信号数据结构和插件接口，所有下游处理器（通知、交易、记录等）
均实现 BasePlugin 接口，通过 SignalDispatcher 接收信号。
"""
from __future__ import annotations

import abc
import datetime as dt
from dataclasses import dataclass, field
from typing import Any


SIGNAL_LABEL = {1: "BUY", -1: "SELL", 0: "HOLD"}


@dataclass
class SignalEvent:
    """一次策略信号的完整快照"""

    timestamp: dt.datetime
    symbol: str
    strategy: str

    signal: int                     # 1=BUY, -1=SELL, 0=HOLD
    strength: float                 # 0.0 ~ 1.0
    reason: str

    current_price: float
    point_forecast: list[float] | None = None
    quantile_summary: dict[str, float] | None = None
    features: dict[str, float] = field(default_factory=dict)

    @property
    def signal_label(self) -> str:
        return SIGNAL_LABEL.get(self.signal, "UNKNOWN")

    @property
    def is_actionable(self) -> bool:
        return self.signal != 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "strategy": self.strategy,
            "signal": self.signal_label,
            "strength": round(self.strength, 4),
            "reason": self.reason,
            "current_price": round(self.current_price, 4),
            "point_forecast": (
                [round(v, 4) for v in self.point_forecast]
                if self.point_forecast else None
            ),
            "quantile_summary": self.quantile_summary,
            "features": {k: round(v, 4) for k, v in self.features.items()},
        }


class BasePlugin(abc.ABC):
    """
    插件基类 —— 所有信号处理器必须继承此类。

    子类需要实现:
      - name (property): 插件名称
      - handle(event): 处理一个信号事件

    可选覆写:
      - on_error(event, error): 处理异常时的回调
      - startup() / shutdown(): 生命周期钩子
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def handle(self, event: SignalEvent) -> None:
        ...

    def on_error(self, event: SignalEvent, error: Exception) -> None:
        print(f"  [PLUGIN:{self.name}] Error: {error}")

    def startup(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"<Plugin:{self.name}>"
