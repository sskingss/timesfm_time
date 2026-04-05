"""
信号调度中心
============
接收策略产生的 SignalEvent，分发到所有已注册的插件。
插件异常不影响其他插件和主流程。
"""
from __future__ import annotations

from collections import OrderedDict

from .plugins.base import BasePlugin, SignalEvent


class SignalDispatcher:
    """
    中央调度器，负责：
    1. 管理插件生命周期 (startup / shutdown)
    2. 将信号事件分发给所有插件
    3. 隔离单个插件的异常
    """

    def __init__(self):
        self._plugins: list[BasePlugin] = []

    def register(self, plugin: BasePlugin) -> "SignalDispatcher":
        self._plugins.append(plugin)
        return self

    def register_many(self, plugins: list[BasePlugin]) -> "SignalDispatcher":
        self._plugins.extend(plugins)
        return self

    def startup(self) -> None:
        for p in self._plugins:
            try:
                p.startup()
            except Exception as e:
                print(f"  [Dispatcher] Plugin '{p.name}' startup failed: {e}")

    def shutdown(self) -> None:
        for p in self._plugins:
            try:
                p.shutdown()
            except Exception as e:
                print(f"  [Dispatcher] Plugin '{p.name}' shutdown failed: {e}")

    def dispatch(self, event: SignalEvent) -> None:
        for p in self._plugins:
            try:
                p.handle(event)
            except Exception as e:
                try:
                    p.on_error(event, e)
                except Exception:
                    print(f"  [Dispatcher] Plugin '{p.name}' on_error also failed")

    def dispatch_batch(self, events: list[SignalEvent]) -> None:
        """按标的分组后调用插件的 handle_batch，一个标的一条消息。"""
        grouped: OrderedDict[str, list[SignalEvent]] = OrderedDict()
        for e in events:
            grouped.setdefault(e.symbol, []).append(e)

        for symbol, symbol_events in grouped.items():
            for p in self._plugins:
                try:
                    p.handle_batch(symbol, symbol_events)
                except Exception as e:
                    try:
                        p.on_error(symbol_events[0], e)
                    except Exception:
                        print(f"  [Dispatcher] Plugin '{p.name}' on_error also failed")

    @property
    def plugin_names(self) -> list[str]:
        return [p.name for p in self._plugins]
