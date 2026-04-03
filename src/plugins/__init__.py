"""
插件包
======
提供插件注册、发现和调度功能。

内置插件:
  - ConsolePlugin   : 控制台彩色输出
  - FeishuPlugin    : 飞书 Webhook / SDK 通知

自定义插件只需继承 BasePlugin 并实现 name + handle 即可。
"""
from .base import BasePlugin, SignalEvent, SIGNAL_LABEL
from .console_plugin import ConsolePlugin
from .feishu_plugin import FeishuPlugin

__all__ = [
    "BasePlugin",
    "SignalEvent",
    "SIGNAL_LABEL",
    "ConsolePlugin",
    "FeishuPlugin",
    "create_plugins_from_config",
]


def create_plugins_from_config(plugin_config: dict) -> list[BasePlugin]:
    """
    根据配置字典自动创建启用的插件实例。

    plugin_config 结构见 config.PLUGIN_CONFIG
    """
    plugins: list[BasePlugin] = []

    if plugin_config.get("console", {}).get("enabled", True):
        plugins.append(ConsolePlugin(
            verbose=plugin_config.get("console", {}).get("verbose", False),
        ))

    feishu_cfg = plugin_config.get("feishu", {})
    if feishu_cfg.get("enabled", False):
        plugins.append(FeishuPlugin(
            webhook_url=feishu_cfg.get("webhook_url", ""),
            secret=feishu_cfg.get("secret", ""),
            app_id=feishu_cfg.get("app_id", ""),
            app_secret=feishu_cfg.get("app_secret", ""),
            receive_id=feishu_cfg.get("receive_id", ""),
            receive_id_type=feishu_cfg.get("receive_id_type", ""),
            use_card=feishu_cfg.get("use_card", True),
        ))

    return plugins
