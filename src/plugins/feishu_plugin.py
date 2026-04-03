"""
飞书通知插件
============
支持两种发送模式，按优先级自动选择：

1. **Webhook 模式**（推荐快速上手）
   - 在飞书群 → 群机器人 → 添加自定义机器人 → 获取 webhook_url
   - 可选：开启签名校验 → 填入 secret
   - 配置 webhook_url 即可使用

2. **SDK 模式**（功能更强，需安装 lark-oapi）
   - pip install lark-oapi
   - 在飞书开放平台创建应用 → 获取 app_id / app_secret
   - receive_id_type 支持:
     · "email"   → 直接用邮箱给个人发消息（最简单）
     · "open_id" → 用户的 open_id
     · "user_id" → 用户的 user_id
     · "chat_id" → 群聊 ID

消息格式：
  - use_card=True  → 发送交互式卡片消息（美观，推荐）
  - use_card=False → 发送纯文本消息
"""
from __future__ import annotations

import hashlib
import hmac
import base64
import json
import time
import urllib.request
import urllib.error
from typing import Any

from .base import BasePlugin, SignalEvent


_SIGNAL_EMOJI = {1: "📈", -1: "📉", 0: "⏸️"}
_SIGNAL_COLOR = {1: "green", -1: "red", 0: "grey"}
_SIGNAL_CN = {1: "买入", -1: "卖出", 0: "观望"}


class FeishuPlugin(BasePlugin):

    def __init__(
        self,
        webhook_url: str = "",
        secret: str = "",
        app_id: str = "",
        app_secret: str = "",
        receive_id: str = "",
        receive_id_type: str = "",
        use_card: bool = True,
    ):
        self._webhook_url = webhook_url
        self._secret = secret
        self._app_id = app_id
        self._app_secret = app_secret
        self._receive_id = receive_id
        self._receive_id_type = receive_id_type
        self._use_card = use_card
        self._sdk_client = None

    @property
    def name(self) -> str:
        return "Feishu"

    # ── lifecycle ─────────────────────────────────────────────

    def startup(self) -> None:
        if self._app_id and self._app_secret:
            self._init_sdk_client()

    def shutdown(self) -> None:
        self._sdk_client = None

    # ── main entry ────────────────────────────────────────────

    def handle(self, event: SignalEvent) -> None:
        if self._use_card:
            payload = self._build_card_message(event)
        else:
            payload = self._build_text_message(event)

        if self._webhook_url:
            self._send_webhook(payload)
        elif self._sdk_client and self._receive_id:
            self._send_sdk(payload)
        else:
            print(f"  [Feishu] ⚠ No webhook_url or SDK receive_id configured, skipping")

    # ── message builders ──────────────────────────────────────

    def _build_text_message(self, event: SignalEvent) -> dict[str, Any]:
        emoji = _SIGNAL_EMOJI.get(event.signal, "❓")
        cn = _SIGNAL_CN.get(event.signal, "未知")

        lines = [
            f"{emoji} 【交易信号】{cn}",
            f"━━━━━━━━━━━━━━━━━━",
            f"标的: {event.symbol}",
            f"策略: {event.strategy}",
            f"方向: {event.signal_label} ({cn})",
            f"强度: {event.strength:.0%}",
            f"现价: {event.current_price:.2f}",
            f"原因: {event.reason}",
            f"时间: {event.timestamp:%Y-%m-%d %H:%M}",
        ]

        if event.point_forecast:
            fc = ", ".join(f"{v:.2f}" for v in event.point_forecast[:5])
            lines.append(f"预测: [{fc}]")

        key_feats = {k: v for k, v in event.features.items()
                     if k in ("rsi", "adx", "volume_ratio", "macro_score", "volatility")}
        if key_feats:
            feats_str = " | ".join(f"{k}={v:.2f}" for k, v in key_feats.items())
            lines.append(f"指标: {feats_str}")

        return {
            "msg_type": "text",
            "content": {"text": "\n".join(lines)},
        }

    def _build_card_message(self, event: SignalEvent) -> dict[str, Any]:
        emoji = _SIGNAL_EMOJI.get(event.signal, "❓")
        cn = _SIGNAL_CN.get(event.signal, "未知")
        color = _SIGNAL_COLOR.get(event.signal, "grey")

        header_template = {"blue": "blue", "green": "green", "red": "red", "grey": "grey"}
        template = header_template.get(color, "blue")

        elements: list[dict] = []

        elements.append({
            "tag": "div",
            "fields": [
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**标的**\n{event.symbol}"}},
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**策略**\n{event.strategy}"}},
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**方向**\n{emoji} {cn}"}},
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**强度**\n{event.strength:.0%}"}},
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**现价**\n¥{event.current_price:.2f}"}},
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**时间**\n{event.timestamp:%m-%d %H:%M}"}},
            ],
        })

        elements.append({"tag": "hr"})

        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"**信号原因**\n{event.reason}"},
        })

        if event.point_forecast:
            fc_items = " → ".join(f"{v:.2f}" for v in event.point_forecast[:5])
            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**预测路径**\n{fc_items}"},
            })

        key_feats = {k: v for k, v in event.features.items()
                     if k in ("rsi", "adx", "volume_ratio", "macro_score", "volatility")}
        if key_feats:
            feat_parts = [f"`{k}={v:.2f}`" for k, v in key_feats.items()]
            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**关键指标**\n{' | '.join(feat_parts)}"},
            })

        elements.append({"tag": "hr"})
        elements.append({
            "tag": "note",
            "elements": [{"tag": "plain_text", "content": "TimesFM Quant Signal System · 仅供参考，不构成投资建议"}],
        })

        card = {
            "header": {
                "title": {"tag": "plain_text", "content": f"{emoji} 交易信号 — {event.symbol} {cn}"},
                "template": template,
            },
            "elements": elements,
        }

        return {"msg_type": "interactive", "card": card}

    # ── webhook sender ────────────────────────────────────────

    def _gen_sign(self, timestamp: str) -> str:
        string_to_sign = f"{timestamp}\n{self._secret}"
        hmac_code = hmac.new(
            string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
        ).digest()
        return base64.b64encode(hmac_code).decode("utf-8")

    def _send_webhook(self, payload: dict) -> None:
        if self._secret:
            ts = str(int(time.time()))
            payload["timestamp"] = ts
            payload["sign"] = self._gen_sign(ts)

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._webhook_url,
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if result.get("code", 0) != 0:
                    print(f"  [Feishu] Webhook error: {result.get('msg', 'unknown')}")
                else:
                    print(f"  [Feishu] ✓ Webhook sent")
        except urllib.error.URLError as e:
            print(f"  [Feishu] Webhook failed: {e}")

    # ── SDK sender ────────────────────────────────────────────

    def _init_sdk_client(self) -> None:
        try:
            import lark_oapi as lark
            self._sdk_client = (
                lark.Client.builder()
                .app_id(self._app_id)
                .app_secret(self._app_secret)
                .build()
            )
            print(f"  [Feishu] SDK client initialized (receive → {self._receive_id_type}:{self._receive_id})")
        except ImportError:
            print(f"  [Feishu] lark-oapi not installed, falling back to webhook")
            self._sdk_client = None

    def _send_sdk(self, payload: dict) -> None:
        if not self._sdk_client:
            return

        try:
            from lark_oapi.api.im.v1 import (
                CreateMessageRequest,
                CreateMessageRequestBody,
            )

            if payload["msg_type"] == "interactive":
                content_str = json.dumps(payload["card"], ensure_ascii=False)
            else:
                content_str = json.dumps(payload["content"], ensure_ascii=False)

            body = (
                CreateMessageRequestBody.builder()
                .receive_id(self._receive_id)
                .msg_type(payload["msg_type"])
                .content(content_str)
                .build()
            )

            request = (
                CreateMessageRequest.builder()
                .receive_id_type(self._receive_id_type)
                .request_body(body)
                .build()
            )

            response = self._sdk_client.im.v1.message.create(request)
            if response.success():
                print(f"  [Feishu] ✓ SDK message sent → {self._receive_id_type}:{self._receive_id}")
            else:
                print(f"  [Feishu] SDK error: code={response.code}, msg={response.msg}")

        except Exception as e:
            print(f"  [Feishu] SDK send failed: {e}")
