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
  - use_card=True  → 发送交互式卡片消息（按标的聚合，推荐）
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

_KEY_FEATURES = ("rsi", "adx", "volume_ratio", "macro_score", "volatility")


def _pick_header_color(events: list[SignalEvent]) -> str:
    """根据一组信号的综合方向决定卡片头颜色。"""
    has_buy = any(e.signal == 1 for e in events)
    has_sell = any(e.signal == -1 for e in events)
    if has_buy and not has_sell:
        return "green"
    if has_sell and not has_buy:
        return "red"
    if has_buy and has_sell:
        return "orange"
    return "grey"


def _pick_header_summary(events: list[SignalEvent]) -> str:
    """生成卡片标题中的方向摘要文字。"""
    buy = sum(1 for e in events if e.signal == 1)
    sell = sum(1 for e in events if e.signal == -1)
    hold = sum(1 for e in events if e.signal == 0)
    parts = []
    if buy:
        parts.append(f"{buy}买入")
    if sell:
        parts.append(f"{sell}卖出")
    if hold:
        parts.append(f"{hold}观望")
    return " / ".join(parts)


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
        self.handle_batch(event.symbol, [event])

    def handle_batch(self, symbol: str, events: list[SignalEvent]) -> None:
        if self._use_card:
            payload = self._build_batch_card(symbol, events)
        else:
            payload = self._build_batch_text(symbol, events)

        self._send(payload)

    # ── senders ───────────────────────────────────────────────

    def _send(self, payload: dict) -> None:
        if self._webhook_url:
            self._send_webhook(payload)
        elif self._sdk_client and self._receive_id:
            self._send_sdk(payload)
        else:
            print(f"  [Feishu] ⚠ No webhook_url or SDK receive_id configured, skipping")

    # ── batch message builders (per symbol) ───────────────────

    def _build_batch_text(self, symbol: str, events: list[SignalEvent]) -> dict[str, Any]:
        """纯文本模式：一个标的一条消息，内含所有策略结果。"""
        ref = events[0]
        summary = _pick_header_summary(events)

        lines = [
            f"📊 【信号汇总】{symbol}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━",
            f"现价: ¥{ref.current_price:.2f}  |  时间: {ref.timestamp:%Y-%m-%d %H:%M}",
            f"方向汇总: {summary}",
            "",
        ]

        for i, event in enumerate(events, 1):
            emoji = _SIGNAL_EMOJI.get(event.signal, "❓")
            cn = _SIGNAL_CN.get(event.signal, "未知")
            lines.append(f"{'─' * 20}")
            lines.append(f"{emoji} 策略 {i}: {event.strategy}")
            lines.append(f"  方向: {event.signal_label} ({cn})  |  强度: {event.strength:.0%}")
            lines.append(f"  原因: {event.reason}")

        if ref.point_forecast:
            fc = " → ".join(f"{v:.2f}" for v in ref.point_forecast[:5])
            lines.extend(["", f"预测路径: {fc}"])

        key_feats = {k: v for k, v in ref.features.items() if k in _KEY_FEATURES}
        if key_feats:
            feats_str = " | ".join(f"{k}={v:.2f}" for k, v in key_feats.items())
            lines.append(f"关键指标: {feats_str}")

        return {
            "msg_type": "text",
            "content": {"text": "\n".join(lines)},
        }

    def _build_batch_card(self, symbol: str, events: list[SignalEvent]) -> dict[str, Any]:
        """卡片模式：一个标的一张卡片，每个策略一个独立区块。"""
        ref = events[0]
        color = _pick_header_color(events)
        summary = _pick_header_summary(events)

        elements: list[dict] = []

        # ── 标的概览 ──
        elements.append({
            "tag": "div",
            "fields": [
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**标的**\n{symbol}"}},
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**现价**\n¥{ref.current_price:.2f}"}},
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**时间**\n{ref.timestamp:%m-%d %H:%M}"}},
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**方向汇总**\n{summary}"}},
            ],
        })

        # ── 预测路径（共享同一个模型输出） ──
        if ref.point_forecast:
            fc_items = " → ".join(f"{v:.2f}" for v in ref.point_forecast[:5])
            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**预测路径**  {fc_items}"},
            })

        # ── 关键指标（同一标的共享） ──
        key_feats = {k: v for k, v in ref.features.items() if k in _KEY_FEATURES}
        if key_feats:
            feat_parts = [f"`{k}={v:.2f}`" for k, v in key_feats.items()]
            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**关键指标**  {' | '.join(feat_parts)}"},
            })

        elements.append({"tag": "hr"})

        # ── 每个策略一个区块 ──
        for event in events:
            emoji = _SIGNAL_EMOJI.get(event.signal, "❓")
            cn = _SIGNAL_CN.get(event.signal, "未知")

            elements.append({
                "tag": "div",
                "fields": [
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**{event.strategy}**\n{emoji} {cn}"}},
                    {"is_short": True, "text": {"tag": "lark_md", "content": f"**强度**\n{event.strength:.0%}"}},
                ],
            })
            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"{event.reason}"},
            })
            elements.append({"tag": "hr"})

        # ── 底部 disclaimer ──
        elements.append({
            "tag": "note",
            "elements": [{"tag": "plain_text", "content": "TimesFM Quant Signal System · 仅供参考，不构成投资建议"}],
        })

        card = {
            "header": {
                "title": {"tag": "plain_text", "content": f"📊 信号汇总 — {symbol} ({summary})"},
                "template": color,
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
