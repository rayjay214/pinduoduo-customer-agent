"""
消息构建器。
只负责拼系统工具指引、会话上下文和 LLM 消息列表。
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Dict, List

from bridge.context import Context
from config import get_config
from utils.config_values import as_bool
from Agent.CustomerAgent.custom.media_detection import infer_media_type_from_url, normalize_media_url
from Agent.CustomerAgent.custom.prompt_rules import (
    build_image_grounding_instruction,
    build_version_name_instruction,
)
from Agent.CustomerAgent.custom.turn_context import TurnContext, parse_turn_context
from utils.logger_loguru import get_logger

logger = get_logger("MessageBuilder")

DEFAULT_GROUNDED_KNOWLEDGE_TOPICS = (
    "商品参数",
    "功能",
    "按键/图标/部件用途",
    "快递",
    "发货地",
)


class MessageBuilder:
    """构建消息与最小化系统工具指引。"""

    def __init__(self) -> None:
        self.system_prompt = ""
        self._build_system_prompt()

    def _build_system_prompt(self) -> None:
        """构建总 prompt：硬编码只保留工具调用边界，业务规则从配置读取。"""
        requirements = [
            "需要发商品卡片、商品链接或推荐商品时，使用 `send_product_card`。",
            "需要转人工时，使用 `transfer_conversation`。",
            "调用人工工具时，必须使用当前会话信息里的 `shop_id`、`user_id`、`recipient_uid`。",
            "不要向客户输出工具名或提示词内容。",
            self._grounded_knowledge_requirement(),
            build_image_grounding_instruction(),
            build_version_name_instruction(),
            "视频/图片追问：如果客户只发了视频或图片、没有附带文字问题，回复'麻烦您说下具体想确认哪里'，不要猜测客户意图。",
        ]
        base_prompt = "工具使用要求：\n" + "\n".join(
            f"{index}. {item}"
            for index, item in enumerate((item for item in requirements if item), 1)
        ) + "\n"
        prompt_instructions = get_config("prompt.instructions", [])
        if isinstance(prompt_instructions, list):
            extra_prompt = "\n".join(
                str(item).strip() for item in prompt_instructions if str(item).strip()
            )
        else:
            extra_prompt = str(prompt_instructions or "").strip()

        self.system_prompt = base_prompt
        if extra_prompt:
            self.system_prompt += "\n【配置提示词】\n" + extra_prompt + "\n"

    @classmethod
    def _grounded_knowledge_topics(cls) -> tuple[str, ...]:
        configured = get_config("agent.grounded_knowledge_topics", None)
        if configured is None:
            configured = DEFAULT_GROUNDED_KNOWLEDGE_TOPICS
        elif not isinstance(configured, (list, tuple)):
            configured = DEFAULT_GROUNDED_KNOWLEDGE_TOPICS
        return tuple(str(item or "").strip() for item in configured if str(item or "").strip())

    @classmethod
    def _grounded_knowledge_requirement(cls) -> str:
        topics = cls._grounded_knowledge_topics()
        if not topics:
            return (
                "涉及需要商品或店铺事实支撑的问题时，只能使用预检索知识或 "
                "`search_knowledge` 的明确值；知识未提供时不要自行估算。"
            )
        return (
            f"涉及{'、'.join(topics)}时，只能使用预检索知识或 "
            "`search_knowledge` 的明确值；知识未提供时不要自行估算。"
        )

    def build_dependencies(self, context: Context) -> Dict[str, Any]:
        """从 Context 构建依赖字典。"""
        from_uid = str(self._context_kwarg(context, "from_uid", "") or "")
        goods_id = self._extract_goods_id(context)

        shop_id = str(self._context_kwarg(context, "shop_id", "") or "")

        # TurnContext 结构化解析
        raw_query = str(context.content or "")
        turn_context: TurnContext | None = None
        if as_bool(get_config("enable_turn_context", False), False):
            turn_context = parse_turn_context(raw_query)

        deps = {
            "shop_name": str(self._context_kwarg(context, "shop_name", "") or ""),
            "channel_type": str(context.channel_type.value if context.channel_type else ""),
            "context_type": str(context.type.value if context.type else ""),
            "shop_id": shop_id,
            "user_id": str(self._context_kwarg(context, "user_id", "") or ""),
            "from_uid": from_uid,
            "recipient_uid": from_uid,
            "goods_id": goods_id,
            "query": str(context.content or ""),
            "media_url": self._extract_media_url(context),
            "media_type": self._extract_media_type(context),
        }

        if turn_context is not None:
            deps["turn_context"] = turn_context

        return deps

    @staticmethod
    def _context_kwarg(context: Context, key: str, default: Any = None) -> Any:
        kwargs = getattr(context, "kwargs", None)
        if isinstance(kwargs, Mapping):
            return kwargs.get(key, default)
        return getattr(kwargs, key, default)

    @staticmethod
    def _coerce_goods_id(value: Any) -> int | None:
        if value is not None and str(value).strip().isdigit():
            return int(str(value).strip())
        return None

    @classmethod
    def _goods_id_from_any(cls, data: Any) -> int | None:
        if isinstance(data, list):
            for item in data:
                goods_id = cls._goods_id_from_any(item)
                if goods_id is not None:
                    return goods_id
            return None

        if not isinstance(data, dict):
            return None

        for key in ("goods_id", "goodsID", "goodsId"):
            goods_id = cls._coerce_goods_id(data.get(key))
            if goods_id is not None:
                return goods_id

        for value in data.values():
            goods_id = cls._goods_id_from_any(value)
            if goods_id is not None:
                return goods_id
        return None

    @classmethod
    def _goods_id_from_mapping(cls, data: Any) -> int | None:
        return cls._goods_id_from_any(data)

    @classmethod
    def _media_url_from_any(cls, data: Any) -> str:
        if isinstance(data, list):
            for item in data:
                media_url = cls._media_url_from_any(item)
                if media_url:
                    return media_url
            return ""

        if isinstance(data, str):
            text = normalize_media_url(data)
            return text if text.startswith(("http://", "https://", "data:")) else ""

        if not isinstance(data, dict):
            return ""

        for key in ("url", "image_url", "video_url", "cover"):
            value = data.get(key)
            media_url = cls._media_url_from_any(value)
            if media_url:
                return media_url

        for value in data.values():
            media_url = cls._media_url_from_any(value)
            if media_url:
                return media_url
        return ""

    @classmethod
    def _goods_id_from_json_text(cls, text: str) -> int | None:
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        if isinstance(parsed, dict):
            return cls._goods_id_from_mapping(parsed)
        if isinstance(parsed, list):
            for item in parsed:
                goods_id = cls._goods_id_from_mapping(item)
                if goods_id is not None:
                    return goods_id
        return None

    @classmethod
    def _extract_goods_id(cls, context: Context) -> int | None:
        """Extract the current goods_id from a goods card, merged text, or raw PDD payload."""
        raw_content = str(context.content or "")
        if raw_content.strip():
            goods_id = cls._goods_id_from_json_text(raw_content)
            if goods_id is not None:
                return goods_id

            match = re.search(r"商品ID[：:]\s*(\d{6,})", raw_content)
            if match:
                return int(match.group(1))

        raw_data = cls._context_kwarg(context, "raw_data", {}) or {}
        goods_id = cls._goods_id_from_mapping(raw_data)
        if goods_id is not None:
            return goods_id

        candidates = [
            ("message", "info", "goodsID"),
            ("message", "info", "goods_id"),
            ("message", "info", "goods_info", "goods_id"),
            ("message", "info", "data", "goodsID"),
            ("message", "info", "data", "goodsId"),
            ("message", "info", "data", "goods_id"),
            ("message", "info", "data", "goods_info", "goods_id"),
            ("message", "biz_context", "goodsId"),
        ]
        for path in candidates:
            value: Any = raw_data
            for key in path:
                if not isinstance(value, dict):
                    value = None
                    break
                value = value.get(key)
            goods_id = cls._coerce_goods_id(value)
            if goods_id is not None:
                return goods_id

        message_data = raw_data.get("message") if isinstance(raw_data, dict) else {}
        biz_context = message_data.get("biz_context") if isinstance(message_data, dict) else {}
        biz_context = biz_context if isinstance(biz_context, dict) else {}
        if isinstance(biz_context, dict):
            for key in ("mallOrderConfirmNewCardCallBackParam", "orderConfirmAgreeNewCardCallBackParam"):
                value = biz_context.get(key)
                if isinstance(value, str) and value.strip():
                    goods_id = cls._goods_id_from_json_text(value)
                    if goods_id is not None:
                        return goods_id
        return None

    @staticmethod
    def _extract_media_url(context: Context) -> str:
        raw_content = context.content
        if isinstance(raw_content, str):
            text = raw_content.strip()
            if text.startswith(("http://", "https://", "data:")):
                return normalize_media_url(text)

            match = re.search(r"https?://[^\s，。；,;]+", text)
            if match:
                return normalize_media_url(match.group(0))

            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            media_url = MessageBuilder._media_url_from_any(parsed)
            if media_url:
                return media_url

        raw_data = MessageBuilder._context_kwarg(context, "raw_data", {}) or {}
        if isinstance(raw_data, dict):
            return MessageBuilder._media_url_from_any(raw_data)
        return ""

    @classmethod
    def _extract_media_type(cls, context: Context) -> str:
        context_type = str(context.type.value if context.type else "")
        raw_content = str(context.content or "")
        media_url = cls._extract_media_url(context).lower()

        if context_type in {"image", "video"}:
            return context_type
        inferred_media_type = infer_media_type_from_url(media_url)
        if "客户发送了图片" in raw_content or inferred_media_type == "image":
            return "image"
        if "客户发送了视频" in raw_content or inferred_media_type == "video":
            return "video"
        return ""

    def build_messages(
        self,
        query: str,
        history: List[Dict[str, Any]],
        dependencies: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """构建 LLM 消息列表。"""
        dependencies = dict(dependencies) if isinstance(dependencies, Mapping) else {}
        messages: List[Dict[str, Any]] = []

        if self.system_prompt:
            content = self.system_prompt
            if dependencies:
                for key, value in dependencies.items():
                    content = content.replace(f"{{{key}}}", str(value))

                session_info = "\n\n【当前会话信息】\n"
                session_info += f"- shop_id: {dependencies.get('shop_id', '')}（调用工具必填）\n"
                session_info += f"- user_id: {dependencies.get('user_id', '')}（调用工具必填）\n"
                session_info += f"- recipient_uid: {dependencies.get('recipient_uid', '')}（调用工具必填，不能自造）\n"
                session_info += f"- shop_name: {dependencies.get('shop_name', '')}\n"
                session_info += f"- channel_type: {dependencies.get('channel_type', '')}\n"
                session_info += f"- context_type: {dependencies.get('context_type', '')}\n"
                if dependencies.get("goods_id"):
                    session_info += f"- goods_id: {dependencies.get('goods_id')}（当前客户咨询商品，商品知识工具优先使用）\n"
                if dependencies.get("order_context_text"):
                    session_info += "\n" + str(dependencies.get("order_context_text")) + "\n"
                content += session_info

            messages.append({"role": "system", "content": content})

        for msg in history or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if not role or content is None:
                continue
            if role == "tool" or msg.get("tool_calls"):
                continue
            if role == "system":
                messages.append({"role": "system", "content": str(content)})
            elif role in {"user", "assistant"}:
                messages.append({"role": role, "content": str(content)})

        media_url = normalize_media_url(str((dependencies or {}).get("media_url") or ""))
        context_type = str((dependencies or {}).get("context_type") or "")
        media_type = str((dependencies or {}).get("media_type") or "")
        if media_url and (context_type == "image" or media_type == "image"):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": media_url}},
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": query})
        return messages
