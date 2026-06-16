# 消息处理模块
import asyncio
import json
import time
from collections.abc import Mapping

from websockets import exceptions as ws_exceptions

from bridge.context import ChannelType, Context, ContextType
from Channel.pinduoduo.pdd_message import PDDChatMessage
from config import get_config
from database import db_manager
from utils.config_values import as_bool
from utils.logger_loguru import get_logger
from core.base_service import _sanitize_for_log as _core_sanitize_for_log


class MessageHandlerMixin:
    """消息处理 Mixin"""

    QUEUE_DEBOUNCE_SECONDS = 1.0
    RECENT_TEXT_TTL_SECONDS = 90.0
    WEBSOCKET_ERROR_WARNING_THRESHOLD = 3
    CARD_CONTEXT_TYPES = {
        ContextType.GOODS_INQUIRY,
        ContextType.GOODS_SPEC,
        ContextType.ORDER_INFO,
        ContextType.GOODS_CARD,
    }

    @staticmethod
    def _safe_json_dumps(data):
        """Safely serialize websocket data for trace logs."""
        try:
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(data)

    @staticmethod
    def _sanitize_for_log(value):
        return _core_sanitize_for_log(value)

    @staticmethod
    def _context_kwarg(context: Context, key: str, default=None):
        kwargs = getattr(context, "kwargs", None)
        if isinstance(kwargs, Mapping):
            return kwargs.get(key, default)
        return getattr(kwargs, key, default)

    @staticmethod
    def _extract_ws_meta(message_data):
        """Build a compact summary for websocket packets."""
        if not isinstance(message_data, dict):
            return {"response": None, "type": None, "sub_type": None}

        message = message_data.get("message", {})
        if not isinstance(message, dict):
            message = {}

        from_info = message.get("from", {})
        if not isinstance(from_info, dict):
            from_info = {}

        to_info = message.get("to", {})
        if not isinstance(to_info, dict):
            to_info = {}

        return {
            "response": message_data.get("response"),
            "type": message.get("type"),
            "sub_type": message.get("sub_type"),
            "msg_id": message.get("msg_id"),
            "from_uid": from_info.get("uid"),
            "from_role": from_info.get("role"),
            "to_uid": to_info.get("uid"),
            "to_role": to_info.get("role"),
            "nickname_chars": len(str(message.get("nickname") or "")),
            "time": message.get("time"),
        }

    def _record_websocket_processing_error(
        self,
        *,
        shop_id: str,
        user_id: str,
        username: str,
        queue_name: str,
        reason: str,
        error: object = None,
    ) -> None:
        """Record websocket processing failures without interrupting the live connection."""
        if not hasattr(self, "_pdd_ws_processing_errors"):
            self._pdd_ws_processing_errors = {}

        key = (str(shop_id or ""), str(user_id or ""), str(queue_name or ""))
        state = self._pdd_ws_processing_errors.get(key, {"count": 0})
        state["count"] = int(state.get("count") or 0) + 1
        state["last_reason"] = reason
        state["last_error"] = self._sanitize_for_log(error) if error is not None else ""
        state["last_at"] = time.monotonic()
        state["username"] = username
        self._pdd_ws_processing_errors[key] = state

        threshold = self.WEBSOCKET_ERROR_WARNING_THRESHOLD
        if state["count"] >= threshold:
            self.logger.warning(
                "WebSocket 消息处理连续失败: "
                f"shop_id={shop_id}, user_id={user_id}, queue={queue_name}, "
                f"count={state['count']}, reason={reason}, error={state['last_error']}"
            )

    def _clear_websocket_processing_errors(self, shop_id: str, user_id: str, queue_name: str) -> None:
        errors = getattr(self, "_pdd_ws_processing_errors", None)
        if not errors:
            return
        errors.pop((str(shop_id or ""), str(user_id or ""), str(queue_name or "")), None)

    def _log_websocket_raw(self, message_data, shop_id: str, user_id: str, username: str):
        """Log raw websocket payload for later field-path analysis."""
        meta = self._extract_ws_meta(message_data)
        payload_text = self._safe_json_dumps(message_data)
        self.logger.info(
            "PDD_WS_RAW shop_id=%s user_id=%s username=%s meta=%s payload_type=%s payload_chars=%s"
            % (
                shop_id,
                user_id,
                username,
                self._safe_json_dumps(meta),
                type(message_data).__name__,
                len(payload_text),
            )
        )

    def _log_websocket_parsed(self, pdd_message: PDDChatMessage, context: Context, queue_name: str):
        """Log parsed result to compare against the raw websocket payload."""
        kwargs = getattr(context, "kwargs", None)
        parsed_snapshot = {
            "queue_name": queue_name,
            "context_type": str(context.type) if context and context.type else "",
            "msg_id": getattr(kwargs, "msg_id", ""),
            "from_uid": getattr(kwargs, "from_uid", ""),
            "to_uid": getattr(kwargs, "to_uid", ""),
            "nickname_chars": len(str(getattr(kwargs, "nickname", "") or "")),
            "content_chars": len(str(context.content if context else "")),
            "pdd_user_msg_type": str(pdd_message.user_msg_type) if pdd_message.user_msg_type else "",
        }
        self.logger.info(f"PDD_WS_PARSED {self._safe_json_dumps(parsed_snapshot)}")

    async def _setup_message_consumer(self, queue_name: str):
        """设置消息消费者和处理器链"""
        from Agent.CustomerAgent.custom.customer_agent import CustomerAgent
        from Message import handler_chain, message_consumer_manager, queue_manager

        try:
            existing_consumer = message_consumer_manager.get_consumer(queue_name)
            if existing_consumer:
                self.logger.info(f"消费者 {queue_name} 已存在，先停止并重新创建")
                try:
                    await message_consumer_manager.stop_consumer(queue_name)
                except Exception as e:
                    self.logger.warning(f"停止旧消费者失败: {queue_name}, {self._sanitize_for_log(e)}")
                try:
                    queue_manager.recreate_queue(queue_name)
                except Exception as e:
                    self.logger.warning(f"重新创建队列失败: {queue_name}, {self._sanitize_for_log(e)}")

            consumer = message_consumer_manager.create_consumer(queue_name, max_concurrent=10)

            try:
                from core.di_container import container

                bot = container.get(CustomerAgent)
            except Exception:
                bot = CustomerAgent()
            handlers = handler_chain(use_ai=True, businessHours=self.businessHours, bot=bot)
            for handler in handlers:
                consumer.add_handler(handler)

            await message_consumer_manager.start_consumer(queue_name)
            self.logger.debug(f"消息消费者已启动: {queue_name}")

        except Exception as e:
            self.logger.error(f"设置消息消费者失败: {self._sanitize_for_log(e)}")
            raise

    @staticmethod
    def _conversation_key(queue_name: str, context: Context) -> tuple:
        return (
            queue_name,
            str(MessageHandlerMixin._context_kwarg(context, "shop_id", "") or ""),
            str(MessageHandlerMixin._context_kwarg(context, "user_id", "") or ""),
            str(MessageHandlerMixin._context_kwarg(context, "from_uid", "") or ""),
        )

    @staticmethod
    def _copy_context(context: Context, *, content: str, context_type: ContextType, msg_id: str) -> Context:
        kwargs = getattr(context, "kwargs", None)
        if hasattr(kwargs, "model_copy"):
            new_kwargs = kwargs.model_copy(update={"msg_id": msg_id, "user_msg_type": context_type})
        elif isinstance(kwargs, Mapping):
            new_kwargs = dict(kwargs)
            new_kwargs.update({"msg_id": msg_id, "user_msg_type": context_type})
        elif hasattr(kwargs, "copy"):
            new_kwargs = kwargs.copy(update={"msg_id": msg_id, "user_msg_type": context_type})
        else:
            new_kwargs = kwargs

        if hasattr(context, "model_copy"):
            return context.model_copy(update={"content": content, "type": context_type, "kwargs": new_kwargs})
        return context.copy(update={"content": content, "type": context_type, "kwargs": new_kwargs})

    @staticmethod
    def _parse_context_json(content: str) -> dict:
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _summarize_context_for_agent(self, context: Context) -> str:
        content = str(context.content or "").strip()
        if context.type == ContextType.TEXT:
            return f"客户消息：{content}" if content else ""
        if context.type in {ContextType.IMAGE, ContextType.VIDEO}:
            return f"客户发送了{'图片' if context.type == ContextType.IMAGE else '视频'}：{content}"
        if context.type in self.CARD_CONTEXT_TYPES:
            data = self._parse_context_json(content)
            if context.type == ContextType.ORDER_INFO:
                parts = [
                    f"订单号：{data.get('order_id') or ''}",
                    f"商品：{data.get('goods_name') or ''}",
                    f"商品ID：{data.get('goods_id') or ''}",
                    f"规格：{data.get('spec') or ''}",
                    f"订单主状态码：{data.get('order_status') if data.get('order_status') is not None else ''}",
                    f"物流状态码：{data.get('shipping_status') if data.get('shipping_status') is not None else ''}",
                    f"订单状态码：{data.get('status') if data.get('status') is not None else ''}",
                ]
                return "订单卡片：" + "，".join(part for part in parts if not part.endswith("："))

            parts = [
                f"商品：{data.get('goods_name') or ''}",
                f"商品ID：{data.get('goods_id') or ''}",
                f"价格：{data.get('goods_price') or ''}",
                f"规格：{data.get('goods_spec') or data.get('spec') or ''}",
                f"链接：{data.get('link_url') or ''}",
            ]
            return "商品卡片：" + "，".join(part for part in parts if not part.endswith("："))
        return content

    def _get_recent_text_for_context(self, key: tuple) -> str:
        recent = getattr(self, "_pdd_recent_customer_text", {}).get(key)
        if not recent:
            return ""
        text, created_at = recent
        if time.monotonic() - created_at > self.RECENT_TEXT_TTL_SECONDS:
            return ""
        return text

    def _merge_contexts_for_queue(self, key: tuple, contexts: list[Context]) -> Context:
        base = contexts[-1]
        msg_ids = []
        summaries = []
        has_text = False

        for item in contexts:
            msg_id = str(self._context_kwarg(item, "msg_id", "") or "").strip()
            if msg_id:
                msg_ids.append(msg_id)
            if item.type == ContextType.TEXT:
                has_text = True
            summary = self._summarize_context_for_agent(item)
            if summary and summary not in summaries:
                summaries.append(summary)

        if not has_text and base.type in self.CARD_CONTEXT_TYPES:
            recent_text = self._get_recent_text_for_context(key)
            if recent_text:
                summaries.insert(0, f"上一条客户问题：{recent_text}")
                has_text = True

        if len(summaries) <= 1 and not has_text:
            return base

        merged_type = ContextType.TEXT if has_text else base.type
        merged_content = "\n".join(summaries).strip()
        merged_msg_id = "+".join(msg_ids[-4:]) if msg_ids else str(self._context_kwarg(base, "msg_id", "") or "")
        return self._copy_context(base, content=merged_content, context_type=merged_type, msg_id=merged_msg_id)

    async def _flush_debounced_context(self, queue_name: str, key: tuple):
        buffer = None
        try:
            await asyncio.sleep(self.QUEUE_DEBOUNCE_SECONDS)
            buffers = getattr(self, "_pdd_reply_buffers", {})
            buffer = buffers.pop(key, None)
            if not buffer:
                return

            contexts = buffer.get("contexts") or []
            if not contexts:
                return

            merged_context = self._merge_contexts_for_queue(key, contexts)
            from Message import put_message

            msg_id = await put_message(queue_name, merged_context)
            self.logger.debug(
                f"合并入队: queue={queue_name}, merged_msg_id={msg_id}, "
                f"source_count={len(contexts)}, type={merged_context.type}"
            )
        except asyncio.CancelledError:
            if buffer:
                self._restore_debounce_buffer(key, buffer)
            return
        except Exception as exc:
            if buffer:
                self._restore_debounce_buffer(key, buffer)
            self.logger.error(f"合并消息入队失败: queue={queue_name}, key={key}, error={self._sanitize_for_log(exc)}")

    def _restore_debounce_buffer(self, key: tuple, buffer: dict) -> None:
        """恢复已弹出的 debounce buffer；若新 buffer 已存在则合并，避免取消竞态丢新消息。"""
        buffers = getattr(self, "_pdd_reply_buffers", {})
        existing = buffers.get(key)
        if existing is None or existing is buffer:
            buffers[key] = buffer
            return

        old_contexts = list(buffer.get("contexts") or [])
        new_contexts = list(existing.get("contexts") or [])
        existing["contexts"] = old_contexts + new_contexts

    async def _queue_message_with_debounce(self, queue_name: str, context: Context):
        if not hasattr(self, "_pdd_reply_buffers"):
            self._pdd_reply_buffers = {}
        if not hasattr(self, "_pdd_recent_customer_text"):
            self._pdd_recent_customer_text = {}

        key = self._conversation_key(queue_name, context)
        if context.type == ContextType.TEXT and str(context.content or "").strip():
            self._pdd_recent_customer_text[key] = (str(context.content).strip(), time.monotonic())

        buffer = self._pdd_reply_buffers.setdefault(key, {"contexts": [], "task": None})
        buffer["contexts"].append(context)

        old_task = buffer.get("task")
        if old_task and not old_task.done():
            old_task.cancel()

        buffer["task"] = asyncio.create_task(self._flush_debounced_context(queue_name, key))

    async def _process_websocket_message(self, message: str, shop_id: str, user_id: str, username: str, queue_name: str):
        """处理单条 WebSocket 消息"""
        try:
            if not message or not message.strip():
                self.logger.debug(f"收到空消息，跳过处理: {shop_id}-{username}")
                return

            message_data = json.loads(message)
            # self._log_websocket_raw(message_data, shop_id, user_id, username)  # 已关闭原始数据日志
            ws_meta = self._extract_ws_meta(message_data)
            msg_type = ws_meta.get("type", "unknown")
            from_uid_log = ws_meta.get("from_uid") or "unknown"
            self.logger.debug(f"收到消息: type={msg_type}, from_uid={from_uid_log}, shop_id={shop_id}")

            try:
                pdd_message = PDDChatMessage(message_data)
            except Exception as pdd_error:
                self.logger.error(f"创建 PDD 消息对象失败: {shop_id}-{username}, 错误: {self._sanitize_for_log(pdd_error)}")
                self._record_websocket_processing_error(
                    shop_id=shop_id,
                    user_id=user_id,
                    username=username,
                    queue_name=queue_name,
                    reason="pdd_message_parse",
                    error=pdd_error,
                )
                return

            try:
                context = self._convert_to_context(pdd_message, shop_id, user_id, username)
                if not context:
                    self.logger.debug(f"消息转换失败，跳过处理: {shop_id}-{username}")
                    self._record_websocket_processing_error(
                        shop_id=shop_id,
                        user_id=user_id,
                        username=username,
                        queue_name=queue_name,
                        reason="context_empty",
                    )
                    return
                # self._log_websocket_parsed(pdd_message, context, queue_name)  # 已关闭解析数据日志
            except Exception as ctx_error:
                self.logger.error(f"转换 Context 失败: {shop_id}-{username}, 错误: {self._sanitize_for_log(ctx_error)}")
                self._record_websocket_processing_error(
                    shop_id=shop_id,
                    user_id=user_id,
                    username=username,
                    queue_name=queue_name,
                    reason="context_convert",
                    error=ctx_error,
                )
                return

            if context:
                if self._should_process_immediately(context):
                    await self._handle_immediate_message(context, shop_id, user_id)
                    self.logger.debug(f"立即处理消息: {context.type}, ID: {pdd_message.msg_id}")
                elif self._should_queue_message(context):
                    await self._queue_message_with_debounce(queue_name, context)
                    self.logger.debug(f"消息等待合并入队: {queue_name}, 类型: {context.type}, ID: {pdd_message.msg_id}")
                else:
                    self.logger.debug(f"忽略消息: {context.type}, ID: {pdd_message.msg_id}")
                self._clear_websocket_processing_errors(shop_id, user_id, queue_name)
            else:
                self.logger.warning("消息转换失败，跳过处理")
                self._record_websocket_processing_error(
                    shop_id=shop_id,
                    user_id=user_id,
                    username=username,
                    queue_name=queue_name,
                    reason="context_empty",
                )

        except json.JSONDecodeError as json_error:
            self.logger.error(f"JSON 解析失败: message_chars={len(str(message or ''))}")
            self._record_websocket_processing_error(
                shop_id=shop_id,
                user_id=user_id,
                username=username,
                queue_name=queue_name,
                reason="json_decode",
                error=json_error,
            )
        except Exception as e:
            self.logger.error(f"处理 WebSocket 消息失败: {self._sanitize_for_log(e)}")
            self._record_websocket_processing_error(
                shop_id=shop_id,
                user_id=user_id,
                username=username,
                queue_name=queue_name,
                reason="unexpected",
                error=e,
            )

    def _should_process_immediately(self, context: Context) -> bool:
        """判断消息是否需要立即处理"""
        immediate_types = {
            ContextType.SYSTEM_STATUS,
            ContextType.AUTH,
            ContextType.WITHDRAW,
            ContextType.SYSTEM_HINT,
            ContextType.MALL_CS,
            ContextType.TRANSFER,
        }
        return context.type in immediate_types

    def _should_queue_message(self, context: Context) -> bool:
        """判断消息是否需要放入队列处理"""
        queue_types = {
            ContextType.TEXT,
            ContextType.IMAGE,
            ContextType.VIDEO,
            ContextType.EMOTION,
            ContextType.GOODS_INQUIRY,
            ContextType.ORDER_INFO,
            ContextType.GOODS_CARD,
            ContextType.GOODS_SPEC,
        }
        return context.type in queue_types

    async def _handle_immediate_message(self, context: Context, shop_id: str, user_id: str):
        """立即处理消息"""
        username = self._context_kwarg(context, "username", "")
        recipient_uid = self._context_kwarg(context, "from_uid", "")
        content_chars = len(str(context.content or ""))
        try:
            from Channel.pinduoduo.utils.API.send_message import SendMessage

            send_message = SendMessage(shop_id, user_id)
            if context.type == ContextType.AUTH:
                auth_info = context.content
                if isinstance(auth_info, dict):
                    result = auth_info.get("result")
                    if result == "ok":
                        self.logger.info(f"{username} 认证成功")
                    else:
                        self.logger.warning(f"{username} 认证失败")

            elif context.type == ContextType.WITHDRAW:
                self.logger.info(f"收到撤回消息: content_chars={content_chars}")
                await self._send_immediate_ack(send_message, recipient_uid, context.type)

            elif context.type == ContextType.SYSTEM_STATUS:
                self.logger.debug(f"系统状态消息: content_chars={content_chars}")

            elif context.type == ContextType.SYSTEM_HINT:
                self.logger.info(f"系统提示: content_chars={content_chars}")
                await self._send_immediate_ack(send_message, recipient_uid, context.type)

            elif context.type == ContextType.MALL_CS:
                self.logger.debug(f"收到客服消息: content_chars={content_chars}")

            elif context.type == ContextType.SYSTEM_BIZ:
                self.logger.info(f"系统业务消息: content_chars={content_chars}")

            elif context.type == ContextType.MALL_SYSTEM_MSG:
                self.logger.info(f"商城系统消息: content_chars={content_chars}")

            elif context.type == ContextType.TRANSFER:
                self.logger.info(f"转接消息: content_chars={content_chars}")
                await self._send_immediate_ack(send_message, recipient_uid, context.type)

        except Exception as e:
            self.logger.error(f"立即处理消息失败: {self._sanitize_for_log(e)}")

    async def _send_immediate_ack(self, send_message, recipient_uid: str, context_type: ContextType) -> None:
        ack_enabled = as_bool(get_config("pinduoduo.immediate_ack.enabled", True), True)
        if not ack_enabled:
            self.logger.debug(f"跳过即时确认发送，配置已关闭: context_type={context_type}")
            return

        configured_types = get_config(
            "pinduoduo.immediate_ack.context_types",
            [ContextType.WITHDRAW.value, ContextType.SYSTEM_HINT.value, ContextType.TRANSFER.value],
        )
        enabled_types = {str(item).strip().lower() for item in configured_types or [] if str(item).strip()}
        if context_type.value.lower() not in enabled_types:
            self.logger.debug(f"跳过即时确认发送，消息类型未启用: context_type={context_type}")
            return

        ack_message = str(get_config("pinduoduo.immediate_ack.message", "[玫瑰]") or "").strip()
        if not ack_message:
            self.logger.warning(f"跳过即时确认发送，回执文案为空: context_type={context_type}")
            return

        recipient_uid = str(recipient_uid or "").strip()
        if not recipient_uid:
            self.logger.warning(f"跳过即时确认发送，缺少 recipient_uid: context_type={context_type}")
            return
        try:
            send_result = await asyncio.to_thread(send_message.send_text, recipient_uid, ack_message)
        except Exception as exc:
            self.logger.warning(
                f"即时确认发送失败: context_type={context_type}, error={self._sanitize_for_log(exc)}"
            )
            return

        if isinstance(send_result, dict) and not as_bool(send_result.get("success"), False):
            self.logger.warning(
                f"即时确认发送失败: context_type={context_type}, result={self._sanitize_for_log(send_result)}"
            )

    def _convert_to_context(self, pdd_message: PDDChatMessage, shop_id: str, user_id: str, username: str) -> Context:
        """将拼多多消息转换为 Context 格式"""
        shop_info = db_manager.get_shop(self.channel_name, shop_id) or {}
        if not shop_info:
            self.logger.warning(f"店铺信息不存在，继续按 shop_id 处理消息: channel={self.channel_name}, shop_id={shop_id}")
        shop_name = shop_info.get("shop_name", "")

        content = pdd_message.content
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        elif content is None:
            content = ""
        else:
            content = str(content)

        context = Context.create_pinduoduo_context(
            content=content,
            msg_id=str(pdd_message.msg_id) if pdd_message.msg_id is not None else "",
            from_user=str(pdd_message.from_user) if pdd_message.from_user is not None else "",
            from_uid=str(pdd_message.from_uid) if pdd_message.from_uid is not None else "",
            to_user=str(pdd_message.to_user) if pdd_message.to_user is not None else "",
            to_uid=str(pdd_message.to_uid) if pdd_message.to_uid is not None else "",
            nickname=str(pdd_message.nickname) if pdd_message.nickname is not None else "",
            timestamp=pdd_message.timestamp,
            user_msg_type=pdd_message.user_msg_type,
            shop_id=str(shop_id),
            user_id=str(user_id),
            username=str(username),
            shop_name=str(shop_name),
            raw_data=pdd_message.raw_data,
            channel_type=ChannelType.PINDUODUO,
        )
        return context


__all__ = ["MessageHandlerMixin"]
