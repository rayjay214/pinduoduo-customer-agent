"""
自定义 CustomerAgent 实现

完全自主实现，不依赖 Agno 框架。

本模块已重构，职责分离为：
- agent_config.py: 配置管理
- llm_client.py: LLM 客户端封装
- message_builder.py: 消息和 Prompt 构建
- tool_executor.py: 工具执行器
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import re
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

import jieba

from Agent.bot import Bot

# 导入工具模块，触发 @agent_tool 装饰器注册
from Agent.CustomerAgent.tools import (
    move_conversation,                 # noqa: F401  — 注册 transfer_conversation 工具
    search_knowledge,                 # noqa: F401  — 注册 search_knowledge 工具
    send_product_card,                # noqa: F401  — 注册 send_product_card 工具
)
from config import get_config
from core.di_container import container
from utils.config_values import as_bool
from bridge.context import Context
from bridge.reply import Reply, ReplyType
from Agent.CustomerAgent.custom.session_manager import SessionManager
from Agent.CustomerAgent.custom.tool_decorator import execute_tool, get_tools_for_llm
from database.knowledge_service import KnowledgeService
from utils.logger_loguru import get_logger
from utils.runtime_path import get_resource_path
from utils.scene_prompt_paths import DEFAULT_SCENE_PROMPT_FILES, resolve_scene_prompt_files, scene_prompt_read_candidates
from Channel.pinduoduo.utils.API.order_manager import OrderManager, build_order_context_text
from core.base_service import _sanitize_for_log

# 导入重构后的模块
from Agent.CustomerAgent.custom.agent_config import (
    AgentConfig,
    DEFAULT_DB_PATH,
    DEFAULT_TOKEN_WINDOW,
    DEFAULT_COMPRESS_RATIO,
    DEFAULT_RETAIN_COUNT,
    DEFAULT_MAX_LOOPS,
    DEFAULT_TEMPERATURE,
)
from Agent.CustomerAgent.custom.llm_client import LLMClient, LLMResponse
from Agent.CustomerAgent.custom.media_detection import infer_media_type_from_url
from Agent.CustomerAgent.custom.message_builder import MessageBuilder
from Agent.CustomerAgent.custom.prompt_rules import build_image_grounding_constraint
from Agent.CustomerAgent.custom.tool_executor import ToolExecutor, ToolResult, _TOOL_EXECUTOR
from Agent.CustomerAgent.custom.knowledge_action_router import sanitize_final_reply
from Agent.CustomerAgent.custom.turn_context import TurnContext, parse_turn_context
from utils.night_mode import (
    NIGHT_MODE_TRANSFER_RESULT_PREFIX,
    get_night_mode_prompt_values,
    is_night_mode,
)

logger = get_logger("CustomerAgent")


def _sanitize_exception_for_log(exc: Exception) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}: {_sanitize_for_log(message)}"
    return type(exc).__name__


def _knowledge_service() -> KnowledgeService:
    try:
        return container.get(KnowledgeService)
    except Exception:
        return KnowledgeService()


SESSION_GOODS_ID_CACHE_LIMIT = 5000
DEFAULT_DAYTIME_NIGHT_MODE_LEAK_MARKERS = (
    "夜间时段",
    "夜间不转人工",
    "高级客服已下班",
    "高级客服下班了",
    "高级客服目前下班",
    "当前高级客服不在线",
    "还没上班",
    "上班时间是早上8点",
    "上班时间为早上8点",
    "高级客服上班时间",
    "建议您晚点联系",
    "建议您晚点再联系",
)


class CustomerAgent(Bot):
    """
    自定义客服 Agent

    核心循环：
    1. 加载历史消息
    2. 检查上下文压缩
    3. 构建 messages 列表
    4. 调用 LLM → 解析 tool_calls
    5. 并行执行工具 → 回传结果
    6. 循环直到无工具调用
    7. 返回最终回复

    职责已分离到子模块：
    - AgentConfig: 配置管理
    - LLMClient: LLM API 调用
    - MessageBuilder: 消息和 Prompt 构建
    - ToolExecutor: 工具执行
    - SessionManager: 会话管理（已有独立模块）
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        token_window: int = DEFAULT_TOKEN_WINDOW,
        compress_ratio: float = DEFAULT_COMPRESS_RATIO,
        retain_count: int = DEFAULT_RETAIN_COUNT,
        max_loops: int = DEFAULT_MAX_LOOPS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        super().__init__()
        self._is_initialized = False

        # 配置参数
        self._config = AgentConfig(
            db_path=db_path or DEFAULT_DB_PATH,
            token_window=token_window,
            compress_ratio=compress_ratio,
            retain_count=retain_count,
            max_loops=max_loops,
            temperature=temperature,
        )

        # 子组件（延迟初始化）
        self._llm_client: Optional[LLMClient] = None
        self._message_builder: Optional[MessageBuilder] = None
        self._tool_executor: Optional[ToolExecutor] = None
        self._session_manager: Optional[SessionManager] = None
        self._tools: List[Dict[str, Any]] = []
        self._scene_prompt_cache: Dict[str, Dict[str, Any]] = {}
        self._session_goods_id_cache: OrderedDict[str, int] = OrderedDict()  # session_id -> goods_id
        self._anonymous_session_id = f"unknown:fallback:{id(self)}"

        logger.info("CustomerAgent 实例创建成功")

    async def initialize_async(self) -> bool:
        """异步初始化 Agent"""
        if self._is_initialized:
            return True

        try:
            # 1. 从配置文件加载配置
            self._config = AgentConfig.load_from_config()

            # 2. 验证配置
            if not self._config.validate():
                return False

            # 3. 初始化 LLM 客户端
            self._llm_client = LLMClient(
                api_key=self._config.api_key,
                api_base=self._config.api_base,
                model_name=self._config.model_name,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                tool_call_max_tokens=self._config.tool_call_max_tokens,
                request_timeout_seconds=self._config.request_timeout_seconds,
                max_concurrent_requests=self._config.max_concurrent_requests,
                fallback_api_key=self._config.fallback_api_key,
                fallback_api_base=self._config.fallback_api_base,
                fallback_model_name=self._config.fallback_model_name,
                fallback_timeout_seconds=self._config.fallback_timeout_seconds,
                fallback_enabled=self._config.fallback_enabled,
                disable_thinking=self._config.disable_thinking,
                disable_thinking_api_base_patterns=self._config.disable_thinking_api_base_patterns,
                disable_thinking_model_prefixes=self._config.disable_thinking_model_prefixes,
            )
            await self._llm_client.initialize()

            # 4. 初始化会话管理器
            self._session_manager = SessionManager(
                db_path=self._config.db_path,
                token_window=self._config.token_window,
                compress_ratio=self._config.compress_ratio,
                retain_count=self._config.retain_count,
                model_name=self._config.model_name,
            )

            # 5. 初始化消息构建器
            self._message_builder = MessageBuilder()

            # 6. 初始化工具执行器
            self._tool_executor = ToolExecutor()

            # 7. 加载工具列表
            self._tools = get_tools_for_llm()
            self._llm_client.tools = self._tools
            tool_names = [t.get("function", {}).get("name", "unknown") for t in self._tools]
            logger.info(f"已加载 {len(self._tools)} 个工具: {tool_names}")

            self._is_initialized = True
            logger.info(f"CustomerAgent 初始化成功: model={self._config.model_name}")
            return True

        except Exception:
            logger.exception("CustomerAgent 初始化失败")
            return False

    async def async_reply(self, query: str, context: Context = None) -> Reply:
        """异步回复接口"""
        # 延迟初始化
        if not self._is_initialized:
            if not await self.initialize_async():
                return Reply(ReplyType.TEXT, "AI客服初始化失败，请检查配置。")

        session_id: Optional[str] = None
        user_message_saved = False

        try:
            # 构建 session_id 和 dependencies
            if context and context.channel_type and self._context_kwarg(context, "user_id"):
                dependencies = self._message_builder.build_dependencies(context)
                session_id = self._build_session_id(context, dependencies)
            else:
                # 降级：尽量从上下文字段构造稳定会话，避免同文本客户串话。
                session_id = self._build_fallback_session_id(context)
                dependencies = {}

            # goods_id 会话级缓存：首次获取后复用，避免后续纯文本消息丢失商品上下文
            current_gid = dependencies.get("goods_id")
            if current_gid:
                self._remember_session_goods_id(session_id, current_gid)
            else:
                self._restore_session_goods_id(session_id, dependencies)

            # TurnContext: log-only 模式
            turn_context = dependencies.get("turn_context")
            if turn_context is None and as_bool(get_config("enable_turn_context", False), False):
                turn_context = parse_turn_context(str(query or ""))
                dependencies["turn_context"] = turn_context
            if turn_context is not None and as_bool(get_config("enable_turn_context_log_only", True), True):
                self._log_turn_context(session_id, turn_context)

            await self._refresh_order_context(dependencies)

            # 加载历史并检查压缩；场景判定需要历史和订单上下文。
            history = self._session_manager.get_history(session_id)
            if self._session_manager.should_compress(session_id):
                logger.info(f"触发上下文压缩: session_id={session_id}")
                await self._compress_with_llm(session_id, history)

            customer_scene = self._resolve_customer_scene(query, history, dependencies)
            dependencies["_customer_scene"] = customer_scene
            self._log_scene_resolution(session_id, customer_scene, dependencies, query)

            # 纯商品卡没有客户真实问题，直接追问意图，避免模型展开商品参数。
            if self._is_product_card_only_turn(turn_context):
                self._session_manager.add_message(
                    session_id=session_id,
                    role="user",
                    content=query,
                )
                user_message_saved = True
                final_content = "亲，您想了解这款商品的哪方面呢？"
                self._session_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=final_content,
                )
                return Reply(ReplyType.TEXT, final_content)

            # 售后场景收到图片/视频时，直接转人工，避免模型看图/看视频臆断。
            if customer_scene == "aftersale" and self._has_media_input(dependencies):
                self._session_manager.add_message(
                    session_id=session_id,
                    role="user",
                    content=query,
                )
                user_message_saved = True
                final_content = await self._transfer_to_human(dependencies, session_id, reason="aftersale_media")
                self._session_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=final_content,
                )
                return Reply(ReplyType.TEXT, final_content)

            # 售后高风险问题直接转人工，避免模型编造处理方案。
            if customer_scene == "aftersale" and self._is_high_risk_aftersale_transfer_issue(query):
                self._session_manager.add_message(
                    session_id=session_id,
                    role="user",
                    content=query,
                )
                user_message_saved = True
                final_content = await self._transfer_to_human(
                    dependencies,
                    session_id,
                    reason="high_risk_aftersale_issue",
                )
                self._session_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=final_content,
                )
                return Reply(ReplyType.TEXT, final_content)

            # 视频/图片无文字时直接追问，不走 LLM
            if self._is_media_only_query(query, dependencies):
                self._session_manager.add_message(
                    session_id=session_id,
                    role="user",
                    content=query,
                )
                user_message_saved = True
                final_content = "麻烦您说下具体想确认哪里"
                self._session_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=final_content,
                )
                return Reply(ReplyType.TEXT, final_content)

            if self._should_ask_for_product_identity(query, dependencies):
                self._session_manager.add_message(
                    session_id=session_id,
                    role="user",
                    content=query,
                )
                user_message_saved = True
                final_content = self._missing_goods_identity_question()
                self._session_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=final_content,
                )
                return Reply(ReplyType.TEXT, final_content)

            # 构建 messages
            messages = self._message_builder.build_messages(query, history, dependencies)
            self._session_manager.add_message(
                session_id=session_id,
                role="user",
                content=query,
            )
            user_message_saved = True
            self._append_scene_prompt(messages, customer_scene)
            self._append_night_mode_constraint(messages)
            self._append_order_hard_constraints(messages, customer_scene, dependencies, session_id, query)
            self._append_unconfirmed_receipt_constraint(messages, customer_scene, dependencies, session_id, query)
            self._append_image_grounding_constraint(messages, dependencies, session_id)
            await asyncio.to_thread(
                self._inject_pre_retrieved_knowledge,
                messages,
                query,
                dependencies,
                customer_scene,
                history,
            )
            self._append_missing_goods_knowledge_constraint(messages, query, dependencies, session_id)

            # 执行 Agent 循环
            final_content = await self._run_agent_loop(messages, dependencies)
            final_content = self._sanitize_daytime_night_mode_reply(final_content, messages)
            final_content = sanitize_final_reply(final_content)

            # 回复去重：与最近一条 assistant 回复比较
            final_content = await self._dedup_reply(
                final_content, messages, history, session_id,
            )
            logger.info(
                "[最终回复] session={} scene={} reply_len={}".format(
                    session_id,
                    customer_scene,
                    len(final_content or ""),
                )
            )

            # 保存最终回复到历史
            self._session_manager.add_message(
                session_id=session_id,
                role="assistant",
                content=final_content,
            )

            return Reply(ReplyType.TEXT, final_content or "亲，客服正在为您处理，请稍等片刻哦～")

        except Exception:
            logger.exception("CustomerAgent 回复失败")
            fallback_content = "亲，客服正在为您处理，请稍等片刻哦～"
            if user_message_saved and session_id and self._session_manager:
                try:
                    self._session_manager.add_message(
                        session_id=session_id,
                        role="assistant",
                        content=fallback_content,
                    )
                except Exception:
                    logger.exception(f"保存异常兜底回复失败: session_id={session_id}")
            return Reply(ReplyType.TEXT, fallback_content)

    def _build_session_id(self, context: Context, dependencies: Dict[str, Any]) -> str:
        """按店铺账号和买家隔离会话历史，避免不同客户互相污染。"""
        channel = str(context.channel_type.value if context.channel_type else "unknown")
        shop_id = str(dependencies.get("shop_id") or self._context_kwarg(context, "shop_id", "") or "")
        user_id = str(dependencies.get("user_id") or self._context_kwarg(context, "user_id", "") or "")
        recipient_uid = str(
            dependencies.get("recipient_uid")
            or dependencies.get("from_uid")
            or self._context_kwarg(context, "from_uid", "")
            or ""
        )

        if recipient_uid:
            return f"{channel}:{shop_id}:{user_id}:{recipient_uid}"
        return self._build_fallback_session_id(context)

    @staticmethod
    def _context_kwarg(context: Context, key: str, default: Any = None) -> Any:
        kwargs = getattr(context, "kwargs", None)
        if isinstance(kwargs, Mapping):
            return kwargs.get(key, default)
        return getattr(kwargs, key, default)

    def _remember_session_goods_id(self, session_id: str, goods_id: Any) -> None:
        """Remember the latest goods_id for a session using LRU eviction."""
        if not session_id or not goods_id:
            return
        self._session_goods_id_cache[str(session_id)] = goods_id
        self._session_goods_id_cache.move_to_end(str(session_id))
        while len(self._session_goods_id_cache) > SESSION_GOODS_ID_CACHE_LIMIT:
            self._session_goods_id_cache.popitem(last=False)

    def _restore_session_goods_id(self, session_id: str, dependencies: Dict[str, Any]) -> None:
        """Restore cached goods_id and refresh LRU position."""
        if not session_id or session_id not in self._session_goods_id_cache:
            return
        dependencies["goods_id"] = self._session_goods_id_cache[session_id]
        self._session_goods_id_cache.move_to_end(session_id)

    def _build_fallback_session_id(self, context: Optional[Context]) -> str:
        """没有完整 PDD kwargs 时，仍用稳定上下文字段隔离会话。"""
        if context is None:
            return self._anonymous_session_id

        channel = str(
            context.channel_type.value
            if getattr(context, "channel_type", None) and hasattr(context.channel_type, "value")
            else getattr(context, "channel_type", None) or "unknown"
        )
        parts = [
            channel,
            str(getattr(context, "type", "") or ""),
            str(self._context_kwarg(context, "shop_id", "") or ""),
            str(self._context_kwarg(context, "user_id", "") or ""),
            str(self._context_kwarg(context, "from_uid", "") or ""),
            str(self._context_kwarg(context, "to_uid", "") or ""),
            str(self._context_kwarg(context, "msg_id", "") or ""),
        ]
        raw_data = self._context_kwarg(context, "raw_data", None)
        if isinstance(raw_data, Mapping):
            message = raw_data.get("message") if isinstance(raw_data.get("message"), Mapping) else {}
            sender = message.get("from") if isinstance(message.get("from"), Mapping) else {}
            receiver = message.get("to") if isinstance(message.get("to"), Mapping) else {}
            parts.extend(
                [
                    str(raw_data.get("conversation_id") or raw_data.get("session_id") or ""),
                    str(message.get("conversation_id") or message.get("session_id") or ""),
                    str(sender.get("uid") or ""),
                    str(receiver.get("uid") or ""),
                    str(message.get("msg_id") or ""),
                ]
            )
        key = "|".join(part for part in parts if part)
        if not key or key in {"unknown"}:
            return self._anonymous_session_id
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return f"{channel}:fallback:{digest}"

    def _sanitize_daytime_night_mode_reply(
        self,
        content: str,
        messages: List[Dict[str, Any]],
    ) -> str:
        """非夜间时清理被历史污染带出的夜间转人工话术。"""
        reply = str(content or "")
        if not reply or is_night_mode():
            return reply

        leak_markers = self._daytime_night_mode_leak_markers()
        if not any(marker in reply for marker in leak_markers):
            return reply

        tool_contents = [
            str(msg.get("content") or "")
            for msg in messages
            if isinstance(msg, dict) and msg.get("role") == "tool"
        ]
        logger.warning(f"非夜间回复包含夜间话术，已清理: reply_chars={len(reply)}")
        if any(text.strip() == "会话转接成功" for text in tool_contents):
            return "亲，已经为您转接人工客服，会尽快为您处理，请稍等一下～"

        marker_pattern = "|".join(re.escape(marker) for marker in leak_markers)
        cleaned = re.sub(
            rf"[^。！？!?]*?(?:{marker_pattern})[^。！？!?]*[。！？!?]?",
            "",
            reply,
        ).strip(" ，,。.!！~～")
        return cleaned or "亲，客服正在为您处理，请稍等片刻哦～"

    @staticmethod
    def _tool_transfer_result_reply(content: Any) -> str:
        text = str(content or "").strip()
        if text == "会话转接成功":
            return "亲，已为您转接人工处理，请稍等。"
        if text in {"当前无可用的人工客服", "指定人工客服当前不可转接"}:
            return "亲，当前人工客服暂时不可转接，您先把问题发我，我这边继续帮您记录。"
        if (
            text.startswith("转接失败")
            or text.startswith("转接过程中发生错误")
            or "缺少必要的会话信息" in text
            or "工具执行错误" in text
        ):
            return "亲，转人工暂时没成功，您先把问题发我，我这边继续帮您看。"
        return ""

    @staticmethod
    def _daytime_night_mode_leak_markers() -> tuple[str, ...]:
        configured = get_config("agent.daytime_night_mode_leak_markers", None)
        if configured is None:
            configured = DEFAULT_DAYTIME_NIGHT_MODE_LEAK_MARKERS
        elif not isinstance(configured, (list, tuple)):
            configured = DEFAULT_DAYTIME_NIGHT_MODE_LEAK_MARKERS

        markers = []
        for marker in configured:
            text = str(marker or "").strip()
            if text:
                markers.append(text)
        return tuple(markers)

    @staticmethod
    def _log_turn_context(session_id: str, tc: TurnContext) -> None:
        """log-only 模式：记录 TurnContext 结构化数据，不参与回复流程。"""
        logger.info(
            "[TurnContext] session={session} customer_text_chars={ct_len} "
            "has_product_card={hpc} has_order_card={hoc} has_media={hm} "
            "goods_id={gid} has_order_sn={has_osn} scene={scene} warnings={warn}".format(
                session=session_id,
                ct_len=len(str(tc.customer_text or "")),
                hpc=tc.turn_type.has_product_card,
                hoc=tc.turn_type.has_order_card,
                hm=tc.turn_type.has_media,
                gid=tc.product_card.goods_id or "",
                has_osn=bool(tc.order_card.order_sn),
                scene=tc.raw_scene_hint or "",
                warn=tc.parse_warnings,
            )
        )

    @staticmethod
    def _log_scene_resolution(
        session_id: str,
        customer_scene: str,
        dependencies: Dict[str, Any],
        query: str,
    ) -> None:
        logger.info(
            "[场景判定] session={} scene={} shop_id={} goods_id={} context_type={} query_chars={}".format(
                session_id,
                customer_scene,
                dependencies.get("shop_id"),
                dependencies.get("goods_id"),
                dependencies.get("context_type"),
                len(str(query or "")),
            )
        )

    @staticmethod
    def _is_product_card_only_turn(tc: Any) -> bool:
        """只有商品卡、没有客户文字/订单/媒体时，不进入大模型回答。"""
        if not isinstance(tc, TurnContext):
            return False
        return (
            tc.turn_type.has_product_card
            and not tc.turn_type.has_text
            and not tc.turn_type.has_order_card
            and not tc.turn_type.has_media
        )

    async def _run_agent_loop(
        self,
        messages: List[Dict[str, Any]],
        dependencies: Dict[str, Any],
    ) -> str:
        """
        Agent 循环核心

        调用 LLM → 检查 tool_calls → 并行执行工具 → 回传结果 → 循环
        """
        loop_count = 0

        while loop_count < self._config.max_loops:
            # 1. 调用 LLM
            try:
                response = await self._llm_client.chat(messages, tool_choice="auto")
            except Exception as e:
                logger.error(f"LLM 调用失败: {_sanitize_exception_for_log(e)}")
                if loop_count == 0:
                    return "亲，客服正在为您处理，请稍等片刻哦～"
                # 已有中间结果，返回已生成的内容
                for msg in reversed(messages):
                    if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                        return msg["content"]
                return "亲，客服正在为您处理，请稍等片刻哦～"

            # 2. 解析响应
            if not response.has_tool_calls:
                # 无工具调用，返回内容
                content = response.content or ""
                messages.append({"role": "assistant", "content": content})
                return content

            # 3. 保存 assistant 消息（包含 tool_calls）
            assistant_msg = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    ToolExecutor.to_assistant_tool_call(tc, index)
                    for index, tc in enumerate(response.tool_calls)
                ],
            }
            messages.append(assistant_msg)

            # 4. 并行执行所有工具调用
            tool_names = [
                str(ToolExecutor.get_tool_call_function(tc)[0] or "unknown_tool")
                for tc in response.tool_calls
            ]
            logger.info(
                "[工具调用] tools={} scene={} shop_id={} goods_id={}".format(
                    tool_names,
                    dependencies.get("_customer_scene"),
                    dependencies.get("shop_id"),
                    dependencies.get("goods_id"),
                )
            )
            tool_results = await self._tool_executor.execute_parallel(
                response.tool_calls, dependencies
            )

            # 6. 夜间转人工结果直接返回，不交给 LLM 二次生成
            for result in tool_results:
                if result.content and result.content.startswith(NIGHT_MODE_TRANSFER_RESULT_PREFIX):
                    # 提取前缀后的客户可见回复
                    night_reply = result.content
                    sep_idx = night_reply.find("：")
                    if sep_idx != -1:
                        night_reply = night_reply[sep_idx + 1:]
                    night_reply = night_reply.strip()
                    if night_reply:
                        logger.info(f"[夜间转人工直返] reply_chars={len(night_reply)}")
                        return night_reply
                transfer_reply = self._tool_transfer_result_reply(result.content)
                if transfer_reply:
                    logger.info(f"[工具转人工直返] reply_chars={len(transfer_reply)}")
                    return transfer_reply

            # 7. 将结果追加到消息列表
            for result in tool_results:
                messages.append(result.to_dict())

            # 8. 检查循环上限。必须先追加 tool 消息，再请求最终回复，否则会违反工具调用协议。
            if loop_count >= self._config.max_loops - 1:
                logger.warning(f"工具调用达到上限 {self._config.max_loops}，基于工具结果强制生成最终回复")
                messages.append({
                    "role": "user",
                    "content": "[已达到最大工具调用次数，请基于已有工具结果给出最终客户回复，不要再调用工具。]",
                })
                try:
                    final_response = await self._llm_client.chat(messages, tool_choice="none")
                    if final_response.content:
                        return final_response.content
                except Exception as exc:
                    logger.warning(f"工具循环上限后的最终回复生成失败: {_sanitize_exception_for_log(exc)}")

                for result in reversed(tool_results):
                    if result.content:
                        return result.content
                return assistant_msg["content"] or "亲，客服正在为您处理，请稍等片刻哦～"

            loop_count += 1

        # 兜底
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("content"):
                return str(msg.get("content") or "")
        return ""

    def _resolve_customer_scene(
        self,
        query: str,
        history: List[Dict[str, Any]],
        dependencies: Dict[str, Any],
    ) -> str:
        """基于订单/物流/确认收到判断当前会话大场景。"""
        cached_scene = KnowledgeService.normalize_customer_scene(dependencies.get("_customer_scene")) or ""
        if cached_scene in {"presale", "insale", "aftersale"}:
            return cached_scene

        order_scene = str(dependencies.get("order_scene_hint") or "")
        if order_scene == "mixed_orders":
            return "insale"

        has_order = bool(
            dependencies.get("order_id")
            or order_scene in {"aftersale", "insale"}
        )
        if not has_order:
            return "presale"

        if self._is_order_signed(dependencies):
            return "aftersale"

        combined = str(query or "")
        recent_user_text = " ".join(
            str(msg.get("content") or "")
            for msg in history[-6:]
            if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content")
        )
        if recent_user_text:
            combined = f"{combined} {recent_user_text}"
        if self._has_received_confirmation(combined):
            return "aftersale"

        return "insale"

    async def _refresh_order_context(self, dependencies: Dict[str, Any]) -> None:
        """Fetch latest read-only PDD order context for scene detection."""
        shop_id = dependencies.get("shop_id")
        account_user_id = dependencies.get("user_id")
        customer_uid = dependencies.get("recipient_uid") or dependencies.get("from_uid")
        if not shop_id or not account_user_id or not customer_uid:
            return

        def load_context() -> Dict[str, Any]:
            manager = OrderManager(shop_id=str(shop_id), user_id=str(account_user_id))
            orders = manager.get_user_orders(str(customer_uid), page_size=5)
            return build_order_context_text(orders)

        try:
            context = await asyncio.to_thread(load_context)
        except Exception as exc:
            logger.warning(
                "[订单上下文] 获取失败 shop_id={} account_user_id={} customer_uid={} error={}".format(
                    shop_id,
                    account_user_id,
                    customer_uid,
                    exc,
                )
            )
            return

        dependencies["order_context_text"] = context.get("text") or ""
        dependencies["order_scene_hint"] = context.get("scene_hint") or ""
        dependencies["order_business_status"] = context.get("business_status") or ""
        dependencies["order_shipping_status"] = context.get("shipping_status") or ""
        dependencies["order_latest_trace"] = context.get("latest_trace") or ""
        dependencies["order_id"] = context.get("order_id") or ""
        logger.info(
            "[订单上下文] scene_hint={} business_status={} shipping_status={} order_id={} latest_trace={}".format(
                dependencies.get("order_scene_hint"),
                dependencies.get("order_business_status"),
                dependencies.get("order_shipping_status"),
                dependencies.get("order_id"),
                str(dependencies.get("order_latest_trace") or "")[:100],
            )
        )

    def _load_scene_prompt(self, customer_scene: str) -> str:
        scene_key = KnowledgeService.normalize_customer_scene(customer_scene) or "presale"
        configured = get_config("agent.scene_prompt_files", DEFAULT_SCENE_PROMPT_FILES)
        candidates = scene_prompt_read_candidates(scene_key, configured)
        if not candidates:
            return ""

        last_error = None
        for relative_path in candidates:
            try:
                prompt_path = get_resource_path(relative_path)
                stat = prompt_path.stat()
                cached = self._scene_prompt_cache.get(scene_key)
                if (
                    cached
                    and cached.get("path") == str(prompt_path)
                    and cached.get("mtime_ns") == stat.st_mtime_ns
                    and cached.get("size") == stat.st_size
                ):
                    return str(cached.get("prompt") or "")

                prompt = prompt_path.read_text(encoding="utf-8").strip()
                self._scene_prompt_cache[scene_key] = {
                    "path": str(prompt_path),
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                    "prompt": prompt,
                }
                return prompt
            except Exception as exc:
                last_error = exc
                continue

        self._scene_prompt_cache.pop(scene_key, None)
        logger.warning(f"场景 Prompt 读取失败: scene={scene_key}, error={_sanitize_exception_for_log(last_error)}")
        return ""

    @staticmethod
    def _scene_prompt_files() -> Dict[str, str]:
        configured = get_config("agent.scene_prompt_files", DEFAULT_SCENE_PROMPT_FILES)
        return resolve_scene_prompt_files(configured)

    def _append_scene_prompt(self, messages: List[Dict[str, Any]], customer_scene: str) -> None:
        prompt = self._load_scene_prompt(customer_scene)
        if not prompt:
            return

        scene_label = KnowledgeService.customer_scene_label(customer_scene)
        scene_note = (
            "【当前会话场景规则】\n"
            f"当前场景规则按“{scene_label}”加载；场景名仅供内部判断，不要对客户输出。\n\n"
            f"{prompt}"
        )

        self._prepend_system_note(messages, scene_note)

    @staticmethod
    def _prepend_system_note(messages: List[Dict[str, Any]], note: str) -> None:
        """Append a note to the leading system message, or insert one if the head is malformed."""
        if (
            messages
            and isinstance(messages[0], dict)
            and messages[0].get("role") == "system"
        ):
            existing = str(messages[0].get("content") or "").strip()
            if note not in existing:
                messages[0]["content"] = f"{existing}\n\n{note}" if existing else note
            return
        messages.insert(0, {"role": "system", "content": note})

    DEFAULT_SIGNED_TRACE_KEYWORDS = (
        "包裹已签收",
        "快件已签收",
        "签收人是",
        "已签收",
        "已收货",
        "已取件",
    )

    @classmethod
    def _is_order_signed(cls, dependencies: Dict[str, Any]) -> bool:
        """判断订单是否已签收。"""
        if dependencies.get("order_shipping_status") == "signed":
            return True
        biz = str(dependencies.get("order_business_status") or "")
        if "已签收" in biz or "已收货" in biz:
            return True
        trace = str(dependencies.get("order_latest_trace") or "")
        if any(kw in trace for kw in cls._signed_trace_keywords()):
            return True
        return False

    @classmethod
    def _signed_trace_keywords(cls) -> tuple[str, ...]:
        configured = get_config("pinduoduo.order.signed_trace_keywords", None)
        if configured is None:
            configured = cls.DEFAULT_SIGNED_TRACE_KEYWORDS
        elif not isinstance(configured, (list, tuple)):
            configured = cls.DEFAULT_SIGNED_TRACE_KEYWORDS

        keywords = []
        for keyword in configured:
            text = str(keyword or "").strip()
            if text:
                keywords.append(text)
        return tuple(keywords)

    DEFAULT_ORDER_FAULT_EXAMPLES = (
        "噪音",
        "异响",
        "无法使用",
        "功能异常",
        "坏了",
    )

    @classmethod
    def _order_fault_examples(cls) -> tuple[str, ...]:
        configured = get_config("agent.order_fault_examples", None)
        if configured is None:
            configured = cls.DEFAULT_ORDER_FAULT_EXAMPLES
        elif not isinstance(configured, (list, tuple)):
            configured = cls.DEFAULT_ORDER_FAULT_EXAMPLES
        return tuple(str(item or "").strip() for item in configured if str(item or "").strip())

    @classmethod
    def _order_hard_constraint(cls) -> str:
        examples = cls._order_fault_examples()
        example_text = f"例如{'、'.join(examples)}，" if examples else ""
        return (
            '【订单硬约束】\n'
            '系统根据订单状态和最新物流判断：当前客户已收到商品。\n'
            '禁止使用"收到货后、等收到后、到货后、拿到货后、先试用、先试试"等暗示客户尚未收到商品的话术。\n'
            f'如果客户反馈商品问题，{example_text}应按售后问题处理；'
            '无法直接解决时使用转人工工具。\n'
        )

    DEFAULT_USAGE_FEEDBACK_KEYWORDS = (
        "好响", "声音大", "噪音大", "滋滋", "异响",
        "没反应", "坏了", "不转", "充不进", "充不上",
        "正在用", "收到货了", "已经收到", "已收到",
        "无法使用", "功能异常",
    )

    DEFAULT_UNRECEIVED_PATTERNS = (
        "还没收到",
        "还没有收到",
        "没收到",
        "没有收到",
        "未收到",
        "未收货",
        "没收货",
        "没有收货",
        "未签收",
    )

    @classmethod
    def _has_usage_feedback(cls, query: str) -> bool:
        """客户明确反馈使用体验时，才按已收到商品处理。"""
        text = str(query or "")
        if any(pattern in text for pattern in cls._unreceived_patterns()):
            return False

        keywords = cls._usage_feedback_keywords()
        if not keywords:
            return False

        text_lower = text.lower()
        has_product_or_fault_context = any(
            token in text_lower
            for token in (
                "风扇",
                "商品",
                "产品",
                "电机",
                "噪音",
                "声音",
                "异响",
                "故障",
                "坏了",
                "没反应",
                "不转",
                "充不进",
                "充不上",
                "无法使用",
                "功能异常",
                "使用",
                "正在用",
                "收到货",
                "已收到",
            )
        )
        has_service_dispute_context = any(
            actor in text_lower
            for actor in ("客服", "商家", "卖家", "平台", "人工", "店家")
        ) and any(
            marker in text_lower
            for marker in ("吵", "投诉", "举报", "态度", "介入", "骂", "争执")
        )
        if has_service_dispute_context and not has_product_or_fault_context:
            return False

        for keyword in keywords:
            if keyword not in text:
                continue
            return True
        return False

    DEFAULT_RECEIVED_CONFIRMATION_KEYWORDS = (
        "收到货了",
        "已经收到",
        "已收到",
        "收到了",
        "已收货",
        "收到商品",
        "拿到货了",
        "拿到了",
        "拆开了",
        "打开包装",
        "正在用",
        "用了",
    )

    @classmethod
    def _has_received_confirmation(cls, query: str) -> bool:
        """客户明确表示已收到/已使用，才允许未签收订单切售后。"""
        text = str(query or "")
        if not text or any(pattern in text for pattern in cls._unreceived_patterns()):
            return False
        configured = get_config("agent.received_confirmation_keywords", None)
        if configured is None:
            configured = cls.DEFAULT_RECEIVED_CONFIRMATION_KEYWORDS
        elif not isinstance(configured, (list, tuple)):
            configured = cls.DEFAULT_RECEIVED_CONFIRMATION_KEYWORDS
        keywords = tuple(str(item or "").strip() for item in configured if str(item or "").strip())
        return any(keyword in text for keyword in keywords)

    @classmethod
    def _usage_feedback_keywords(cls) -> tuple[str, ...]:
        configured = get_config("agent.usage_feedback_keywords", None)
        if configured is None:
            configured = cls.DEFAULT_USAGE_FEEDBACK_KEYWORDS
        elif not isinstance(configured, (list, tuple)):
            configured = cls.DEFAULT_USAGE_FEEDBACK_KEYWORDS
        keywords = tuple(str(item or "").strip() for item in configured if str(item or "").strip())
        return keywords

    @classmethod
    def _unreceived_patterns(cls) -> tuple[str, ...]:
        configured = get_config("agent.unreceived_patterns", None)
        if configured is None:
            configured = cls.DEFAULT_UNRECEIVED_PATTERNS
        elif not isinstance(configured, (list, tuple)):
            configured = cls.DEFAULT_UNRECEIVED_PATTERNS
        patterns = tuple(str(item or "").strip() for item in configured if str(item or "").strip())
        return patterns

    def _append_order_hard_constraints(
        self,
        messages: List[Dict[str, Any]],
        customer_scene: str,
        dependencies: Dict[str, Any],
        session_id: str,
        query: str = "",
    ) -> None:
        """已签收或客户有使用反馈时注入订单硬约束（所有场景）。"""
        scene = KnowledgeService.normalize_customer_scene(customer_scene)
        is_signed = self._is_order_signed(dependencies)
        has_usage_feedback = self._has_usage_feedback(query)

        if not is_signed and not has_usage_feedback:
            return
        # presale 场景且无签收信息时跳过（避免纯售前咨询误注入）
        if scene == "presale" and not is_signed:
            return

        order_id = dependencies.get("order_id") or ""
        shipping = dependencies.get("order_shipping_status") or ""
        logger.info(
            "[订单硬约束] aftersale signed injected: session={} order_id={} shipping_status={}".format(
                session_id, order_id, shipping,
            )
        )

        self._prepend_system_note(messages, self._order_hard_constraint())

    DEFAULT_UNCONFIRMED_RECEIPT_AFTERSALE_TERMS = (
        "坏了",
        "坏的",
        "不能用",
        "用不了",
        "没反应",
        "不转",
        "不出风",
        "充不了",
        "充不进",
        "声音大",
        "噪音",
        "异响",
        "破损",
        "少件",
        "发错",
        "退货",
        "退款",
        "退钱",
        "售后",
        "质量问题",
    )

    @classmethod
    def _unconfirmed_receipt_aftersale_terms(cls) -> tuple[str, ...]:
        configured = get_config("agent.unconfirmed_receipt_aftersale_terms", None)
        if configured is None:
            configured = cls.DEFAULT_UNCONFIRMED_RECEIPT_AFTERSALE_TERMS
        elif not isinstance(configured, (list, tuple)):
            configured = cls.DEFAULT_UNCONFIRMED_RECEIPT_AFTERSALE_TERMS
        return tuple(str(item or "").strip() for item in configured if str(item or "").strip())

    @classmethod
    def _has_unconfirmed_receipt_aftersale_request(cls, query: str) -> bool:
        text = str(query or "")
        if not text or cls._has_received_confirmation(text):
            return False
        return any(term in text for term in cls._unconfirmed_receipt_aftersale_terms())

    @staticmethod
    def _unconfirmed_receipt_constraint() -> str:
        return (
            "【未确认收货约束】\n"
            "当前订单状态和最新物流尚未确认客户已收到商品。\n"
            "如果客户反馈故障、质量问题、退货或退款诉求，先简短确认客户是否已经收到商品；"
            "未确认收到前，不要直接按售后已收货场景承诺退货、退款、补偿或处理结果。\n"
            "不要向客户提到内部场景、订单硬约束或系统判断。\n"
        )

    def _append_unconfirmed_receipt_constraint(
        self,
        messages: List[Dict[str, Any]],
        customer_scene: str,
        dependencies: Dict[str, Any],
        session_id: str,
        query: str,
    ) -> None:
        scene = KnowledgeService.normalize_customer_scene(customer_scene)
        if scene != "insale":
            return
        if not (dependencies.get("order_id") or dependencies.get("order_context_text")):
            return
        if self._is_order_signed(dependencies) or self._has_received_confirmation(query):
            return
        if not self._has_unconfirmed_receipt_aftersale_request(query):
            return

        logger.info(
            "[未确认收货约束] injected: session={} order_id={} shipping_status={}".format(
                session_id,
                dependencies.get("order_id") or "",
                dependencies.get("order_shipping_status") or "",
            )
        )
        self._prepend_system_note(messages, self._unconfirmed_receipt_constraint())

    DEFAULT_MISSING_GOODS_PARAMETER_KEYWORDS = (
        "商品参数",
        "参数",
        "规格",
        "型号",
        "款式",
        "尺寸",
        "尺码",
        "大小",
        "长度",
        "宽度",
        "高度",
        "重量",
        "容量",
        "功率",
        "电压",
        "材质",
        "面料",
        "成分",
        "保质期",
        "有效期",
        "生产日期",
        "怎么用",
        "使用方法",
        "安装",
        "配套",
        "配件",
        "赠品",
        "发几个",
        "按键",
        "按钮",
        "开关",
        "图标",
        "标识",
        "功能",
        "颜色",
        "库存",
    )
    DEFAULT_MISSING_GOODS_PARAMETER_TOPICS = (
        "参数",
        "规格",
        "尺寸",
        "材质",
        "按键/图标/功能",
        "颜色",
        "配件",
    )
    DEFAULT_MISSING_GOODS_UNVERIFIED_FACT_EXAMPLES = (
        "参数数值",
        "规格型号",
        "材质成分",
        "尺寸重量",
        "功能",
        "赠品承诺",
    )

    @classmethod
    def _is_missing_goods_id(cls, dependencies: Dict[str, Any]) -> bool:
        goods_id = dependencies.get("goods_id")
        if goods_id is None:
            return True
        text = str(goods_id).strip().lower()
        return text in {"", "none", "null", "0"}

    @classmethod
    def _is_product_parameter_query(cls, query: str) -> bool:
        text = str(query or "").lower()
        return any(keyword in text for keyword in cls._missing_goods_parameter_keywords())

    @classmethod
    def _missing_goods_parameter_keywords(cls) -> tuple[str, ...]:
        configured = get_config("agent.missing_goods_parameter_keywords", None)
        if configured is None:
            configured = cls.DEFAULT_MISSING_GOODS_PARAMETER_KEYWORDS
        elif not isinstance(configured, (list, tuple)):
            configured = cls.DEFAULT_MISSING_GOODS_PARAMETER_KEYWORDS

        keywords = []
        for item in configured:
            text = str(item or "").strip().lower()
            if text:
                keywords.append(text)
        return tuple(keywords)

    @classmethod
    def _missing_goods_parameter_topics(cls) -> tuple[str, ...]:
        configured = get_config("agent.missing_goods_parameter_topics", None)
        if configured is None:
            configured = cls.DEFAULT_MISSING_GOODS_PARAMETER_TOPICS
        elif not isinstance(configured, (list, tuple)):
            configured = cls.DEFAULT_MISSING_GOODS_PARAMETER_TOPICS
        return tuple(str(item or "").strip() for item in configured if str(item or "").strip())

    @classmethod
    def _missing_goods_unverified_fact_examples(cls) -> tuple[str, ...]:
        configured = get_config("agent.missing_goods_unverified_fact_examples", None)
        if configured is None:
            configured = cls.DEFAULT_MISSING_GOODS_UNVERIFIED_FACT_EXAMPLES
        elif not isinstance(configured, (list, tuple)):
            configured = cls.DEFAULT_MISSING_GOODS_UNVERIFIED_FACT_EXAMPLES
        return tuple(str(item or "").strip() for item in configured if str(item or "").strip())

    @classmethod
    def _missing_goods_knowledge_constraint(cls) -> str:
        topics = cls._missing_goods_parameter_topics()
        facts = cls._missing_goods_unverified_fact_examples()
        topic_text = "、".join(topics) if topics else "依赖具体商品的信息"
        fact_text = "、".join(facts) if facts else "具体参数或承诺"
        return (
            "【商品知识状态】\n"
            "当前会话没有识别到 goods_id，无法加载该商品的专属知识。\n"
            f"如果客户询问{topic_text}等依赖具体商品的信息，"
            f"不要根据商品名、昵称、图片符号或经验猜测具体参数，不要编造{fact_text}。\n"
            "应回复：亲，不同款式/规格参数会不一样，麻烦您发一下具体商品链接或点一下商品卡片，我按对应款式帮您确认哦。\n"
            "不要向客户提到“知识库、goods_id、系统无法加载、商品知识状态”等内部信息。\n"
        )

    def _append_missing_goods_knowledge_constraint(
        self,
        messages: List[Dict[str, Any]],
        query: str,
        dependencies: Dict[str, Any],
        session_id: str,
    ) -> None:
        """无 goods_id 的商品参数问题注入约束，避免模型编造具体商品参数。"""
        if not self._is_missing_goods_id(dependencies):
            return
        if not self._is_product_parameter_query(query):
            return

        logger.info(
            "[无商品知识约束] injected: session={} shop_id={} query_chars={}".format(
                session_id,
                dependencies.get("shop_id"),
                len(str(query or "")),
            )
        )
        messages.append({"role": "system", "content": self._missing_goods_knowledge_constraint()})

    @classmethod
    def _should_ask_for_product_identity(cls, query: str, dependencies: Dict[str, Any]) -> bool:
        """缺少商品身份时，商品参数类问题必须先反问，不能让模型猜。"""
        return cls._is_missing_goods_id(dependencies) and cls._is_product_parameter_query(query)

    @staticmethod
    def _missing_goods_identity_question() -> str:
        return "亲，不同款式/规格参数会不一样，麻烦您发一下具体商品链接或点一下商品卡片，我按对应款式帮您确认哦。"

    DEFAULT_NIGHT_MODE_FAULT_EXAMPLES = (
        "坏了",
        "没反应",
        "不转",
        "充不进电",
        "噪音大",
    )

    @classmethod
    def _night_mode_fault_examples(cls) -> tuple[str, ...]:
        configured = get_config("agent.night_mode_fault_examples", None)
        if configured is None:
            configured = cls.DEFAULT_NIGHT_MODE_FAULT_EXAMPLES
        elif not isinstance(configured, (list, tuple)):
            configured = cls.DEFAULT_NIGHT_MODE_FAULT_EXAMPLES
        return tuple(str(item or "").strip() for item in configured if str(item or "").strip())

    @classmethod
    def _night_mode_fault_constraint(cls) -> str:
        values = get_night_mode_prompt_values()
        examples = cls._night_mode_fault_examples()
        example_text = f"（{'、'.join(examples)}）" if examples else ""
        return (
            "【夜间模式约束】\n"
            f"当前为夜间值守时段（{values['range_text']}），无法转接人工客服。\n"
            '禁止回复"已为您转接""稍后会有专员""已转人工"等虚假转接话术。\n'
            f"如果客户反馈商品故障{example_text}，"
            f"应回复：已记录您的问题，夜间无法转接人工，建议您{values['resume_text']}后联系，会有专人为您处理。\n"
            "如果客户反复反馈同一问题，不要重复相同话术，简短确认已记录即可。\n"
        )

    @staticmethod
    def _has_media_input(dependencies: Dict[str, Any]) -> bool:
        """判断本轮是否包含图片或视频。"""
        context_type = str(dependencies.get("context_type") or "")
        media_type = str(dependencies.get("media_type") or "")
        media_url = str(dependencies.get("media_url") or "").lower()
        if context_type in {"image", "video"} or media_type in {"image", "video"}:
            return True
        return infer_media_type_from_url(media_url) in {"image", "video"}

    @staticmethod
    def _has_image_input(dependencies: Dict[str, Any]) -> bool:
        context_type = str(dependencies.get("context_type") or "")
        media_type = str(dependencies.get("media_type") or "")
        media_url = str(dependencies.get("media_url") or "")
        return (
            context_type == "image"
            or media_type == "image"
            or infer_media_type_from_url(media_url) == "image"
        )

    DEFAULT_HIGH_RISK_AFTERSALE_TRANSFER_PHRASES = ()

    @classmethod
    def _is_high_risk_aftersale_transfer_issue(cls, query: str) -> bool:
        text = str(query or "").replace(" ", "")
        return any(phrase in text for phrase in cls._high_risk_aftersale_transfer_phrases())

    @classmethod
    def _high_risk_aftersale_transfer_phrases(cls) -> tuple[str, ...]:
        configured = get_config(
            "agent.high_risk_aftersale_transfer_phrases",
            list(cls.DEFAULT_HIGH_RISK_AFTERSALE_TRANSFER_PHRASES),
        )
        if not isinstance(configured, (list, tuple)):
            configured = cls.DEFAULT_HIGH_RISK_AFTERSALE_TRANSFER_PHRASES

        phrases = []
        for phrase in configured:
            text = str(phrase or "").replace(" ", "")
            if text:
                phrases.append(text)
        return tuple(phrases)

    @staticmethod
    def _is_media_only_query(query: str, dependencies: Dict[str, Any]) -> bool:
        """判断是否只有图片/视频、没有文字问题。"""
        context_type = str(dependencies.get("context_type") or "")
        media_type = str(dependencies.get("media_type") or "")
        is_media = context_type in {"image", "video"} or media_type in {"image", "video"}
        if not is_media:
            return False
        # 检查 query 是否包含客户实际文字（排除 URL、[视频消息]、[图片消息] 等）
        text = str(query or "").strip()
        # 去掉 URL
        text_no_url = re.sub(r"https?://[^\s]+", "", text).strip()
        # 去掉标记
        text_no_url = re.sub(r"\[(视频|图片|video|image)消息\]", "", text_no_url, flags=re.IGNORECASE).strip()
        text_no_url = re.sub(r"客户消息[：:]\s*", "", text_no_url).strip()
        text_no_url = re.sub(r"客户发送了(图片|视频)[：:]*\s*", "", text_no_url).strip()
        return len(text_no_url) < 2

    def _append_image_grounding_constraint(
        self,
        messages: List[Dict[str, Any]],
        dependencies: Dict[str, Any],
        session_id: str,
    ) -> None:
        """有图片输入时约束模型不要把可见符号臆断为商品功能。"""
        if not self._has_image_input(dependencies):
            return
        constraint = build_image_grounding_constraint()
        if not constraint:
            return

        logger.info(
            "[图片硬约束] injected: session={} shop_id={} goods_id={} media_type={}".format(
                session_id,
                dependencies.get("shop_id"),
                dependencies.get("goods_id"),
                dependencies.get("media_type"),
            )
        )
        self._prepend_system_note(messages, constraint)

    def _append_night_mode_constraint(self, messages: List[Dict[str, Any]]) -> None:
        """夜间模式时注入约束，防止 LLM 生成虚假转接话术。"""
        if not is_night_mode():
            return
        constraint = self._night_mode_fault_constraint()
        self._prepend_system_note(messages, constraint)

    async def _transfer_to_human(
        self,
        dependencies: Dict[str, Any],
        session_id: str,
        reason: str,
    ) -> str:
        """预检无法安全自动处理时转人工，不经过 LLM。"""
        try:
            isolated_dependencies = copy.deepcopy(dependencies)
        except Exception:
            isolated_dependencies = dict(dependencies)

        result = await asyncio.get_running_loop().run_in_executor(
            _TOOL_EXECUTOR,
            execute_tool,
            "transfer_conversation",
            "{}",
            isolated_dependencies,
        )
        content = str(result or "").strip()
        logger.info(f"[售后直转人工] session={session_id} reason={reason} result_chars={len(content)}")

        if content.startswith(NIGHT_MODE_TRANSFER_RESULT_PREFIX):
            sep_idx = content.find("：")
            if sep_idx != -1:
                content = content[sep_idx + 1:]
            content = content.strip()
            if content:
                return sanitize_final_reply(content)

        if "会话转接成功" in content:
            return "亲，已转人工为您处理，请稍等。"

        if "当前无可用" in content or "不可转接" in content:
            return "亲，当前人工客服暂时不可转接，您先把问题发我，我这边继续帮您记录。"

        if "转接失败" in content or "缺少必要的会话信息" in content or "工具执行错误" in content:
            return "亲，转人工暂时没成功，您先把问题发我，我这边继续帮您看。"

        return "亲，已为您转接人工处理，请稍等。"

    def _inject_pre_retrieved_knowledge(
        self,
        messages: List[Dict[str, Any]],
        query: str,
        dependencies: Dict[str, Any],
        customer_scene: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """预检索知识并注入 system prompt，提高第一轮回复稳定性。"""
        try:
            # 从 dependencies 获取 shop_id 和 goods_id
            shop_id = dependencies.get("shop_id")
            goods_id = dependencies.get("goods_id")

            # 参数校验：goods_id 可选（没有则只查店铺通用知识）
            if not shop_id or not customer_scene:
                return
            if customer_scene not in {"presale", "insale", "aftersale"}:
                return

            current_query = self._knowledge_retrieval_query(query)
            retrieval_query = self._contextual_knowledge_retrieval_query(current_query, history)

            # 调用检索
            ks = _knowledge_service()
            results = ks.search_scene_knowledge(
                scene=customer_scene,
                shop_id=shop_id,
                goods_id=goods_id,
                query=retrieval_query,
                limit=3,
            )

            if not results:
                logger.info(
                    "[预检索] scene={} shop_id={} goods_id={} hit=0 query_chars={}".format(
                        customer_scene,
                        shop_id,
                        goods_id,
                        len(str(retrieval_query or "")),
                    )
                )
                return

            valid_results = [item for item in results if isinstance(item, dict)]
            if not valid_results:
                return

            top1 = valid_results[0]
            logger.info(
                "[预检索] scene={} shop_id={} goods_id={} hit={} top1_section={} top1_intent={} match={} score={} query_chars={}".format(
                    customer_scene,
                    shop_id,
                    goods_id,
                    len(valid_results),
                    top1.get("section_title") or "",
                    top1.get("sub_intent") or "",
                    top1.get("match_type") or "",
                    top1.get("score") or "",
                    len(str(retrieval_query or "")),
                )
            )

            # 构建注入内容，控制长度
            knowledge_lines = []
            total_length = 0
            max_single = 300
            max_total = 1200

            for i, r in enumerate(valid_results, 1):
                answer = str(r.get("answer") or "")[:max_single]
                section = str(r.get("section_title") or "")
                sub_intent = str(r.get("sub_intent") or "")
                tags = str(r.get("tags") or "").strip()
                source_type = str(r.get("source_type") or "").strip()
                match_type = str(r.get("match_type") or "").strip()

                entry = f"{i}. 分类：{section}\n   意图：{sub_intent}\n   答案：{answer}"
                metadata_lines = []
                if tags:
                    metadata_lines.append(f"标签：{tags}")
                if source_type:
                    metadata_lines.append(f"来源：{source_type}")
                if match_type:
                    metadata_lines.append(f"匹配：{match_type}")
                if metadata_lines:
                    entry += "\n   " + "\n   ".join(metadata_lines)
                if total_length + len(entry) > max_total:
                    break
                knowledge_lines.append(entry)
                total_length += len(entry)

            if not knowledge_lines:
                return

            context_lines = [
                f"客户当前问题：{current_query}",
                f"当前场景：{customer_scene}",
            ]
            if retrieval_query != current_query:
                context_lines.append(f"上下文检索问题：{retrieval_query}")
            if dependencies.get("order_shipping_status"):
                context_lines.append(f"订单物流状态：{dependencies.get('order_shipping_status')}")
            if dependencies.get("order_business_status"):
                context_lines.append(f"订单业务状态：{dependencies.get('order_business_status')}")
            if goods_id:
                context_lines.append(f"当前商品ID：{goods_id}")

            # 组装注入文本
            inject_text = (
                "【本轮预检索知识】\n"
                "以下知识由系统根据当前店铺、商品、场景和客户问题自动检索，仅供本轮回复使用。\n"
                + "\n".join(context_lines)
                + "\n"
                "先判断客户当前问题主题、场景和订单/商品上下文，再选择能直接回答该主题的候选知识。\n"
                "不要只因为某条候选排序靠前就使用它；候选与当前问题主题冲突时，应忽略该候选并继续检索或转人工。\n"
                "如果候选知识能回答客户问题，优先按知识直接回复。\n"
                "不要告诉客户“知识库、RAG、预检索、系统检索”等内部信息。\n\n"
                + "\n\n".join(knowledge_lines)
            )

            # 注入到 messages[0]（system prompt）
            if (
                messages
                and isinstance(messages[0], dict)
                and messages[0].get("role") == "system"
            ):
                existing = str(messages[0].get("content") or "").strip()
                messages[0]["content"] = f"{existing}\n\n{inject_text}" if existing else inject_text
            else:
                messages.insert(0, {"role": "system", "content": inject_text})

            logger.debug(f"预检索知识注入完成: scene={customer_scene}, 条数={len(knowledge_lines)}")

        except Exception as e:
            logger.warning(f"预检索知识注入失败（不影响正常流程）: {_sanitize_exception_for_log(e)}")

    @staticmethod
    def _knowledge_retrieval_query(query: str) -> str:
        """预检索只使用客户真实文本，避免商品卡片价格/标题污染匹配。"""
        text = str(query or "").strip()
        marker = "客户消息："
        if marker not in text:
            return text

        customer_part = text.split(marker, 1)[1]
        stop_markers = (
            "\n商品卡片：",
            "\n商品：",
            "\n订单信息：",
            "\n物流信息：",
            "\n客户发送了图片",
            "\n客户发送了视频",
        )
        for stop in stop_markers:
            if stop in customer_part:
                customer_part = customer_part.split(stop, 1)[0]
        return customer_part.strip() or text

    @classmethod
    def _contextual_knowledge_retrieval_query(
        cls,
        current_query: str,
        history: Optional[List[Dict[str, Any]]],
    ) -> str:
        """Build a retrieval query that uses recent customer context only when it helps."""
        current = str(current_query or "").strip()
        if not current or not history:
            return current

        recent_user_queries: List[str] = []
        for msg in reversed(history[-8:]):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = cls._knowledge_retrieval_query(str(msg.get("content") or ""))
            if content and content != current:
                recent_user_queries.append(content)
            if len(recent_user_queries) >= 3:
                break

        if not recent_user_queries:
            return current

        current_terms = cls._retrieval_signal_terms(current)
        current_topics = cls._retrieval_topic_hints(current)
        context_terms: List[str] = []
        seen = set(current_terms)
        for text in recent_user_queries:
            text_topics = cls._retrieval_topic_hints(text)
            if current_topics and (not text_topics or not (current_topics & text_topics)):
                continue
            for term in cls._retrieval_signal_terms(text):
                if term in seen:
                    continue
                seen.add(term)
                context_terms.append(term)
                if len(context_terms) >= 8:
                    break
            if len(context_terms) >= 8:
                break

        if not context_terms:
            return current

        if cls._needs_context_for_retrieval(current):
            return f"{current} {' '.join(context_terms)}".strip()

        overlap = bool(current_terms and any(term in context_terms for term in current_terms))
        if overlap:
            return f"{current} {' '.join(context_terms[:4])}".strip()
        return current

    @classmethod
    def _needs_context_for_retrieval(cls, text: str) -> bool:
        clean = cls._normalize_retrieval_text(text)
        if not clean:
            return False
        if len(clean) <= 6:
            return True
        pronoun_markers = ("这个", "这款", "那个", "那款", "它", "上面", "刚才", "前面")
        if any(marker in clean for marker in pronoun_markers) and len(clean) <= 14:
            return True
        terms = cls._retrieval_signal_terms(clean)
        return len(terms) <= 1 and len(clean) <= 10

    @staticmethod
    def _normalize_retrieval_text(text: str) -> str:
        return re.sub(r"[\s，。！？!?,.;；:：、~～\"'“”‘’（）()【】\[\]{}<>《》]+", "", str(text or ""))

    @classmethod
    def _retrieval_signal_terms(cls, text: str) -> List[str]:
        clean = cls._normalize_retrieval_text(text)
        if not clean:
            return []
        stop_terms = {
            "客户消息",
            "内容",
            "商品卡片",
            "商品",
            "价格",
            "链接",
            "您好",
            "你好",
            "亲",
            "这个",
            "这款",
            "那个",
            "那款",
            "可以",
            "一下",
            "吗",
            "呢",
        }
        terms: List[str] = []
        if "充一次" in clean and "充电" not in clean:
            terms.append("充电")
        if "用多久" in clean and "续航" not in clean:
            terms.append("续航")
        for word in jieba.cut_for_search(clean):
            term = str(word or "").strip()
            if len(term) < 2 or term in stop_terms:
                continue
            if term.isdigit():
                continue
            terms.append(term)
        if not terms and len(clean) >= 2 and clean not in stop_terms:
            terms.append(clean)
        return terms[:12]

    @classmethod
    def _retrieval_topic_hints(cls, text: str) -> set[str]:
        clean = cls._normalize_retrieval_text(text)
        if not clean:
            return set()
        groups = {
            "price": ("价格", "多少钱", "优惠", "便宜", "贵", "券", "活动"),
            "battery": ("电池", "电量", "容量", "多大电", "毫安", "mah", "充电", "续航", "多久", "用多久"),
            "wind": ("风力", "风大", "档位", "几档", "风速"),
            "noise": ("声音", "噪音", "静音", "吵"),
            "logistics": ("快递", "物流", "发货", "到哪", "到货", "什么时候到"),
            "aftersale": ("退货", "退款", "换货", "坏了", "质保", "保修", "售后"),
            "color": ("颜色", "色", "白色", "黑色", "绿色", "粉色"),
        }
        hints = set()
        lower = clean.lower()
        for name, markers in groups.items():
            if any(marker in lower for marker in markers):
                hints.add(name)
        return hints

    async def _dedup_reply(
        self,
        content: str,
        messages: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
        session_id: str,
    ) -> str:
        """如果最终回复与上一条 assistant 回复完全相同，保留原回复。"""
        if not content or len(content) < 4:
            return content

        # 找最近一条 assistant 回复
        last_assistant = ""
        for msg in reversed(history):
            if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                last_assistant = str(msg["content"]).strip()
                break

        if not last_assistant:
            return content

        # 规范化比较：去除空白、标点
        def _norm(text: str) -> str:
            return re.sub(r"[\s。！？!?，,~～.]+", "", text)

        if _norm(content) != _norm(last_assistant):
            return content

        # 不再二次调用模型重写。旧逻辑复用带工具说明的 messages，可能把内部 tool_call 格式输出给客户。
        _ = messages
        logger.warning(f"[回复去重] 检测到重复回复，保留原回复: session={session_id}, content_chars={len(content)}")
        return content

    async def _compress_with_llm(
        self,
        session_id: str,
        history: List[Dict[str, Any]],
    ) -> None:
        """使用 LLM 生成摘要并压缩历史"""
        if not self._llm_client or not self._session_manager:
            return

        retain_count = max(0, int(getattr(self._session_manager, "retain_count", 0) or 0))
        old_history = history[:-retain_count] if retain_count else history
        if not old_history:
            return

        summary_prompt = (
            "请简洁地总结以下对话的要点，保留关键信息和用户意图。\n\n"
            f"对话内容（共 {len(old_history)} 条消息）：\n"
            + "\n".join(
                f"[{msg.get('role', 'unknown')}]: {str(msg.get('content', ''))[:200]}"
                for msg in old_history
                if msg.get("content")
            )
        )

        try:
            response = await self._llm_client.chat(
                messages=[
                    {"role": "system", "content": "你是一个对话摘要助手。请简洁地总结对话要点。"},
                    {"role": "user", "content": summary_prompt},
                ],
                tool_choice="none",
            )
            summary = str(response.content or "").strip()
        except Exception as exc:
            logger.error(f"生成会话摘要失败，保留原历史不压缩: {_sanitize_exception_for_log(exc)}")
            return

        self._session_manager.compress_history(session_id, lambda _: summary)
