"""
LLM 客户端模块

封装与 LLM API 的交互，提供类型安全的请求和响应处理。
"""
from __future__ import annotations

import asyncio
import weakref
from collections.abc import Mapping
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError("openai package is required: pip install openai>=1.109.1")

from utils.logger_loguru import get_logger
from utils.config_values import as_bool, as_float, as_int
from utils.volcengine_models import ChatCompletionsRequest
from core.base_service import _sanitize_for_log

logger = get_logger("LLMClient")


_LLM_SEMAPHORE_LOCKS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = weakref.WeakKeyDictionary()
_LLM_SEMAPHORES: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]]" = weakref.WeakKeyDictionary()
# Backwards-compatible diagnostics for existing tests/debugging.
_LLM_SEMAPHORE_LOCK: Optional[asyncio.Lock] = None
_LLM_SEMAPHORE: Optional[asyncio.Semaphore] = None


@dataclass
class LLMResponse:
    """LLM 响应封装"""
    content: Optional[str]
    tool_calls: Optional[List[Any]]
    raw_response: Any
    reasoning_content: Optional[str] = None

    @property
    def has_tool_calls(self) -> bool:
        """是否有工具调用"""
        return self.tool_calls is not None and len(self.tool_calls) > 0


class LLMClient:
    """LLM 客户端封装"""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        tool_call_max_tokens: int = 256,
        request_timeout_seconds: float = 20.0,
        max_concurrent_requests: int = 2,
        fallback_api_key: str = "",
        fallback_api_base: str = "",
        fallback_model_name: str = "",
        fallback_timeout_seconds: float = 20.0,
        fallback_enabled: bool = False,
        disable_thinking: bool = True,
        disable_thinking_api_base_patterns: Optional[List[str]] = None,
        disable_thinking_model_prefixes: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        初始化 LLM 客户端

        Args:
            api_key: API 密钥
            api_base: API 基础地址
            model_name: 模型名称
            temperature: 温度参数
            tools: 可用工具列表
        """
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max(1, as_int(max_tokens, 1))
        self.tool_call_max_tokens = max(self.max_tokens, as_int(tool_call_max_tokens, self.max_tokens))
        self.request_timeout_seconds = max(1.0, as_float(request_timeout_seconds or 20.0, 20.0))
        self.max_concurrent_requests = max(1, as_int(max_concurrent_requests or 2, 2))
        self.fallback_api_key = fallback_api_key
        self.fallback_api_base = fallback_api_base
        self.fallback_model_name = fallback_model_name
        self.fallback_timeout_seconds = max(
            1.0,
            as_float(fallback_timeout_seconds or self.request_timeout_seconds, self.request_timeout_seconds),
        )
        self.fallback_enabled = as_bool(fallback_enabled, False) and bool(fallback_api_key and fallback_model_name)
        self.disable_thinking = as_bool(disable_thinking, True)
        self.disable_thinking_api_base_patterns = tuple(
            str(item).lower()
            for item in (
                disable_thinking_api_base_patterns
                if disable_thinking_api_base_patterns is not None
                else (
                    "127.0.0.1",
                    "localhost",
                    "xiaomimimo.com",
                    "siliconflow.cn",
                )
            )
            if str(item).strip()
        )
        self.disable_thinking_model_prefixes = tuple(
            str(item).lower()
            for item in (
                disable_thinking_model_prefixes
                if disable_thinking_model_prefixes is not None
                else (
                    "mimo-",
                    "glm-",
                    "qwen/",
                    "nex-agi/",
                )
            )
            if str(item).strip()
        )
        self.tools = tools or []

        self._client: Optional[AsyncOpenAI] = None
        self._fallback_client: Optional[AsyncOpenAI] = None

    @staticmethod
    async def _get_global_semaphore(limit: int) -> asyncio.Semaphore:
        global _LLM_SEMAPHORE_LOCK, _LLM_SEMAPHORE
        normalized_limit = max(1, as_int(limit or 1, 1))
        current_loop = asyncio.get_running_loop()
        lock = _LLM_SEMAPHORE_LOCKS.get(current_loop)
        if lock is None:
            lock = asyncio.Lock()
            _LLM_SEMAPHORE_LOCKS[current_loop] = lock
        _LLM_SEMAPHORE_LOCK = lock

        async with lock:
            cached = _LLM_SEMAPHORES.get(current_loop)
            if cached is None or cached[0] != normalized_limit:
                semaphore = asyncio.Semaphore(normalized_limit)
                _LLM_SEMAPHORES[current_loop] = (normalized_limit, semaphore)
                _LLM_SEMAPHORE = semaphore
                logger.info(f"LLM API 并发限制已设置为 {normalized_limit}")
            else:
                _LLM_SEMAPHORE = cached[1]
            return _LLM_SEMAPHORE

    @staticmethod
    def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """确保 system 消息只出现在最前面。"""
        if not messages:
            return []

        system_contents: List[str] = []
        other_messages: List[Dict[str, Any]] = []

        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "system":
                content = str(message.get("content") or "").strip()
                if content:
                    system_contents.append(content)
            else:
                other_messages.append(message)

        normalized_messages: List[Dict[str, Any]] = []
        if system_contents:
            normalized_messages.append({
                "role": "system",
                "content": "\n\n".join(system_contents),
            })
        normalized_messages.extend(other_messages)
        return normalized_messages

    @staticmethod
    def _normalize_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """确保每个 tool 都显式带上 type=function。"""
        normalized_tools: List[Dict[str, Any]] = []

        for tool in tools or []:
            if not isinstance(tool, Mapping):
                continue
            raw_function = tool.get("function", {})
            if not isinstance(raw_function, Mapping):
                continue
            function = dict(raw_function)
            if not str(function.get("name") or "").strip():
                continue
            raw_parameters = function.get("parameters", {})
            parameters = dict(raw_parameters) if isinstance(raw_parameters, Mapping) else {}
            function["parameters"] = parameters

            normalized_tools.append({
                "type": "function",
                "function": function,
            })

        return normalized_tools

    def _should_disable_chat_template_thinking_for(self, api_base: str, model_name: str) -> bool:
        """Thinking models can spend the whole completion budget on reasoning."""
        if not self.disable_thinking:
            return False
        api_base = str(api_base or "").lower()
        model_name = str(model_name or "").lower()
        return any(pattern in api_base for pattern in self.disable_thinking_api_base_patterns) or any(
            model_name.startswith(prefix)
            for prefix in self.disable_thinking_model_prefixes
        )

    def _should_disable_chat_template_thinking(self) -> bool:
        return self._should_disable_chat_template_thinking_for(self.api_base, self.model_name)

    async def initialize(self) -> None:
        """初始化 OpenAI 客户端"""
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base or None,
            timeout=self.request_timeout_seconds,
        )
        if self.fallback_enabled:
            self._fallback_client = AsyncOpenAI(
                api_key=self.fallback_api_key,
                base_url=self.fallback_api_base or None,
                timeout=self.fallback_timeout_seconds,
            )
            logger.info(
                f"LLM兜底模型已启用: primary={self.model_name}, fallback={self.fallback_model_name}, "
                f"timeout={self.request_timeout_seconds}s/{self.fallback_timeout_seconds}s"
            )
        logger.debug(f"LLM 客户端初始化成功: model={self.model_name}")

    async def _create_completion(
        self,
        client: AsyncOpenAI,
        payload: Dict[str, Any],
        model_name: str,
        timeout_seconds: float,
    ) -> Any:
        request_payload = dict(payload)
        request_payload["model"] = model_name
        return await asyncio.wait_for(
            client.chat.completions.create(**request_payload),
            timeout=timeout_seconds,
        )

    @staticmethod
    def _extract_response_message(response: Any, model_name: str) -> Any:
        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError(f"LLM response missing choices: model={model_name}")
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None:
            raise RuntimeError(f"LLM response missing message: model={model_name}")
        return message

    @staticmethod
    def _tool_call_name_for_log(tool_call: Any) -> str:
        function = (
            tool_call.get("function")
            if isinstance(tool_call, dict)
            else getattr(tool_call, "function", None)
        )
        name = function.get("name") if isinstance(function, dict) else getattr(function, "name", None)
        text = str(name or "").strip()
        return text or "unknown_tool"

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tool_choice: str = "auto",
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """
        发送聊天请求到 LLM

        Args:
            messages: 消息列表
            tool_choice: 工具选择策略

        Returns:
            LLMResponse 封装的响应
        """
        if not self._client:
            raise RuntimeError("LLM 客户端未初始化，请先调用 initialize()")

        normalized_messages = self._normalize_messages(messages)
        selected_tools = self.tools if tools is None else tools
        normalized_tools = self._normalize_tools(selected_tools)

        # 1. 构建请求参数字典
        effective_max_tokens = self.max_tokens
        if normalized_tools and tool_choice != "none":
            effective_max_tokens = self.tool_call_max_tokens

        request_dict: Dict[str, Any] = {
            "model": self.model_name,
            "messages": normalized_messages,
            "temperature": self.temperature,
            "max_tokens": effective_max_tokens,
        }

        if normalized_tools and tool_choice != "none":
            request_dict["tools"] = normalized_tools
            request_dict["tool_choice"] = tool_choice

        # 2. 使用 Pydantic 模型验证请求参数
        try:
            validated_request = ChatCompletionsRequest(**request_dict)
            logger.debug("请求参数验证通过")
        except Exception as e:
            safe_error = _sanitize_for_log(e)
            logger.error(f"请求参数验证失败: {safe_error}")
            raise RuntimeError(str(safe_error)) from e

        # 3. 调试日志：输出发送给 LLM 的消息（限制内容长度，避免泄露敏感信息）
        logger.debug(f"发送给 LLM 的消息数: {len(normalized_messages)}")
        for i, msg in enumerate(normalized_messages):
            role = msg.get("role", "unknown")
            # 只记录消息角色和长度，不记录内容（避免泄露用户隐私）
            content = str(msg.get("content", ""))
            logger.debug(f"消息 {i} [{role}]: 长度={len(content)}")

        # 4. 调用 API
        payload = validated_request.model_dump(exclude_none=True)
        payload["messages"] = normalized_messages
        logger.info(f"LLM请求 max_tokens={payload.get('max_tokens')}, effective={effective_max_tokens}, model={self.model_name}")

        if normalized_tools and tool_choice != "none":
            payload["tools"] = normalized_tools
            payload["tool_choice"] = tool_choice
        else:
            payload.pop("tools", None)
            payload.pop("tool_choice", None)

        if not payload.get("logprobs"):
            payload.pop("logprobs", None)
            payload.pop("top_logprobs", None)

        # 禁用 thinking 模式，避免 reasoning 吃光 tokens
        if self._should_disable_chat_template_thinking():
            payload["extra_body"] = {
                "enable_thinking": False,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            }

        semaphore = await self._get_global_semaphore(self.max_concurrent_requests)
        async with semaphore:
            try:
                response = await self._create_completion(
                    self._client,
                    payload,
                    self.model_name,
                    self.request_timeout_seconds,
                )
                used_model = self.model_name
                message = self._extract_response_message(response, used_model)
                # 提取 reasoning_content（思考模型会返回）
                reasoning_content = getattr(message, 'reasoning_content', None)
                if reasoning_content:
                    logger.debug(f"LLM reasoning ({used_model}): reasoning_chars={len(str(reasoning_content))}")
                # 用 content 作为实际回复
                actual_content = getattr(message, "content", None)
                tool_calls = getattr(message, "tool_calls", None)
                if not tool_calls and not str(actual_content or "").strip():
                    # 如果 content 为空但有 reasoning，说明 reasoning 吃掉了所有 tokens
                    if reasoning_content:
                        logger.warning(f"LLM content 为空但 reasoning 存在 ({used_model}): reasoning长度={len(str(reasoning_content))}")
                        raise RuntimeError(f"LLM返回空内容(推理占用): model={used_model}")
                    raise RuntimeError(f"LLM返回空内容: model={used_model}")
            except Exception as primary_exc:
                # 记录主模型失败时的详细信息（token、reasoning等）
                primary_resp = locals().get('response')
                primary_reasoning = locals().get('reasoning_content')
                primary_actual = locals().get('actual_content')
                primary_msg = locals().get('message')
                if primary_resp and hasattr(primary_resp, 'usage') and primary_resp.usage:
                    logger.warning(
                        f"主模型失败Token详情: model={self.model_name}, "
                        f"total={primary_resp.usage.total_tokens}, "
                        f"prompt={primary_resp.usage.prompt_tokens}, "
                        f"completion={primary_resp.usage.completion_tokens}, "
                        f"reasoning_chars={len(str(primary_reasoning)) if primary_reasoning else 0}, "
                        f"content_chars={len(str(primary_actual)) if primary_actual else 0}, "
                        f"has_tool_calls={bool(primary_msg and getattr(primary_msg, 'tool_calls', None))}"
                    )
                elif primary_resp:
                    logger.warning(f"主模型失败但无usage数据: model={self.model_name}")
                else:
                    logger.warning(f"主模型调用异常(无response): model={self.model_name}, error={_sanitize_for_log(primary_exc)}")

                if not self.fallback_enabled or not self._fallback_client:
                    raise RuntimeError(str(_sanitize_for_log(primary_exc))) from primary_exc
                primary_error = _sanitize_for_log(primary_exc)
                logger.opt(exception=primary_exc).warning(
                    f"主模型调用失败，切换兜底模型: primary={self.model_name}, "
                    f"fallback={self.fallback_model_name}, error={primary_error}"
                )
                fallback_payload = dict(payload)
                # 兜底模型也禁用 thinking
                if self._should_disable_chat_template_thinking_for(
                    self.fallback_api_base,
                    self.fallback_model_name,
                ):
                    fallback_payload["extra_body"] = {
                        "enable_thinking": False,
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                        }
                    }
                response = await self._create_completion(
                    self._fallback_client,
                    fallback_payload,
                    self.fallback_model_name,
                    self.fallback_timeout_seconds,
                )
                used_model = self.fallback_model_name
                message = self._extract_response_message(response, used_model)
                # 提取 reasoning_content
                reasoning_content = getattr(message, 'reasoning_content', None)
                if reasoning_content:
                    logger.debug(f"LLM reasoning ({used_model}): reasoning_chars={len(str(reasoning_content))}")
                actual_content = getattr(message, "content", None)
                tool_calls = getattr(message, "tool_calls", None)

        # 空内容检查
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls and not str(actual_content or "").strip():
            if reasoning_content:
                logger.warning(f"LLM content 为空但 reasoning 存在 ({used_model}): reasoning长度={len(str(reasoning_content))}")
                raise RuntimeError(f"LLM返回空内容(推理占用): model={used_model}")
            raise RuntimeError(f"LLM返回空内容: model={used_model}")

        # 5. 记录 token 使用情况
        usage = getattr(response, "usage", None)
        if usage:
            logger.info(f"Token使用: model={used_model}, total={usage.total_tokens}, "
                        f"prompt={usage.prompt_tokens}, "
                        f"completion={usage.completion_tokens}, "
                        f"reasoning_chars={len(str(reasoning_content)) if reasoning_content else 0}, "
                        f"content_chars={len(str(actual_content)) if actual_content else 0}")

        # 6. 调试日志：输出 LLM 的响应
        if tool_calls:
            tool_names = [self._tool_call_name_for_log(tc) for tc in tool_calls]
            logger.info(f"LLM 决定调用工具: {tool_names}, model={used_model}")
        else:
            logger.debug(f"LLM 直接回复: model={used_model}, content_chars={len(str(actual_content or ''))}")

        return LLMResponse(
            content=actual_content,
            tool_calls=tool_calls,
            raw_response=response,
            reasoning_content=reasoning_content,
        )
