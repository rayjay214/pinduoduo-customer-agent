"""
工具执行器模块

负责并行执行 Agent 工具调用。
"""
from __future__ import annotations

import asyncio
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from core.base_service import _sanitize_for_log
from utils.logger_loguru import get_logger
from Agent.CustomerAgent.custom.tool_decorator import execute_tool

logger = get_logger("ToolExecutor")
_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="agent-tool")


def _sanitize_executor_error(value: Any) -> Any:
    return _sanitize_for_log(value)


class ToolResult:
    """工具执行结果"""

    def __init__(self, tool_call_id: str, content: str):
        self.tool_call_id = tool_call_id
        self.content = content

    def to_dict(self) -> Dict[str, Any]:
        """转换为 LLM 消息格式的字典"""
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }


class ToolExecutor:
    """工具执行器"""

    @staticmethod
    def _read_field(value: Any, field_name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(field_name, default)
        return getattr(value, field_name, default)

    @classmethod
    def get_tool_call_id(cls, tool_call: Any, index: int) -> str:
        tool_call_id = cls._read_field(tool_call, "id")
        text = str(tool_call_id or "").strip()
        return text or f"tool_call_{index}"

    @classmethod
    def get_tool_call_function(cls, tool_call: Any) -> tuple[Any, Any]:
        function = cls._read_field(tool_call, "function")
        name = cls._read_field(function, "name")
        arguments = cls._read_field(function, "arguments", "{}")
        return name, arguments if arguments is not None else "{}"

    @classmethod
    def to_assistant_tool_call(cls, tool_call: Any, index: int) -> Dict[str, Any]:
        name, arguments = cls.get_tool_call_function(tool_call)
        name_text = str(name or "").strip() or "unknown_tool"
        if isinstance(arguments, str):
            arguments_text = arguments
        else:
            try:
                arguments_text = json.dumps(arguments, ensure_ascii=False)
            except Exception:
                arguments_text = "{}"
        return {
            "type": "function",
            "id": cls.get_tool_call_id(tool_call, index),
            "function": {
                "name": name_text,
                "arguments": arguments_text,
            },
        }

    async def execute_parallel(
        self,
        tool_calls: List[Any],
        dependencies: Dict[str, Any],
    ) -> List[ToolResult]:
        """
        并行执行多个工具调用

        Args:
            tool_calls: 工具调用列表
            dependencies: 依赖字典

        Returns:
            工具执行结果列表（按原始顺序）
        """
        if not tool_calls:
            return []

        logger.debug(f"开始并行执行 {len(tool_calls)} 个工具")
        loop = asyncio.get_running_loop()

        # 构建任务列表。模型或供应商返回畸形 tool_call 时，转成工具错误结果，
        # 保持 OpenAI 工具调用协议的 assistant/tool 消息配对。
        results: List[ToolResult | None] = [None] * len(tool_calls)
        tasks: List[Any] = []
        for index, tc in enumerate(tool_calls):
            tool_call_id = self.get_tool_call_id(tc, index)
            name, arguments = self.get_tool_call_function(tc)
            name_text = str(name or "").strip()
            if not name_text:
                results[index] = ToolResult(tool_call_id, "[工具调用格式错误: 缺少 function.name]")
                continue

            try:
                isolated_dependencies = copy.deepcopy(dependencies)
            except Exception:
                isolated_dependencies = dict(dependencies)

            task = loop.run_in_executor(
                _TOOL_EXECUTOR,
                execute_tool,
                name_text,
                arguments,
                isolated_dependencies,
            )
            tasks.append((index, tool_call_id, task))

        # 等待所有任务完成
        for index, tool_call_id, task in tasks:
            try:
                content = await task
                results[index] = ToolResult(tool_call_id, content)
                logger.debug(f"工具执行完成: {tool_call_id}")
            except Exception as e:
                safe_error = _sanitize_executor_error(str(e))
                logger.error(f"工具执行失败: {tool_call_id}, error: {safe_error}")
                results[index] = ToolResult(tool_call_id, f"[工具执行错误: {safe_error}]")

        return [result for result in results if result is not None]

    def results_to_messages(self, results: List[ToolResult]) -> List[Dict[str, str]]:
        """
        将工具执行结果转换为 LLM 消息格式

        Args:
            results: 工具执行结果列表

        Returns:
            LLM 消息格式的结果列表
        """
        return [result.to_dict() for result in results]
