import asyncio
import json
from types import SimpleNamespace

from Agent.CustomerAgent.custom import tool_executor
from Agent.CustomerAgent.custom.tool_executor import ToolExecutor


def test_tool_executor_returns_error_for_malformed_tool_call():
    async def run():
        executor = ToolExecutor()

        results = await executor.execute_parallel([SimpleNamespace(id="call-1")], {})

        assert len(results) == 1
        assert results[0].tool_call_id == "call-1"
        assert "工具调用格式错误" in results[0].content

    asyncio.run(run())


def test_tool_executor_generates_id_for_missing_tool_call_id():
    async def run():
        executor = ToolExecutor()
        tool_call = SimpleNamespace(
            function=SimpleNamespace(name="missing_tool", arguments="{}")
        )

        results = await executor.execute_parallel([tool_call], {})

        assert len(results) == 1
        assert results[0].tool_call_id == "tool_call_0"
        assert "工具不存在" in results[0].content

    asyncio.run(run())


def test_to_assistant_tool_call_preserves_dict_arguments_as_json_string():
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="search_knowledge",
            arguments={"query": "续航", "token": "not-a-secret-field"},
        ),
    )

    message_tool_call = ToolExecutor.to_assistant_tool_call(tool_call, 0)

    assert isinstance(message_tool_call["function"]["arguments"], str)
    assert json.loads(message_tool_call["function"]["arguments"]) == {
        "query": "续航",
        "token": "not-a-secret-field",
    }


def test_tool_executor_masks_sensitive_values_in_executor_exception(monkeypatch):
    messages = []

    class FakeLogger:
        def debug(self, *_args, **_kwargs):
            pass

        def error(self, message):
            messages.append(str(message))

    async def run():
        executor = ToolExecutor()
        loop = asyncio.get_running_loop()

        def fake_run_in_executor(*_args, **_kwargs):
            future = loop.create_future()
            future.set_exception(RuntimeError("token=secret-token"))
            return future

        monkeypatch.setattr(tool_executor, "logger", FakeLogger())
        monkeypatch.setattr(loop, "run_in_executor", fake_run_in_executor)
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="some_tool", arguments="{}"),
        )

        results = await executor.execute_parallel([tool_call], {})

        assert len(results) == 1
        assert "secret-token" not in results[0].content
        assert "token=***" in results[0].content
        assert "secret-token" not in "\n".join(messages)

    asyncio.run(run())


def test_tool_executor_uses_running_loop_dedicated_executor_and_dependency_copy(monkeypatch):
    captured = []

    async def run():
        executor = ToolExecutor()
        loop = asyncio.get_running_loop()
        dependencies = {"shared": {"value": 1}}

        def fake_run_in_executor(executor_arg, func, name, arguments, deps):
            captured.append((executor_arg, func, name, arguments, deps))
            deps["shared"]["value"] = 2
            future = loop.create_future()
            future.set_result("ok")
            return future

        monkeypatch.setattr(loop, "run_in_executor", fake_run_in_executor)
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="some_tool", arguments="{}"),
        )

        results = await executor.execute_parallel([tool_call], dependencies)

        assert results[0].content == "ok"
        assert dependencies == {"shared": {"value": 1}}

    asyncio.run(run())

    assert captured
    assert captured[0][0] is tool_executor._TOOL_EXECUTOR
