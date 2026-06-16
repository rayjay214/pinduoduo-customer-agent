import asyncio
from types import SimpleNamespace

from Agent.CustomerAgent.custom.agent_config import AgentConfig
from Agent.CustomerAgent.custom import customer_agent as customer_agent_module
from Agent.CustomerAgent.custom.customer_agent import CustomerAgent
from Agent.CustomerAgent.custom.llm_client import LLMResponse
from Agent.CustomerAgent.custom.tool_executor import ToolResult


def _tool_call(tool_call_id="call-1", name="search_knowledge", arguments="{}"):
    return SimpleNamespace(
        id=tool_call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeLLMClient:
    def __init__(self):
        self.calls = []

    async def chat(self, messages, tool_choice="auto"):
        self.calls.append((tool_choice, [dict(item) for item in messages]))
        if len(self.calls) == 1:
            return LLMResponse(
                content="",
                tool_calls=[_tool_call()],
                raw_response=None,
            )
        return LLMResponse(
            content="最终回复",
            tool_calls=None,
            raw_response=None,
        )


class FakeToolExecutor:
    async def execute_parallel(self, tool_calls, dependencies):
        return [ToolResult(tool_calls[0].id, "工具结果")]


class TransferSuccessToolExecutor:
    async def execute_parallel(self, tool_calls, dependencies):
        return [ToolResult(tool_calls[0].id, "会话转接成功")]


class FailingAfterToolLLMClient:
    def __init__(self):
        self.calls = 0

    async def chat(self, messages, tool_choice="auto"):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="中间回复",
                tool_calls=[_tool_call()],
                raw_response=None,
            )
        raise RuntimeError("temporary llm failure")


def test_max_loop_executes_tool_result_before_final_reply():
    agent = CustomerAgent()
    agent._config = AgentConfig(max_loops=1)
    agent._llm_client = FakeLLMClient()
    agent._tool_executor = FakeToolExecutor()
    messages = [{"role": "user", "content": "查一下"}]

    reply = asyncio.run(agent._run_agent_loop(messages, {}))

    assert reply == "最终回复"
    assert agent._llm_client.calls[1][0] == "none"
    second_call_messages = agent._llm_client.calls[1][1]
    assistant_index = next(i for i, msg in enumerate(second_call_messages) if msg["role"] == "assistant")
    tool_index = next(i for i, msg in enumerate(second_call_messages) if msg["role"] == "tool")
    final_user_index = len(second_call_messages) - 1
    assert assistant_index < tool_index < final_user_index
    assert second_call_messages[tool_index]["tool_call_id"] == "call-1"


def test_loop_failure_after_tool_skips_malformed_messages_when_falling_back():
    agent = CustomerAgent()
    agent._config = AgentConfig(max_loops=2)
    agent._llm_client = FailingAfterToolLLMClient()
    agent._tool_executor = FakeToolExecutor()
    messages = ["bad", {"role": "user", "content": "查一下"}]

    reply = asyncio.run(agent._run_agent_loop(messages, {}))

    assert reply == "中间回复"


def test_agent_loop_returns_transfer_result_without_second_llm_call():
    agent = CustomerAgent()
    agent._config = AgentConfig(max_loops=2)
    agent._llm_client = FakeLLMClient()
    agent._tool_executor = TransferSuccessToolExecutor()
    messages = [{"role": "user", "content": "查一下"}]

    reply = asyncio.run(agent._run_agent_loop(messages, {}))

    assert reply == "亲，已为您转接人工处理，请稍等。"
    assert len(agent._llm_client.calls) == 1


def test_agent_loop_final_fallback_skips_malformed_last_message():
    agent = CustomerAgent()
    agent._config = AgentConfig(max_loops=0)
    messages = [{"role": "assistant", "content": "可用兜底"}, "bad-last"]

    reply = asyncio.run(agent._run_agent_loop(messages, {}))

    assert reply == "可用兜底"


def test_transfer_to_human_masks_sensitive_tool_result_in_logs(monkeypatch):
    messages = []

    class FakeLogger:
        def info(self, message):
            messages.append(str(message))

        def warning(self, message):
            messages.append(str(message))

        def error(self, message):
            messages.append(str(message))

        def debug(self, message):
            messages.append(str(message))

        def exception(self, message):
            messages.append(str(message))

    def fake_execute_tool(*_args, **_kwargs):
        return "转接失败 token=secret-token"

    monkeypatch.setattr(customer_agent_module, "logger", FakeLogger())
    monkeypatch.setattr(customer_agent_module, "execute_tool", fake_execute_tool)

    agent = CustomerAgent()

    result = asyncio.run(agent._transfer_to_human({}, "session-1", "test"))

    assert result == "亲，转人工暂时没成功，您先把问题发我，我这边继续帮您看。"
    assert "secret-token" not in "\n".join(messages)


def test_run_agent_loop_masks_sensitive_llm_exception_logs(monkeypatch):
    messages = []

    class FakeLogger:
        def info(self, message):
            messages.append(str(message))

        def warning(self, message):
            messages.append(str(message))

        def error(self, message):
            messages.append(str(message))

        def debug(self, message):
            messages.append(str(message))

        def exception(self, message):
            messages.append(str(message))

    class FakeLLM:
        async def chat(self, messages, tool_choice="auto"):
            raise RuntimeError("token=secret-token")

    monkeypatch.setattr(customer_agent_module, "logger", FakeLogger())

    agent = CustomerAgent()
    agent._config = AgentConfig(max_loops=1)
    agent._llm_client = FakeLLM()
    agent._tool_executor = FakeToolExecutor()

    result = asyncio.run(agent._run_agent_loop([{"role": "user", "content": "查一下"}], {}))

    assert result == "亲，客服正在为您处理，请稍等片刻哦～"
    assert "secret-token" not in "\n".join(messages)


def test_run_agent_loop_masks_sensitive_final_reply_exception_logs(monkeypatch):
    messages = []

    class FakeLogger:
        def info(self, message):
            messages.append(str(message))

        def warning(self, message):
            messages.append(str(message))

        def error(self, message):
            messages.append(str(message))

        def debug(self, message):
            messages.append(str(message))

        def exception(self, message):
            messages.append(str(message))

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def chat(self, messages, tool_choice="auto"):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    content="",
                    tool_calls=[_tool_call()],
                    raw_response=None,
                )
            raise RuntimeError("api_key=secret-token")

    monkeypatch.setattr(customer_agent_module, "logger", FakeLogger())

    agent = CustomerAgent()
    agent._config = AgentConfig(max_loops=1)
    agent._llm_client = FakeLLM()
    agent._tool_executor = FakeToolExecutor()

    result = asyncio.run(agent._run_agent_loop([{"role": "user", "content": "查一下"}], {}))

    assert result == "工具结果"
    assert "secret-token" not in "\n".join(messages)


def test_dedup_reply_does_not_log_reply_content(monkeypatch):
    messages = []

    class FakeLogger:
        def info(self, _message):
            pass

        def warning(self, message):
            messages.append(str(message))

    monkeypatch.setattr(customer_agent_module, "logger", FakeLogger())

    agent = CustomerAgent()
    content = "重复回复 token=secret-token"
    result = asyncio.run(
        agent._dedup_reply(
            content,
            messages=[],
            history=[{"role": "assistant", "content": content}],
            session_id="session-1",
        )
    )

    assert result == content
    joined = "\n".join(messages)
    assert "secret-token" not in joined
    assert "token=***" not in joined
    assert "content_chars=" in joined
