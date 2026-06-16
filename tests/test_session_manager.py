import asyncio

from Agent.CustomerAgent.custom.customer_agent import CustomerAgent
from Agent.CustomerAgent.custom.llm_client import LLMResponse
from Agent.CustomerAgent.custom import session_manager as session_manager_module
from Agent.CustomerAgent.custom.session_manager import SessionManager, TokenEstimator


def _add_messages(manager: SessionManager, session_id: str, count: int) -> None:
    for index in range(count):
        manager.add_message(session_id, "user", f"message {index}")


def test_compress_history_keeps_messages_when_summary_fails(tmp_path):
    manager = SessionManager(
        db_path=str(tmp_path / "agent.db"),
        retain_count=2,
        token_window=100,
        compress_ratio=0.5,
    )
    session_id = "session-fail"
    _add_messages(manager, session_id, 5)

    def fail_summary(_messages):
        raise RuntimeError("summary failed")

    assert manager.compress_history(session_id, fail_summary) is False

    history = manager.get_history(session_id)
    assert [item["content"] for item in history] == [f"message {index}" for index in range(5)]


def test_compress_history_masks_summary_exception_logs(tmp_path, monkeypatch):
    manager = SessionManager(
        db_path=str(tmp_path / "agent.db"),
        retain_count=2,
        token_window=100,
        compress_ratio=0.5,
    )
    session_id = "session-secret-fail"
    _add_messages(manager, session_id, 5)
    messages = []

    class FakeLogger:
        def debug(self, *_args, **_kwargs):
            pass

        def info(self, *_args, **_kwargs):
            pass

        def warning(self, message):
            messages.append(str(message))

        def error(self, message):
            messages.append(str(message))

    def fail_summary(_messages):
        raise RuntimeError("api_key=secret-token")

    monkeypatch.setattr(session_manager_module, "logger", FakeLogger())

    assert manager.compress_history(session_id, fail_summary) is False

    joined = "\n".join(messages)
    assert "生成摘要失败" in joined
    assert "secret-token" not in joined
    assert "api_key=***" in joined


def test_compress_history_inserts_summary_before_retained_messages(tmp_path):
    manager = SessionManager(
        db_path=str(tmp_path / "agent.db"),
        retain_count=2,
        token_window=100,
        compress_ratio=0.5,
    )
    session_id = "session-success"
    _add_messages(manager, session_id, 5)

    assert manager.compress_history(session_id, lambda _messages: "summary text") is True

    history = manager.get_history(session_id)
    assert len(history) == 3
    assert history[0]["role"] == "system"
    assert "summary text" in history[0]["content"]
    assert [item["content"] for item in history[1:]] == ["message 3", "message 4"]


def test_customer_agent_compresses_the_requested_session(tmp_path):
    manager = SessionManager(
        db_path=str(tmp_path / "agent.db"),
        retain_count=2,
        token_window=100,
        compress_ratio=0.5,
    )
    session_id = "session-target"
    _add_messages(manager, session_id, 5)

    class FakeLLM:
        async def chat(self, messages, tool_choice=None):
            assert tool_choice == "none"
            return LLMResponse(content="agent summary", tool_calls=None, raw_response=None)

    agent = CustomerAgent(db_path=str(tmp_path / "agent.db"))
    agent._session_manager = manager
    agent._llm_client = FakeLLM()

    asyncio.run(agent._compress_with_llm(session_id, manager.get_history(session_id)))

    history = manager.get_history(session_id)
    assert len(history) == 3
    assert history[0]["role"] == "system"
    assert "agent summary" in history[0]["content"]
    assert [item["content"] for item in history[1:]] == ["message 3", "message 4"]
    assert manager.get_history(None) == []


def test_token_counter_skips_malformed_message_items():
    counter = TokenEstimator()

    total = counter.estimate_messages(["bad", {"role": "user", "content": "你好"}])

    assert total >= counter.estimate("你好")


def test_session_manager_reuses_global_engine_for_same_database(monkeypatch, tmp_path):
    db_path = str(tmp_path / "shared.db")

    class FakeDbManager:
        def __init__(self, engine):
            self.db_path = db_path
            self.engine = engine

    base_manager = SessionManager(db_path=db_path)
    monkeypatch.setattr("database.get_db_manager", lambda: FakeDbManager(base_manager.engine))

    manager = SessionManager(db_path=db_path)

    assert manager.engine is base_manager.engine
