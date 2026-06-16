import asyncio
from types import SimpleNamespace

from Agent.CustomerAgent.custom.llm_client import LLMClient


def _client(fallback_enabled=False):
    client = LLMClient(
        api_key="primary-key",
        api_base="http://primary",
        model_name="primary-model",
        temperature=0.1,
        max_tokens=16,
        fallback_api_key="fallback-key",
        fallback_api_base="http://fallback",
        fallback_model_name="fallback-model",
        fallback_enabled=fallback_enabled,
    )
    client._client = object()
    if fallback_enabled:
        client._fallback_client = object()
    return client


def test_global_semaphore_and_lock_are_recreated_per_event_loop():
    async def get_ids(limit):
        semaphore = await LLMClient._get_global_semaphore(limit)
        import Agent.CustomerAgent.custom.llm_client as module

        return id(semaphore), id(module._LLM_SEMAPHORE_LOCK)

    first = asyncio.run(get_ids(2))
    second = asyncio.run(get_ids(2))

    assert first != second


def test_global_semaphore_is_reused_within_same_event_loop_and_limit():
    async def run():
        first = await LLMClient._get_global_semaphore(2)
        second = await LLMClient._get_global_semaphore(2)
        third = await LLMClient._get_global_semaphore(3)

        assert first is second
        assert third is not first

    asyncio.run(run())


def test_global_semaphore_cache_keeps_loop_state_isolated():
    async def get_ids(limit):
        first = await LLMClient._get_global_semaphore(limit)
        second = await LLMClient._get_global_semaphore(limit)
        return id(first), id(second)

    first_loop_ids = asyncio.run(get_ids(2))
    second_loop_ids = asyncio.run(get_ids(2))

    assert first_loop_ids[0] == first_loop_ids[1]
    assert second_loop_ids[0] == second_loop_ids[1]
    assert first_loop_ids[0] != second_loop_ids[0]


def test_chat_falls_back_when_primary_response_has_no_choices():
    async def run():
        client = _client(fallback_enabled=True)
        calls = []

        async def fake_create_completion(_client_obj, _payload, model_name, _timeout):
            calls.append(model_name)
            if model_name == "primary-model":
                return SimpleNamespace(choices=[], usage=None)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="兜底回复",
                            tool_calls=None,
                            reasoning_content=None,
                        )
                    )
                ],
                usage=None,
            )

        client._create_completion = fake_create_completion

        response = await client.chat([{"role": "user", "content": "你好"}], tool_choice="none")

        assert calls == ["primary-model", "fallback-model"]
        assert response.content == "兜底回复"

    asyncio.run(run())


def test_chat_raises_clear_error_when_response_has_no_choices_without_fallback():
    async def run():
        client = _client(fallback_enabled=False)

        async def fake_create_completion(*_args, **_kwargs):
            return SimpleNamespace(choices=[], usage=None)

        client._create_completion = fake_create_completion

        try:
            await client.chat([{"role": "user", "content": "你好"}], tool_choice="none")
        except RuntimeError as exc:
            assert "missing choices" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")

    asyncio.run(run())


def test_chat_tolerates_malformed_tool_call_during_logging():
    async def run():
        client = _client(fallback_enabled=False)

        async def fake_create_completion(*_args, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="",
                            tool_calls=[SimpleNamespace(id="call-1")],
                            reasoning_content=None,
                        )
                    )
                ],
                usage=None,
            )

        client._create_completion = fake_create_completion

        response = await client.chat([{"role": "user", "content": "查一下"}])

        assert response.tool_calls[0].id == "call-1"

    asyncio.run(run())


def test_chat_tolerates_success_response_without_usage_field():
    async def run():
        client = _client(fallback_enabled=False)

        async def fake_create_completion(*_args, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="正常回复",
                            tool_calls=None,
                            reasoning_content=None,
                        )
                    )
                ],
            )

        client._create_completion = fake_create_completion

        response = await client.chat([{"role": "user", "content": "你好"}], tool_choice="none")

        assert response.content == "正常回复"

    asyncio.run(run())


def test_chat_tolerates_text_message_without_tool_calls_field():
    async def run():
        client = _client(fallback_enabled=False)

        async def fake_create_completion(*_args, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="正常回复",
                            reasoning_content=None,
                        )
                    )
                ],
                usage=None,
            )

        client._create_completion = fake_create_completion

        response = await client.chat([{"role": "user", "content": "你好"}], tool_choice="none")

        assert response.content == "正常回复"
        assert response.tool_calls is None

    asyncio.run(run())


def test_normalize_tools_skips_malformed_tool_definitions():
    tools = LLMClient._normalize_tools(
        [
            "bad",
            {"function": ["bad"]},
            {"function": {"parameters": ["bad"]}},
            {"function": {"name": "valid_tool", "parameters": {"type": "object"}}},
        ]
    )

    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "valid_tool",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_normalize_messages_skips_malformed_items():
    messages = LLMClient._normalize_messages(
        [
            "bad",
            {"role": "system", "content": "系统A"},
            {"role": "user", "content": "你好"},
            {"role": "system", "content": "系统B"},
        ]
    )

    assert messages == [
        {"role": "system", "content": "系统A\n\n系统B"},
        {"role": "user", "content": "你好"},
    ]


def test_chat_masks_request_validation_exception_logs(monkeypatch):
    async def run():
        client = _client(fallback_enabled=False)
        messages = [{"role": "user", "content": "你好"}]

        class FakeRequest:
            def __init__(self, **_kwargs):
                raise RuntimeError("token=secret-token")

        logs = []

        class FakeLogger:
            def debug(self, message):
                logs.append(str(message))

            def info(self, message):
                logs.append(str(message))

            def warning(self, message):
                logs.append(str(message))

            def error(self, message):
                logs.append(str(message))

        monkeypatch.setattr("Agent.CustomerAgent.custom.llm_client.ChatCompletionsRequest", FakeRequest)
        monkeypatch.setattr("Agent.CustomerAgent.custom.llm_client.logger", FakeLogger())

        try:
            await client.chat(messages, tool_choice="none")
        except RuntimeError as exc:
            assert "secret-token" not in str(exc)
        else:
            raise AssertionError("expected RuntimeError")

        joined = "\n".join(logs)
        assert "secret-token" not in joined
        assert "token=***" in joined

    asyncio.run(run())


def test_chat_does_not_log_raw_direct_reply_content(monkeypatch):
    async def run():
        client = _client(fallback_enabled=False)
        logs = []

        class FakeLogger:
            def debug(self, message):
                logs.append(str(message))

            def info(self, message):
                logs.append(str(message))

            def warning(self, message):
                logs.append(str(message))

            def error(self, message):
                logs.append(str(message))

        async def fake_create_completion(*_args, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="客户隐私 token=secret-token",
                            tool_calls=None,
                            reasoning_content=None,
                        )
                    )
                ],
                usage=None,
            )

        monkeypatch.setattr("Agent.CustomerAgent.custom.llm_client.logger", FakeLogger())
        client._create_completion = fake_create_completion

        response = await client.chat([{"role": "user", "content": "你好"}], tool_choice="none")

        assert response.content == "客户隐私 token=secret-token"
        joined = "\n".join(logs)
        assert "secret-token" not in joined
        assert "客户隐私" not in joined
        assert "content_chars=" in joined

    asyncio.run(run())


def test_chat_does_not_log_raw_reasoning_content(monkeypatch):
    async def run():
        client = _client(fallback_enabled=False)
        logs = []

        class FakeLogger:
            def debug(self, message):
                logs.append(str(message))

            def info(self, message):
                logs.append(str(message))

            def warning(self, message):
                logs.append(str(message))

            def error(self, message):
                logs.append(str(message))

        async def fake_create_completion(*_args, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="正常回复",
                            tool_calls=None,
                            reasoning_content="推理里复述 token=secret-token",
                        )
                    )
                ],
                usage=None,
            )

        monkeypatch.setattr("Agent.CustomerAgent.custom.llm_client.logger", FakeLogger())
        client._create_completion = fake_create_completion

        response = await client.chat([{"role": "user", "content": "你好"}], tool_choice="none")

        assert response.content == "正常回复"
        joined = "\n".join(logs)
        assert "secret-token" not in joined
        assert "推理里复述" not in joined
        assert "reasoning_chars=" in joined

    asyncio.run(run())
