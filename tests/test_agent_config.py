from Agent.CustomerAgent.custom.agent_config import (
    AgentConfig,
    DEFAULT_COMPRESS_RATIO,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOKEN_WINDOW,
    DEFAULT_TOOL_CALL_MAX_TOKENS,
)
from Agent.CustomerAgent.custom import agent_config
from Agent.CustomerAgent.custom.llm_client import LLMClient


def test_default_llm_token_budget_is_not_tiny():
    assert DEFAULT_MAX_TOKENS >= 512
    assert DEFAULT_TOOL_CALL_MAX_TOKENS >= DEFAULT_MAX_TOKENS


def test_llm_client_tool_budget_is_at_least_reply_budget():
    client = LLMClient(
        api_key="key",
        api_base="",
        model_name="model",
        temperature=0.1,
        max_tokens=512,
        tool_call_max_tokens=128,
    )

    assert client.max_tokens == 512
    assert client.tool_call_max_tokens == 512


def test_llm_disable_thinking_uses_configurable_patterns():
    default_client = LLMClient(
        api_key="key",
        api_base="https://token-plan-cn.xiaomimimo.com/v1",
        model_name="mimo-v2.5",
        temperature=0.1,
        max_tokens=512,
    )
    custom_client = LLMClient(
        api_key="key",
        api_base="https://token-plan-cn.xiaomimimo.com/v1",
        model_name="mimo-v2.5",
        temperature=0.1,
        max_tokens=512,
        disable_thinking_api_base_patterns=[],
        disable_thinking_model_prefixes=[],
    )
    disabled_client = LLMClient(
        api_key="key",
        api_base="https://token-plan-cn.xiaomimimo.com/v1",
        model_name="mimo-v2.5",
        temperature=0.1,
        max_tokens=512,
        disable_thinking=False,
    )

    assert default_client._should_disable_chat_template_thinking() is True
    assert custom_client._should_disable_chat_template_thinking() is False
    assert disabled_client._should_disable_chat_template_thinking() is False


def test_agent_config_invalid_numeric_values_fall_back_to_defaults(monkeypatch):
    def fake_get_config(key, default=None):
        if key == "agent.token_window":
            return "not-an-int"
        return default

    monkeypatch.setattr(agent_config, "get_config", fake_get_config)

    config = AgentConfig()

    assert config.token_window == DEFAULT_TOKEN_WINDOW


def test_agent_config_non_finite_numeric_values_fall_back_to_defaults(monkeypatch):
    def fake_get_config(key, default=None):
        values = {
            "agent.token_window": "inf",
            "agent.compress_ratio": "nan",
        }
        return values.get(key, default)

    monkeypatch.setattr(agent_config, "get_config", fake_get_config)

    config = AgentConfig()

    assert config.token_window == DEFAULT_TOKEN_WINDOW
    assert config.compress_ratio == DEFAULT_COMPRESS_RATIO


def test_agent_config_string_false_values_are_false(monkeypatch):
    def fake_get_config(key, default=None):
        values = {
            "llm.disable_thinking": "false",
            "llm.fallback.enabled": "0",
        }
        return values.get(key, default)

    monkeypatch.setattr(agent_config, "get_config", fake_get_config)

    config = AgentConfig()

    assert config.disable_thinking is False
    assert config.fallback_enabled is False


def test_agent_config_disable_thinking_patterns_can_be_disabled(monkeypatch):
    def fake_get_config(key, default=None):
        values = {
            "llm.disable_thinking_api_base_patterns": [],
            "llm.disable_thinking_model_prefixes": [],
        }
        return values.get(key, default)

    monkeypatch.setattr(agent_config, "get_config", fake_get_config)

    config = AgentConfig()

    assert config.disable_thinking_api_base_patterns == ()
    assert config.disable_thinking_model_prefixes == ()


def test_llm_client_string_false_disables_boolean_flags():
    client = LLMClient(
        api_key="key",
        api_base="https://token-plan-cn.xiaomimimo.com/v1",
        model_name="mimo-v2.5",
        temperature=0.1,
        max_tokens=512,
        fallback_enabled="false",
        fallback_api_key="fallback-key",
        fallback_model_name="fallback-model",
        disable_thinking="false",
    )

    assert client.fallback_enabled is False
    assert client._should_disable_chat_template_thinking() is False


def test_llm_client_invalid_numeric_values_fall_back_safely():
    client = LLMClient(
        api_key="key",
        api_base="",
        model_name="model",
        temperature=0.1,
        max_tokens="bad",
        tool_call_max_tokens="also-bad",
        request_timeout_seconds="nope",
        max_concurrent_requests="many",
        fallback_timeout_seconds="later",
    )

    assert client.max_tokens == 1
    assert client.tool_call_max_tokens == 1
    assert client.request_timeout_seconds == 20.0
    assert client.max_concurrent_requests == 2
    assert client.fallback_timeout_seconds == 20.0
