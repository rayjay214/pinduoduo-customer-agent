from typing import Optional

from pydantic import BaseModel, Field

from Agent.CustomerAgent.custom import tool_decorator
from Agent.CustomerAgent.custom.tool_decorator import agent_tool, execute_tool, get_tools_for_llm


class _DemoParams(BaseModel):
    shop_id: Optional[str] = Field(default=None)
    user_id: Optional[str] = Field(default=None)
    recipient_uid: Optional[str] = Field(default=None)


def test_tool_schema_is_not_changed_by_legacy_aftersale_prefix(monkeypatch):
    registry = {}
    monkeypatch.setattr(tool_decorator, "TOOL_REGISTRY", registry)

    @agent_tool(
        name="aftersale_demo",
        description="demo",
        param_model=_DemoParams,
    )
    def _demo(params: _DemoParams):
        return "ok"

    tool = get_tools_for_llm()[0]["function"]
    properties = tool["parameters"]["properties"]

    assert {"shop_id", "user_id", "recipient_uid"}.issubset(properties)


def test_tool_description_can_be_resolved_dynamically(monkeypatch):
    registry = {}
    monkeypatch.setattr(tool_decorator, "TOOL_REGISTRY", registry)

    description = {"value": "first"}

    @agent_tool(
        name="dynamic_description_demo",
        description=lambda: description["value"],
        param_model=_DemoParams,
    )
    def _demo(params: _DemoParams):
        return "ok"

    description["value"] = "second"

    tool = get_tools_for_llm()[0]["function"]

    assert tool["description"] == "second"


def test_execute_tool_fills_critical_ids_from_dependencies(monkeypatch):
    registry = {}
    monkeypatch.setattr(tool_decorator, "TOOL_REGISTRY", registry)

    @agent_tool(
        name="demo_transfer",
        description="demo",
        param_model=_DemoParams,
    )
    def _demo(params: _DemoParams):
        return f"{params.shop_id}:{params.user_id}:{params.recipient_uid}"

    result = execute_tool(
        "demo_transfer",
        '{"shop_id": null, "user_id": "", "recipient_uid": "none"}',
        {
            "shop_id": "shop-1",
            "user_id": "user-1",
            "recipient_uid": "buyer-1",
        },
    )

    assert result == "shop-1:user-1:buyer-1"


def test_execute_tool_returns_parse_error_for_invalid_json(monkeypatch):
    registry = {}
    monkeypatch.setattr(tool_decorator, "TOOL_REGISTRY", registry)

    @agent_tool(
        name="aftersale_demo",
        description="demo",
        param_model=_DemoParams,
    )
    def _demo(params: _DemoParams):
        return "ok"

    result = execute_tool("aftersale_demo", '{"shop_id":', {})

    assert result.startswith("[工具参数解析错误:")


def test_execute_tool_rejects_non_object_json_arguments(monkeypatch):
    registry = {}
    monkeypatch.setattr(tool_decorator, "TOOL_REGISTRY", registry)

    calls = []

    @agent_tool(
        name="demo_non_object_args",
        description="demo",
        param_model=_DemoParams,
    )
    def _demo(params: _DemoParams):
        calls.append(params)
        return "should not run"

    result = execute_tool(
        "demo_non_object_args",
        '["not", "an", "object"]',
        {"shop_id": "shop-1"},
    )

    assert result.startswith("[工具参数解析错误:")
    assert calls == []


def test_execute_tool_treats_non_dict_dependencies_as_empty_context(monkeypatch):
    registry = {}
    monkeypatch.setattr(tool_decorator, "TOOL_REGISTRY", registry)

    @agent_tool(
        name="demo_optional_context",
        description="demo",
        param_model=_DemoParams,
    )
    def _demo(params: _DemoParams):
        return f"{params.shop_id}:{params.user_id}:{params.recipient_uid}"

    result = execute_tool("demo_optional_context", "{}", object())

    assert result == "None:None:None"


def test_execute_tool_masks_sensitive_arguments_in_parse_error_logs(monkeypatch):
    registry = {}
    monkeypatch.setattr(tool_decorator, "TOOL_REGISTRY", registry)

    @agent_tool(
        name="demo_parse_error",
        description="demo",
        param_model=_DemoParams,
    )
    def _demo(params: _DemoParams):
        return "ok"

    messages = []

    class FakeLogger:
        def error(self, message):
            messages.append(str(message))

        def debug(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(tool_decorator, "logger", FakeLogger())

    result = execute_tool("demo_parse_error", '{"shop_id": "shop-1", "token": "secret-token",', {})

    assert result.startswith("[工具参数解析错误:")
    assert "secret-token" not in "\n".join(messages)
    assert "argument_chars=" in "\n".join(messages)


def test_execute_tool_does_not_log_raw_arguments_in_parse_error(monkeypatch):
    registry = {}
    monkeypatch.setattr(tool_decorator, "TOOL_REGISTRY", registry)

    @agent_tool(
        name="demo_parse_error_privacy",
        description="demo",
        param_model=_DemoParams,
    )
    def _demo(params: _DemoParams):
        return "ok"

    messages = []

    class FakeLogger:
        def error(self, message):
            messages.append(str(message))

        def debug(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(tool_decorator, "logger", FakeLogger())

    result = execute_tool(
        "demo_parse_error_privacy",
        '{"customer_message": "我的手机号13800138000，地址测试小区1号楼",',
        {},
    )

    joined = "\n".join(messages)
    assert result.startswith("[工具参数解析错误:")
    assert "13800138000" not in joined
    assert "测试小区" not in joined
    assert "argument_chars=" in joined


def test_execute_tool_masks_sensitive_values_in_execution_error_logs(monkeypatch):
    registry = {}
    monkeypatch.setattr(tool_decorator, "TOOL_REGISTRY", registry)

    @agent_tool(
        name="demo_exec_error",
        description="demo",
        param_model=_DemoParams,
    )
    def _demo(params: _DemoParams):
        raise RuntimeError("token=secret-token")

    messages = []

    class FakeLogger:
        def error(self, message):
            messages.append(str(message))

        def debug(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(tool_decorator, "logger", FakeLogger())

    result = execute_tool("demo_exec_error", "{}", {})

    assert "secret-token" not in result
    assert "token=***" in result
    assert "secret-token" not in "\n".join(messages)
