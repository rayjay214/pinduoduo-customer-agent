from Agent.CustomerAgent.tools import move_conversation as module
from Agent.CustomerAgent.custom.tool_decorator import get_tools_for_llm


class CaptureLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(str(message))

    def warning(self, message):
        self.messages.append(str(message))

    def error(self, message):
        self.messages.append(str(message))


def test_transfer_conversation_handles_non_dict_transfer_result(monkeypatch):
    class FakeSender:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def getAssignCsList(self):
            return {"cs-2": {"username": "客服2"}}

        def move_conversation(self, recipient_uid, cs_uid):
            return "temporary platform error"

    class FakeDbManager:
        def get_transfer_target(self, *_args):
            return None

    monkeypatch.setattr(module, "is_night_mode", lambda: False)
    monkeypatch.setattr(module, "SendMessage", FakeSender)
    monkeypatch.setattr(module, "db_manager", FakeDbManager())

    result = module.transfer_conversation(
        module.TransferConversationParams(
            shop_id="shop-1",
            user_id="seller-1",
            recipient_uid="buyer-1",
        )
    )

    assert result == "会话转接失败"


def test_transfer_conversation_treats_string_false_success_as_failure(monkeypatch):
    class FakeSender:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def getAssignCsList(self):
            return {"cs-2": {"username": "客服2"}}

        def move_conversation(self, recipient_uid, cs_uid):
            return {"success": "false", "error_msg": "platform rejected"}

    class FakeDbManager:
        def get_transfer_target(self, *_args):
            return None

    monkeypatch.setattr(module, "is_night_mode", lambda: False)
    monkeypatch.setattr(module, "SendMessage", FakeSender)
    monkeypatch.setattr(module, "db_manager", FakeDbManager())

    result = module.transfer_conversation(
        module.TransferConversationParams(
            shop_id="shop-1",
            user_id="seller-1",
            recipient_uid="buyer-1",
        )
    )

    assert result == "会话转接失败"


def test_transfer_conversation_masks_sensitive_transfer_result_in_logs(monkeypatch):
    class FakeSender:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def getAssignCsList(self):
            return {"cs-2": {"username": "客服2"}}

        def move_conversation(self, recipient_uid, cs_uid):
            return {"success": False, "error_msg": "platform rejected token=secret-token"}

    class FakeDbManager:
        def get_transfer_target(self, *_args):
            return None

    logger = CaptureLogger()
    monkeypatch.setattr(module, "is_night_mode", lambda: False)
    monkeypatch.setattr(module, "SendMessage", FakeSender)
    monkeypatch.setattr(module, "db_manager", FakeDbManager())
    monkeypatch.setattr(module, "logger", logger)

    result = module.transfer_conversation(
        module.TransferConversationParams(
            shop_id="shop-1",
            user_id="seller-1",
            recipient_uid="buyer-1",
        )
    )

    assert result == "会话转接失败"
    assert "secret-token" not in "\n".join(logger.messages)


def test_transfer_conversation_does_not_log_raw_transfer_result_values(monkeypatch):
    class FakeSender:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def getAssignCsList(self):
            return {"cs-2": {"username": "客服2"}}

        def move_conversation(self, recipient_uid, cs_uid):
            return {"success": False, "error_msg": "客户手机号13800138000，地址测试小区1号楼"}

    class FakeDbManager:
        def get_transfer_target(self, *_args):
            return None

    logger = CaptureLogger()
    monkeypatch.setattr(module, "is_night_mode", lambda: False)
    monkeypatch.setattr(module, "SendMessage", FakeSender)
    monkeypatch.setattr(module, "db_manager", FakeDbManager())
    monkeypatch.setattr(module, "logger", logger)

    result = module.transfer_conversation(
        module.TransferConversationParams(
            shop_id="shop-1",
            user_id="seller-1",
            recipient_uid="buyer-1",
        )
    )

    joined = "\n".join(logger.messages)
    assert result == "会话转接失败"
    assert "13800138000" not in joined
    assert "测试小区" not in joined
    assert "result_type=dict" in joined


def test_transfer_conversation_masks_sensitive_exception_in_tool_output_and_logs(monkeypatch):
    class FakeSender:
        def __init__(self, shop_id, user_id):
            raise RuntimeError("token=secret-token")

    logger = CaptureLogger()
    monkeypatch.setattr(module, "is_night_mode", lambda: False)
    monkeypatch.setattr(module, "SendMessage", FakeSender)
    monkeypatch.setattr(module, "logger", logger)

    result = module.transfer_conversation(
        module.TransferConversationParams(
            shop_id="shop-1",
            user_id="seller-1",
            recipient_uid="buyer-1",
        )
    )

    assert "secret-token" not in result
    assert "token=***" in result
    assert "secret-token" not in "\n".join(logger.messages)


def test_transfer_conversation_tool_description_uses_configured_escalation_examples(monkeypatch):
    def fake_get_config(key, default=None):
        if key == "agent.transfer_escalation_examples":
            return ["键盘失灵", "屏幕花屏"]
        return default

    monkeypatch.setattr(module, "get_config", fake_get_config)

    tools = get_tools_for_llm()
    description = next(
        tool["function"]["description"]
        for tool in tools
        if tool["function"]["name"] == "transfer_conversation"
    )

    assert "键盘失灵、屏幕花屏" in description
    assert "噪音大" not in description
    assert "充不进电" not in description
