from Channel.pinduoduo.utils.API import send_message
from Channel.pinduoduo.utils.API.send_message import DEFAULT_TRANSFER_REMARK, SendMessage


class DummySendMessage(SendMessage):
    def __init__(self):
        self.logger = _DummyLogger()

    def generate_request_id(self):
        return 1

    def post(self, url, json_data=None, **kwargs):
        self.sent_url = url
        self.sent_json = json_data
        return {"success": True}


class _DummyLogger:
    def debug(self, *args, **kwargs):
        pass


def _sent_remark(sender: DummySendMessage) -> str:
    return sender.sent_json["data"]["conversation"]["remark"]


def test_move_conversation_uses_default_transfer_remark(monkeypatch):
    monkeypatch.setattr(send_message, "get_config", lambda key, default=None: default)
    sender = DummySendMessage()

    sender.move_conversation("buyer-1", "cs-1")

    assert _sent_remark(sender) == DEFAULT_TRANSFER_REMARK
    assert _sent_remark(sender) != "无原因直接转移"


def test_move_conversation_uses_configured_transfer_remark(monkeypatch):
    monkeypatch.setattr(
        send_message,
        "get_config",
        lambda key, default=None: "售后升级处理"
        if key == "pinduoduo.transfer.default_remark"
        else default,
    )
    sender = DummySendMessage()

    sender.move_conversation("buyer-1", "cs-1")

    assert _sent_remark(sender) == "售后升级处理"


def test_move_conversation_explicit_remark_overrides_config(monkeypatch):
    monkeypatch.setattr(
        send_message,
        "get_config",
        lambda key, default=None: "售后升级处理"
        if key == "pinduoduo.transfer.default_remark"
        else default,
    )
    sender = DummySendMessage()

    sender.move_conversation("buyer-1", "cs-1", remark="客户要求人工")

    assert _sent_remark(sender) == "客户要求人工"
