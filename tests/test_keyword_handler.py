from Message.handlers import keyword_handler
from Message.handlers.keyword_handler import KeywordDetectionHandler
from bridge.context import ChannelType, Context, ContextType


class FakeDbManager:
    def get_all_keywords(self):
        return [
            {"keyword": "客服"},
            {"keyword": "人工"},
            {"keyword": "工单"},
            {"keyword": "催开发票"},
        ]


def _context(content):
    return Context.create_pinduoduo_context(
        content=content,
        msg_id="msg-1",
        from_uid="buyer-1",
        user_msg_type=ContextType.TEXT,
        shop_id="shop-1",
        user_id="seller-1",
        channel_type=ChannelType.PINDUODUO,
    )


def test_keyword_handler_can_handle_safe_transfer_phrase():
    handler = KeywordDetectionHandler()
    handler.keywords = {"转人工"}

    assert handler.can_handle(_context("麻烦帮我转人工")) is True


def test_keyword_handler_masks_sensitive_content_in_match_log():
    class FakeLogger:
        def __init__(self):
            self.messages = []

        def debug(self, message):
            self.messages.append(str(message))

        def info(self, message):
            self.messages.append(str(message))

        def warning(self, message):
            self.messages.append(str(message))

        def error(self, message):
            self.messages.append(str(message))

    handler = KeywordDetectionHandler()
    handler.logger = FakeLogger()
    handler.keywords = {"转人工"}

    assert handler.can_handle(_context("麻烦转人工 token=secret-token")) is True
    joined = "\n".join(handler.logger.messages)
    assert "secret-token" not in joined
    assert "token=***" not in joined
    assert "content_chars=" in joined


def test_keyword_handler_does_not_log_raw_matched_message():
    class FakeLogger:
        def __init__(self):
            self.messages = []

        def debug(self, message):
            self.messages.append(str(message))

        def info(self, message):
            self.messages.append(str(message))

        def warning(self, message):
            self.messages.append(str(message))

        def error(self, message):
            self.messages.append(str(message))

    handler = KeywordDetectionHandler()
    handler.logger = FakeLogger()
    handler.keywords = {"转人工"}

    assert handler.can_handle(_context("麻烦转人工，我的手机号是13800138000")) is True
    joined = "\n".join(handler.logger.messages)
    assert "13800138000" not in joined
    assert "我的手机号" not in joined
    assert "content_chars=" in joined


def test_keyword_handler_ignores_non_text_context():
    handler = KeywordDetectionHandler()
    handler.keywords = {"转人工"}
    context = _context("麻烦帮我转人工")
    context = context.model_copy(update={"type": ContextType.IMAGE})

    assert handler.can_handle(context) is False


def test_keyword_loader_accepts_safe_custom_phrase_and_rejects_broad_words(monkeypatch):
    monkeypatch.setattr(keyword_handler, "db_manager", FakeDbManager())

    handler = KeywordDetectionHandler()

    assert "催开发票" in handler.keywords
    assert "客服" not in handler.keywords
    assert "人工" not in handler.keywords
    assert "工单" not in handler.keywords


def test_keyword_loader_ignores_malformed_rows_without_dropping_valid_keywords(monkeypatch):
    class DirtyDb:
        def get_all_keywords(self):
            return [
                "bad row",
                {"keyword": None},
                {"keyword": "催开发票"},
            ]

    monkeypatch.setattr(keyword_handler, "db_manager", DirtyDb())

    handler = KeywordDetectionHandler()

    assert "催开发票" in handler.keywords


def test_keyword_loader_masks_sensitive_load_exception(monkeypatch):
    class BrokenDb:
        def get_all_keywords(self):
            raise RuntimeError("cookies=secret-cookie")

    class FakeLogger:
        def __init__(self):
            self.messages = []

        def debug(self, message):
            self.messages.append(str(message))

        def info(self, message):
            self.messages.append(str(message))

        def warning(self, message):
            self.messages.append(str(message))

        def error(self, message):
            self.messages.append(str(message))

    monkeypatch.setattr(keyword_handler, "db_manager", BrokenDb())
    monkeypatch.setattr(keyword_handler, "get_logger", lambda _name: FakeLogger())

    handler = KeywordDetectionHandler()

    joined = "\n".join(handler.logger.messages)
    assert "加载关键词失败" in joined
    assert "secret-cookie" not in joined
    assert "cookies=***" in joined
    assert "转人工" in handler.keywords


def test_keyword_handler_returns_true_when_no_cs_notice_is_sent(monkeypatch):
    class FakeSender:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def getAssignCsList(self):
            return {}

        def send_text(self, from_uid, content):
            return {"success": True}

    monkeypatch.setattr(keyword_handler, "SendMessage", FakeSender)
    monkeypatch.setattr(keyword_handler, "is_night_mode", lambda: False)

    async def run():
        handler = KeywordDetectionHandler()
        handler.keywords = {"转人工"}
        assert await handler.handle(_context("麻烦转人工"), {}) is True

    import asyncio

    asyncio.run(run())


def test_keyword_handler_returns_false_when_no_cs_notice_send_fails(monkeypatch):
    class FakeSender:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def getAssignCsList(self):
            return {}

        def send_text(self, from_uid, content):
            return {"success": False, "error_msg": "send failed"}

    monkeypatch.setattr(keyword_handler, "SendMessage", FakeSender)
    monkeypatch.setattr(keyword_handler, "is_night_mode", lambda: False)

    async def run():
        handler = KeywordDetectionHandler()
        handler.keywords = {"转人工"}
        assert await handler.handle(_context("麻烦转人工"), {}) is False

    import asyncio

    asyncio.run(run())


def test_keyword_handler_handles_non_dict_cs_list_items(monkeypatch):
    class FakeSender:
        calls = []

        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def getAssignCsList(self):
            return {"cs-2": "在线客服"}

        def send_text(self, from_uid, content):
            self.calls.append(("send_text", from_uid, content))
            return {"success": True}

        def move_conversation(self, from_uid, cs_uid):
            self.calls.append(("move", from_uid, cs_uid))
            return {"success": True}

    monkeypatch.setattr(keyword_handler, "SendMessage", FakeSender)
    monkeypatch.setattr(keyword_handler, "is_night_mode", lambda: False)

    class FakeDb:
        def get_all_keywords(self):
            return []

        def get_transfer_target(self, *_args):
            return None

    monkeypatch.setattr(keyword_handler, "db_manager", FakeDb())

    async def run():
        handler = KeywordDetectionHandler()
        handler.keywords = {"转人工"}
        assert await handler.handle(_context("麻烦转人工"), {}) is True

    import asyncio

    asyncio.run(run())
    assert ("move", "buyer-1", "cs-2") in FakeSender.calls


def test_keyword_handler_transfers_to_grouped_cs_list_item_not_group_key(monkeypatch):
    class FakeSender:
        calls = []

        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def getAssignCsList(self):
            return {
                "mall_cs": [
                    {"cs_uid": "cs_shop-1_seller-1", "username": "当前客服"},
                    {"cs_uid": "cs_shop-1_seller-2", "username": "客服2"},
                ]
            }

        def send_text(self, from_uid, content):
            self.calls.append(("send_text", from_uid, content))
            return {"success": True}

        def move_conversation(self, from_uid, cs_uid):
            self.calls.append(("move", from_uid, cs_uid))
            return {"success": True}

    monkeypatch.setattr(keyword_handler, "SendMessage", FakeSender)
    monkeypatch.setattr(keyword_handler, "is_night_mode", lambda: False)

    class FakeDb:
        def get_all_keywords(self):
            return []

        def get_transfer_target(self, *_args):
            return None

    monkeypatch.setattr(keyword_handler, "db_manager", FakeDb())

    async def run():
        handler = KeywordDetectionHandler()
        handler.keywords = {"转人工"}
        assert await handler.handle(_context("麻烦转人工"), {}) is True

    import asyncio

    asyncio.run(run())
    assert ("move", "buyer-1", "cs_shop-1_seller-2") in FakeSender.calls
    assert ("move", "buyer-1", "mall_cs") not in FakeSender.calls


def test_keyword_handler_handles_non_dict_transfer_result(monkeypatch):
    class FakeSender:
        calls = []

        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def getAssignCsList(self):
            return {"cs-2": {"username": "客服2"}}

        def send_text(self, from_uid, content):
            self.calls.append(("send_text", from_uid, content))
            return {"success": True}

        def move_conversation(self, from_uid, cs_uid):
            self.calls.append(("move", from_uid, cs_uid))
            return "temporary platform error"

    monkeypatch.setattr(keyword_handler, "SendMessage", FakeSender)
    monkeypatch.setattr(keyword_handler, "is_night_mode", lambda: False)

    class FakeDb:
        def get_all_keywords(self):
            return []

        def get_transfer_target(self, *_args):
            return None

    monkeypatch.setattr(keyword_handler, "db_manager", FakeDb())

    async def run():
        handler = KeywordDetectionHandler()
        handler.keywords = {"转人工"}
        assert await handler.handle(_context("麻烦转人工"), {}) is True

    import asyncio

    asyncio.run(run())
    assert ("move", "buyer-1", "cs-2") in FakeSender.calls
    assert any("转人工暂时没成功" in call[2] for call in FakeSender.calls if call[0] == "send_text")


def test_keyword_handler_treats_string_false_transfer_success_as_failure(monkeypatch):
    class FakeSender:
        calls = []

        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def getAssignCsList(self):
            return {"cs-2": {"username": "客服2"}}

        def send_text(self, from_uid, content):
            self.calls.append(("send_text", from_uid, content))
            return {"success": True}

        def move_conversation(self, from_uid, cs_uid):
            self.calls.append(("move", from_uid, cs_uid))
            return {"success": "false", "error_msg": "platform rejected"}

    monkeypatch.setattr(keyword_handler, "SendMessage", FakeSender)
    monkeypatch.setattr(keyword_handler, "is_night_mode", lambda: False)

    class FakeDb:
        def get_all_keywords(self):
            return []

        def get_transfer_target(self, *_args):
            return None

    monkeypatch.setattr(keyword_handler, "db_manager", FakeDb())

    async def run():
        handler = KeywordDetectionHandler()
        handler.keywords = {"转人工"}
        assert await handler.handle(_context("麻烦转人工"), {}) is True

    import asyncio

    asyncio.run(run())
    assert any("转人工暂时没成功" in call[2] for call in FakeSender.calls if call[0] == "send_text")


def test_keyword_handler_masks_sensitive_values_in_night_mode_notice(monkeypatch):
    class FakeSender:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def send_text(self, from_uid, content):
            return {"success": False, "error_msg": "token=secret-token"}

    class FakeLogger:
        def __init__(self):
            self.messages = []

        def warning(self, message):
            self.messages.append(str(message))

        def error(self, message):
            self.messages.append(str(message))

        def info(self, message):
            self.messages.append(str(message))

    monkeypatch.setattr(keyword_handler, "SendMessage", FakeSender)
    monkeypatch.setattr(keyword_handler, "is_night_mode", lambda: True)
    monkeypatch.setattr(keyword_handler, "get_night_mode_reply", lambda _key: "夜间提示 token=secret-token")

    async def run():
        handler = KeywordDetectionHandler()
        handler.keywords = {"转人工"}
        handler.logger = FakeLogger()
        assert await handler.handle(_context("麻烦转人工"), {}) is False
        joined = "\n".join(handler.logger.messages)
        assert "secret-token" not in joined
        assert "token=***" not in joined
        assert "result_type=dict" in joined

    import asyncio

    asyncio.run(run())


def test_keyword_handler_does_not_log_raw_night_mode_notice_result(monkeypatch):
    class FakeSender:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def send_text(self, from_uid, content):
            return {"success": False, "error_msg": "客户手机号13800138000，地址测试小区1号楼"}

    class FakeLogger:
        def __init__(self):
            self.messages = []

        def warning(self, message):
            self.messages.append(str(message))

        def error(self, message):
            self.messages.append(str(message))

        def info(self, message):
            self.messages.append(str(message))

    monkeypatch.setattr(keyword_handler, "SendMessage", FakeSender)
    monkeypatch.setattr(keyword_handler, "is_night_mode", lambda: True)
    monkeypatch.setattr(keyword_handler, "get_night_mode_reply", lambda _key: "夜间提示")

    async def run():
        handler = KeywordDetectionHandler()
        handler.keywords = {"转人工"}
        handler.logger = FakeLogger()
        assert await handler.handle(_context("麻烦转人工"), {}) is False
        joined = "\n".join(handler.logger.messages)
        assert "13800138000" not in joined
        assert "测试小区" not in joined
        assert "result_type=dict" in joined

    import asyncio

    asyncio.run(run())


def test_keyword_handler_masks_sensitive_values_in_transfer_notice_exception(monkeypatch):
    class FakeSender:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def getAssignCsList(self):
            return {"cs-2": {"username": "客服2"}}

        def send_text(self, from_uid, content):
            raise RuntimeError("token=secret-token")

        def move_conversation(self, from_uid, cs_uid):
            return {"success": True}

    class FakeDb:
        def get_all_keywords(self):
            return []

        def get_transfer_target(self, *_args):
            return None

    class FakeLogger:
        def __init__(self):
            self.messages = []

        def debug(self, message):
            self.messages.append(str(message))

        def warning(self, message):
            self.messages.append(str(message))

        def error(self, message):
            self.messages.append(str(message))

        def info(self, message):
            self.messages.append(str(message))

    monkeypatch.setattr(keyword_handler, "SendMessage", FakeSender)
    monkeypatch.setattr(keyword_handler, "is_night_mode", lambda: False)
    monkeypatch.setattr(keyword_handler, "db_manager", FakeDb())

    async def run():
        handler = KeywordDetectionHandler()
        handler.keywords = {"转人工"}
        handler.logger = FakeLogger()

        assert await handler.handle(_context("麻烦转人工"), {}) is True
        joined = "\n".join(handler.logger.messages)
        assert "secret-token" not in joined
        assert "token=***" in joined

    import asyncio

    asyncio.run(run())
