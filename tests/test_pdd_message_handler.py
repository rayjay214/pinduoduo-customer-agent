import asyncio
from pathlib import Path

from Channel.pinduoduo.core import pdd_message_handler
from Channel.pinduoduo.core.pdd_message_handler import MessageHandlerMixin
from bridge.context import ChannelType, Context, ContextType


class FakeLogger:
    def __init__(self):
        self.warnings = []
        self.errors = []
        self.debugs = []
        self.infos = []

    def warning(self, message):
        self.warnings.append(message)

    def error(self, message):
        self.errors.append(message)

    def debug(self, message, *_args, **_kwargs):
        self.debugs.append(message)

    def info(self, message, *_args, **_kwargs):
        self.infos.append(message)

    def all_messages(self):
        return [*self.warnings, *self.errors, *self.debugs, *self.infos]


class HandlerHarness(MessageHandlerMixin):
    def __init__(self):
        self.channel_name = "pinduoduo"
        self.logger = FakeLogger()
        self.QUEUE_DEBOUNCE_SECONDS = 0


class FakePDDMessage:
    content = "你好"
    msg_id = "msg-1"
    from_user = "buyer"
    from_uid = "buyer-1"
    to_user = "seller"
    to_uid = "seller-1"
    nickname = "买家"
    timestamp = "now"
    user_msg_type = ContextType.TEXT
    raw_data = {"message": {"msg_id": "msg-1"}}


def _context(content="你好"):
    return Context.create_pinduoduo_context(
        content=content,
        msg_id="msg-1",
        from_uid="buyer-1",
        user_msg_type=ContextType.TEXT,
        shop_id="shop-1",
        user_id="seller-1",
        channel_type=ChannelType.PINDUODUO,
    )


def _mapping_context(content="你好", msg_id="msg-1", from_uid="buyer-1"):
    return Context(
        type=ContextType.TEXT,
        content=content,
        channel_type=ChannelType.PINDUODUO,
        kwargs={
            "msg_id": msg_id,
            "shop_id": "shop-1",
            "user_id": "seller-1",
            "from_uid": from_uid,
            "user_msg_type": ContextType.TEXT,
        },
    )


def test_websocket_trace_logs_do_not_include_raw_payload_content():
    harness = HandlerHarness()
    payload = {
        "response": "push",
        "message": {
            "msg_id": "msg-1",
            "type": 0,
            "content": "客户隐私 token=secret-token",
            "from": {"uid": "buyer-1", "role": "user"},
            "to": {"uid": "seller-1", "role": "mall_cs"},
        },
    }

    harness._log_websocket_raw(payload, "shop-1", "seller-1", "客服")

    joined = "\n".join(harness.logger.infos)
    assert "secret-token" not in joined
    assert "token=***" not in joined
    assert "客户隐私" not in joined
    assert "payload_chars=" in joined


def test_websocket_parsed_log_does_not_include_context_content():
    harness = HandlerHarness()
    pdd_message = FakePDDMessage()
    context = _mapping_context("客户隐私 token=secret-token")

    harness._log_websocket_parsed(pdd_message, context, "queue-1")

    joined = "\n".join(harness.logger.infos)
    assert "secret-token" not in joined
    assert "token=***" not in joined
    assert "客户隐私" not in joined
    assert "content_chars" in joined


def test_flush_cancel_restores_buffer(monkeypatch):
    async def run():
        harness = HandlerHarness()
        key = ("queue", "shop-1", "seller-1", "buyer-1")
        original_buffer = {"contexts": [_context()], "task": None}
        harness._pdd_reply_buffers = {key: original_buffer}
        put_started = asyncio.Event()
        release_put = asyncio.Event()

        async def slow_put(_queue_name, _context_arg):
            put_started.set()
            await release_put.wait()
            return "queued"

        monkeypatch.setattr("Message.put_message", slow_put)
        task = asyncio.create_task(harness._flush_debounced_context("queue", key))
        await put_started.wait()
        task.cancel()
        await task

        assert harness._pdd_reply_buffers[key] is original_buffer

    asyncio.run(run())


def test_flush_cancel_merges_with_new_buffer_instead_of_overwriting(monkeypatch):
    async def run():
        harness = HandlerHarness()
        key = ("queue", "shop-1", "seller-1", "buyer-1")
        old_context = _context("旧消息")
        new_context = _context("新消息")
        original_buffer = {"contexts": [old_context], "task": None}
        harness._pdd_reply_buffers = {key: original_buffer}
        put_started = asyncio.Event()
        release_put = asyncio.Event()

        async def slow_put(_queue_name, _context_arg):
            put_started.set()
            await release_put.wait()
            return "queued"

        monkeypatch.setattr("Message.put_message", slow_put)
        task = asyncio.create_task(harness._flush_debounced_context("queue", key))
        await put_started.wait()
        harness._pdd_reply_buffers[key] = {"contexts": [new_context], "task": None}

        task.cancel()
        await task

        restored_contexts = harness._pdd_reply_buffers[key]["contexts"]
        assert [context.content for context in restored_contexts] == ["旧消息", "新消息"]

    asyncio.run(run())


def test_flush_debounced_context_masks_sensitive_enqueue_exception(monkeypatch):
    async def run():
        harness = HandlerHarness()
        key = ("queue", "shop-1", "seller-1", "buyer-1")
        original_buffer = {"contexts": [_context()], "task": None}
        harness._pdd_reply_buffers = {key: original_buffer}

        async def broken_put(_queue_name, _context_arg):
            raise RuntimeError("token=secret-token")

        monkeypatch.setattr("Message.put_message", broken_put)

        await harness._flush_debounced_context("queue", key)

        joined = "\n".join(harness.logger.errors)
        assert "合并消息入队失败" in joined
        assert "secret-token" not in joined
        assert "token=***" in joined
        assert harness._pdd_reply_buffers[key] is original_buffer

    asyncio.run(run())


def test_conversation_key_accepts_mapping_kwargs():
    first = _mapping_context(from_uid="buyer-1")
    second = _mapping_context(from_uid="buyer-2")

    assert MessageHandlerMixin._conversation_key("queue", first) == (
        "queue",
        "shop-1",
        "seller-1",
        "buyer-1",
    )
    assert MessageHandlerMixin._conversation_key("queue", first) != MessageHandlerMixin._conversation_key("queue", second)


def test_merge_contexts_accepts_mapping_kwargs():
    harness = HandlerHarness()
    key = ("queue", "shop-1", "seller-1", "buyer-1")

    merged = harness._merge_contexts_for_queue(
        key,
        [
            _mapping_context("第一条", msg_id="msg-1"),
            _mapping_context("第二条", msg_id="msg-2"),
        ],
    )

    assert merged.content == "客户消息：第一条\n客户消息：第二条"
    assert merged.kwargs["msg_id"] == "msg-1+msg-2"
    assert merged.kwargs["user_msg_type"] == ContextType.TEXT


def test_convert_to_context_keeps_message_when_shop_missing(monkeypatch):
    harness = HandlerHarness()

    class FakeDbManager:
        def get_shop(self, _channel_name, _shop_id):
            return None

    monkeypatch.setattr(pdd_message_handler, "db_manager", FakeDbManager())

    context = harness._convert_to_context(FakePDDMessage(), "shop-1", "seller-1", "客服")

    assert context.content == "你好"
    assert context.kwargs.shop_name == ""
    assert context.kwargs.shop_id == "shop-1"
    assert harness.logger.warnings


def test_extract_ws_meta_reads_nested_from_uid():
    meta = MessageHandlerMixin._extract_ws_meta(
        {
            "response": "push",
            "message": {
                "type": 0,
                "sub_type": 0,
                "msg_id": "msg-1",
                "from": {"role": "user", "uid": "buyer-1"},
                "to": {"role": "mall_cs", "uid": "seller-1"},
            },
        }
    )

    assert meta["from_uid"] == "buyer-1"
    assert meta["to_uid"] == "seller-1"


def test_handle_websocket_message_masks_json_parse_failure_log():
    async def run():
        harness = HandlerHarness()

        await harness._process_websocket_message(
            "bad json token=secret-token cookies=secret-cookie",
            "queue",
            "shop-1",
            "seller-1",
            "客服",
        )

        joined = "\n".join(harness.logger.errors)
        assert "JSON 解析失败" in joined
        assert "secret-token" not in joined
        assert "secret-cookie" not in joined
        assert "token=***" not in joined
        assert "cookies=***" not in joined
        assert "message_chars=" in joined

    asyncio.run(run())


def test_handle_websocket_message_does_not_log_raw_json_parse_failure_payload():
    async def run():
        harness = HandlerHarness()

        await harness._process_websocket_message(
            "bad json 客户手机号13800138000 地址测试小区1号楼",
            "queue",
            "shop-1",
            "seller-1",
            "客服",
        )

        joined = "\n".join(harness.logger.errors)
        assert "JSON 解析失败" in joined
        assert "13800138000" not in joined
        assert "测试小区" not in joined
        assert "message_chars=" in joined

    asyncio.run(run())


def test_send_immediate_ack_skips_when_recipient_uid_missing():
    async def run():
        harness = HandlerHarness()

        class Sender:
            calls = []

            def send_text(self, recipient_uid, content):
                self.calls.append((recipient_uid, content))

        sender = Sender()
        await harness._send_immediate_ack(sender, "", ContextType.SYSTEM_HINT)

        assert sender.calls == []
        assert harness.logger.warnings

    asyncio.run(run())


def test_send_immediate_ack_sends_default_message_when_recipient_uid_present(monkeypatch):
    async def run():
        harness = HandlerHarness()

        class Sender:
            def __init__(self):
                self.calls = []

            def send_text(self, recipient_uid, content):
                self.calls.append((recipient_uid, content))

        monkeypatch.setattr(pdd_message_handler, "get_config", lambda _key, default=None: default)
        sender = Sender()
        await harness._send_immediate_ack(sender, "buyer-1", ContextType.SYSTEM_HINT)

        assert sender.calls == [("buyer-1", "[玫瑰]")]

    asyncio.run(run())


def test_send_immediate_ack_logs_sanitized_send_failure(monkeypatch):
    async def run():
        harness = HandlerHarness()

        class Sender:
            def __init__(self):
                self.calls = []

            def send_text(self, recipient_uid, content):
                self.calls.append((recipient_uid, content))
                return {"success": False, "error_msg": "platform rejected token=secret-token"}

        monkeypatch.setattr(pdd_message_handler, "get_config", lambda _key, default=None: default)
        sender = Sender()
        await harness._send_immediate_ack(sender, "buyer-1", ContextType.SYSTEM_HINT)

        assert sender.calls == [("buyer-1", "[玫瑰]")]
        joined = "\n".join(harness.logger.all_messages())
        assert "即时确认发送失败" in joined
        assert "secret-token" not in joined
        assert "token=***" in joined

    asyncio.run(run())


def test_process_websocket_message_masks_pdd_message_parse_exception(monkeypatch):
    async def run():
        harness = HandlerHarness()

        class BrokenPDDMessage:
            def __init__(self, _message_data):
                raise RuntimeError("token=secret-token")

        monkeypatch.setattr(pdd_message_handler, "PDDChatMessage", BrokenPDDMessage)

        await harness._process_websocket_message(
            '{"response":"push","message":{"type":0}}',
            "shop-1",
            "seller-1",
            "客服",
            "queue",
        )

        joined = "\n".join(harness.logger.errors)
        assert "secret-token" not in joined
        assert "token=***" in joined
        state = harness._pdd_ws_processing_errors[("shop-1", "seller-1", "queue")]
        assert state["count"] == 1
        assert state["last_reason"] == "pdd_message_parse"

    asyncio.run(run())


def test_process_websocket_message_warns_after_repeated_failures(monkeypatch):
    async def run():
        harness = HandlerHarness()
        monkeypatch.setattr(harness, "WEBSOCKET_ERROR_WARNING_THRESHOLD", 2)

        class BrokenPDDMessage:
            def __init__(self, _message_data):
                raise RuntimeError("token=secret-token")

        monkeypatch.setattr(pdd_message_handler, "PDDChatMessage", BrokenPDDMessage)

        for _ in range(2):
            await harness._process_websocket_message(
                '{"response":"push","message":{"type":0}}',
                "shop-1",
                "seller-1",
                "客服",
                "queue",
            )

        joined = "\n".join(harness.logger.warnings)
        assert "WebSocket 消息处理连续失败" in joined
        assert "secret-token" not in joined
        assert "token=***" in joined

    asyncio.run(run())


def test_process_websocket_message_clears_error_state_after_success(monkeypatch):
    async def run():
        harness = HandlerHarness()
        harness._pdd_ws_processing_errors = {
            ("shop-1", "seller-1", "queue"): {"count": 2, "last_reason": "json_decode"}
        }
        queued = []

        monkeypatch.setattr(pdd_message_handler, "PDDChatMessage", lambda _message_data: FakePDDMessage())
        monkeypatch.setattr(
            harness,
            "_convert_to_context",
            lambda _pdd_message, _shop_id, _user_id, _username: _context(),
        )

        async def fake_queue(queue_name, context):
            queued.append((queue_name, context.content))

        monkeypatch.setattr(harness, "_queue_message_with_debounce", fake_queue)

        await harness._process_websocket_message(
            '{"response":"push","message":{"type":0}}',
            "shop-1",
            "seller-1",
            "客服",
            "queue",
        )

        assert queued == [("queue", "你好")]
        assert harness._pdd_ws_processing_errors == {}

    asyncio.run(run())


def test_handle_immediate_message_accepts_mapping_kwargs(monkeypatch):
    async def run():
        harness = HandlerHarness()
        sent = []

        class Sender:
            def __init__(self, *_args, **_kwargs):
                pass

            def send_text(self, recipient_uid, content):
                sent.append((recipient_uid, content))
                return {"success": True}

        monkeypatch.setattr(
            "Channel.pinduoduo.utils.API.send_message.SendMessage",
            Sender,
        )
        monkeypatch.setattr(pdd_message_handler, "get_config", lambda _key, default=None: default)

        context = Context(
            type=ContextType.SYSTEM_HINT,
            content="系统提示",
            channel_type=ChannelType.PINDUODUO,
            kwargs={"username": "客服", "from_uid": "buyer-1"},
        )

        await harness._handle_immediate_message(context, "shop-1", "seller-1")

        assert sent == [("buyer-1", "[玫瑰]")]

    asyncio.run(run())


def test_handle_immediate_message_masks_sensitive_exception(monkeypatch):
    async def run():
        harness = HandlerHarness()

        class Sender:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("cookies=secret-cookie")

        monkeypatch.setattr(
            "Channel.pinduoduo.utils.API.send_message.SendMessage",
            Sender,
        )
        monkeypatch.setattr(pdd_message_handler, "get_config", lambda _key, default=None: default)

        context = Context(
            type=ContextType.SYSTEM_HINT,
            content="系统提示",
            channel_type=ChannelType.PINDUODUO,
            kwargs={"username": "客服", "from_uid": "buyer-1"},
        )

        await harness._handle_immediate_message(context, "shop-1", "seller-1")

        joined = "\n".join(harness.logger.errors)
        assert "secret-cookie" not in joined
        assert "cookies=***" in joined

    asyncio.run(run())


def test_handle_immediate_message_masks_sensitive_system_content(monkeypatch):
    async def run():
        harness = HandlerHarness()

        class Sender:
            def __init__(self, *_args, **_kwargs):
                pass

            def send_text(self, recipient_uid, content):
                return {"success": True}

        monkeypatch.setattr(
            "Channel.pinduoduo.utils.API.send_message.SendMessage",
            Sender,
        )
        monkeypatch.setattr(pdd_message_handler, "get_config", lambda _key, default=None: default)

        context = Context(
            type=ContextType.SYSTEM_HINT,
            content="系统提示 token=secret-token api_key=secret-api",
            channel_type=ChannelType.PINDUODUO,
            kwargs={"username": "客服", "from_uid": "buyer-1"},
        )

        await harness._handle_immediate_message(context, "shop-1", "seller-1")

        joined = "\n".join(harness.logger.all_messages())
        assert "secret-token" not in joined
        assert "secret-api" not in joined
        assert "token=***" not in joined
        assert "api_key=***" not in joined
        assert "content_chars=" in joined

    asyncio.run(run())


def test_handle_immediate_message_masks_sensitive_mall_cs_content(monkeypatch):
    async def run():
        harness = HandlerHarness()

        class Sender:
            def __init__(self, *_args, **_kwargs):
                pass

        monkeypatch.setattr(
            "Channel.pinduoduo.utils.API.send_message.SendMessage",
            Sender,
        )

        context = Context(
            type=ContextType.MALL_CS,
            content="客服消息 cookies=secret-cookie password=hunter2",
            channel_type=ChannelType.PINDUODUO,
            kwargs={"username": "客服", "from_uid": "buyer-1"},
        )

        await harness._handle_immediate_message(context, "shop-1", "seller-1")

        joined = "\n".join(harness.logger.all_messages())
        assert "secret-cookie" not in joined
        assert "hunter2" not in joined
        assert "cookies=***" not in joined
        assert "password=***" not in joined
        assert "content_chars=" in joined

    asyncio.run(run())


def test_handle_immediate_message_does_not_log_raw_system_content(monkeypatch):
    async def run():
        harness = HandlerHarness()

        class Sender:
            def __init__(self, *_args, **_kwargs):
                pass

            def send_text(self, recipient_uid, content):
                return {"success": True}

        monkeypatch.setattr(
            "Channel.pinduoduo.utils.API.send_message.SendMessage",
            Sender,
        )
        monkeypatch.setattr(pdd_message_handler, "get_config", lambda _key, default=None: default)

        context = Context(
            type=ContextType.SYSTEM_HINT,
            content="系统提示：客户手机号13800138000，地址测试小区1号楼",
            channel_type=ChannelType.PINDUODUO,
            kwargs={"username": "客服", "from_uid": "buyer-1"},
        )

        await harness._handle_immediate_message(context, "shop-1", "seller-1")

        joined = "\n".join(harness.logger.all_messages())
        assert "13800138000" not in joined
        assert "测试小区" not in joined
        assert "content_chars=" in joined

    asyncio.run(run())


def test_send_immediate_ack_uses_configured_message(monkeypatch):
    async def run():
        harness = HandlerHarness()

        class Sender:
            def __init__(self):
                self.calls = []

            def send_text(self, recipient_uid, content):
                self.calls.append((recipient_uid, content))

        def fake_get_config(key, default=None):
            values = {
                "pinduoduo.immediate_ack.message": "收到",
                "pinduoduo.immediate_ack.context_types": ["system_hint"],
            }
            return values.get(key, default)

        monkeypatch.setattr(pdd_message_handler, "get_config", fake_get_config)
        sender = Sender()
        await harness._send_immediate_ack(sender, "buyer-1", ContextType.SYSTEM_HINT)

        assert sender.calls == [("buyer-1", "收到")]

    asyncio.run(run())


def test_send_immediate_ack_context_types_are_case_insensitive(monkeypatch):
    async def run():
        harness = HandlerHarness()

        class Sender:
            def __init__(self):
                self.calls = []

            def send_text(self, recipient_uid, content):
                self.calls.append((recipient_uid, content))

        def fake_get_config(key, default=None):
            values = {
                "pinduoduo.immediate_ack.message": "收到",
                "pinduoduo.immediate_ack.context_types": ["SYSTEM_HINT"],
            }
            return values.get(key, default)

        monkeypatch.setattr(pdd_message_handler, "get_config", fake_get_config)
        sender = Sender()
        await harness._send_immediate_ack(sender, "buyer-1", ContextType.SYSTEM_HINT)

        assert sender.calls == [("buyer-1", "收到")]

    asyncio.run(run())


def test_send_immediate_ack_skips_disabled_context_type(monkeypatch):
    async def run():
        harness = HandlerHarness()

        class Sender:
            def __init__(self):
                self.calls = []

            def send_text(self, recipient_uid, content):
                self.calls.append((recipient_uid, content))

        def fake_get_config(key, default=None):
            values = {
                "pinduoduo.immediate_ack.context_types": ["withdraw", "transfer"],
            }
            return values.get(key, default)

        monkeypatch.setattr(pdd_message_handler, "get_config", fake_get_config)
        sender = Sender()
        await harness._send_immediate_ack(sender, "buyer-1", ContextType.SYSTEM_HINT)

        assert sender.calls == []
        assert harness.logger.debugs

    asyncio.run(run())


def test_send_immediate_ack_treats_string_false_as_disabled(monkeypatch):
    async def run():
        harness = HandlerHarness()

        class Sender:
            def __init__(self):
                self.calls = []

            def send_text(self, recipient_uid, content):
                self.calls.append((recipient_uid, content))

        def fake_get_config(key, default=None):
            if key == "pinduoduo.immediate_ack.enabled":
                return "false"
            return default

        monkeypatch.setattr(pdd_message_handler, "get_config", fake_get_config)
        sender = Sender()
        await harness._send_immediate_ack(sender, "buyer-1", ContextType.SYSTEM_HINT)

        assert sender.calls == []
        assert harness.logger.debugs

    asyncio.run(run())


def test_pdd_message_handler_source_has_no_mojibake_markers():
    source = Path("Channel/pinduoduo/core/pdd_message_handler.py").read_text(encoding="utf-8")

    for marker in ("娑堟伅", "澶辫触", "閿欒", "璺宠繃", "篊ontext"):
        assert marker not in source
