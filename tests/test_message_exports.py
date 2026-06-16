import Message
from Message.message import ChatMessage
from Message.core.handlers import CatchAllHandler, MessageHandler
from Message.models.queue_models import MessageWrapper
from bridge.context import ChannelType, Context, ContextType
from bridge.reply import Reply, ReplyType


def test_message_all_exports_are_defined():
    missing = [name for name in Message.__all__ if not hasattr(Message, name)]

    assert missing == []


def test_create_simple_handlers_returns_usable_fallback_handler():
    handlers = Message.create_simple_handlers()

    assert handlers
    assert all(isinstance(handler, MessageHandler) for handler in handlers)
    assert isinstance(handlers[-1], CatchAllHandler)


def test_reply_string_does_not_include_raw_content():
    reply = Reply(ReplyType.TEXT, "客户隐私 token=secret-token")

    text = str(reply)

    assert "secret-token" not in text
    assert "token=***" not in text
    assert "content_chars=" in text


def test_chat_message_string_does_not_include_content_or_raw_data():
    message = ChatMessage({"message": {"content": "raw token=secret-token"}})
    message.msg_id = "msg-1"
    message.from_user = "buyer-1"
    message.to_user = "seller-1"
    message.nickname = "测试昵称"
    message.content = "客户隐私 token=secret-token"
    message.msg_type = "TEXT"
    message.timestamp = "2026-06-14 12:00:00"

    text = str(message)

    assert "secret-token" not in text
    assert "token=***" not in text
    assert "客户隐私" not in text
    assert "raw token" not in text
    assert "content_chars=" in text
    assert "raw_data_type=dict" in text


def test_handler_chain_skips_broken_keyword_handler_with_sanitized_log(monkeypatch):
    class BrokenKeywordHandler:
        def __init__(self):
            raise RuntimeError("token=secret-token")

    class FakeLogger:
        def __init__(self):
            self.messages = []

        def warning(self, message):
            self.messages.append(str(message))

    fake_logger = FakeLogger()

    monkeypatch.setattr(Message, "_cached_keyword_handler", None)
    monkeypatch.setattr(Message, "KeywordDetectionHandler", BrokenKeywordHandler)
    monkeypatch.setattr(Message, "get_logger", lambda *_args, **_kwargs: fake_logger, raising=False)

    handlers = Message.handler_chain(use_ai=False)

    assert len(handlers) == 1
    assert isinstance(handlers[0], CatchAllHandler)
    joined = "\n".join(fake_logger.messages)
    assert "secret-token" not in joined
    assert "token=***" in joined


def test_catch_all_observes_without_acknowledging_processing():
    async def run():
        handler = CatchAllHandler()
        context = Context.create_pinduoduo_context(
            content="未处理消息",
            msg_id="msg-1",
            from_uid="buyer-1",
            user_msg_type=ContextType.TEXT,
            shop_id="shop-1",
            user_id="seller-1",
            channel_type=ChannelType.PINDUODUO,
        )
        wrapper = MessageWrapper(message_id="msg-1", context=context, timestamp=1.0)

        assert await handler.handle(context, wrapper.to_metadata()) is False

    import asyncio

    asyncio.run(run())


def test_catch_all_tolerates_non_dict_metadata():
    async def run():
        handler = CatchAllHandler()
        context = Context.create_pinduoduo_context(
            content="未处理消息",
            msg_id="msg-1",
            from_uid="buyer-1",
            user_msg_type=ContextType.TEXT,
            shop_id="shop-1",
            user_id="seller-1",
            channel_type=ChannelType.PINDUODUO,
        )

        assert await handler.handle(context, ["bad"]) is False

    import asyncio

    asyncio.run(run())


def test_message_handler_on_error_masks_sensitive_exception():
    class DemoHandler(MessageHandler):
        def can_handle(self, context):
            return True

        async def handle(self, context, metadata):
            return False

    async def run():
        messages = []
        handler = DemoHandler()
        handler.logger = type(
            "Logger",
            (),
            {"error": lambda _self, message: messages.append(str(message))},
        )()
        context = Context.create_pinduoduo_context(
            content="hello",
            msg_id="msg-1",
            from_uid="buyer-1",
            user_msg_type=ContextType.TEXT,
            shop_id="shop-1",
            user_id="seller-1",
            channel_type=ChannelType.PINDUODUO,
        )

        await handler.on_error(context, RuntimeError("token=secret-token"))

        joined = "\n".join(messages)
        assert "secret-token" not in joined
        assert "token=***" in joined

    import asyncio

    asyncio.run(run())
