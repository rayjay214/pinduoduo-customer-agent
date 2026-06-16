import asyncio
import threading

from Agent.bot import Bot
from bridge.reply import Reply, ReplyType
from Message.handlers import ai_handler
from Message.handlers.ai_handler import AIReplyHandler, is_internal_context_only_message
from bridge.context import ChannelType, Context, ContextType


class FakeSender:
    called_thread_ids = []

    def __init__(self, shop_id, user_id):
        self.shop_id = shop_id
        self.user_id = user_id

    def send_text(self, from_uid, reply):
        FakeSender.called_thread_ids.append(threading.get_ident())
        return {"success": True}


def test_ai_handler_send_reply_runs_blocking_send_in_thread(monkeypatch):
    async def run():
        loop_thread_id = threading.get_ident()

        class FakeSendMessage:
            def __new__(cls, shop_id, user_id):
                return FakeSender(shop_id, user_id)

        monkeypatch.setattr(
            "Channel.pinduoduo.utils.API.send_message.SendMessage",
            FakeSendMessage,
        )
        handler = AIReplyHandler(bot=None)
        context = Context.create_pinduoduo_context(
            content="你好",
            msg_id="msg-1",
            from_uid="buyer-1",
            user_msg_type=ContextType.TEXT,
            shop_id="shop-1",
            user_id="seller-1",
            channel_type=ChannelType.PINDUODUO,
        )

        success = await handler._send_reply(
            context,
            "回复",
            {"shop_id": "shop-1", "user_id": "seller-1", "from_uid": "buyer-1"},
        )

        assert success is True
        assert FakeSender.called_thread_ids
        assert FakeSender.called_thread_ids[-1] != loop_thread_id

    asyncio.run(run())


def test_ai_handler_init_masks_di_failure_log(monkeypatch):
    messages = []

    class FakeContainer:
        def get(self, _type):
            raise RuntimeError("api_key=secret-token")

    class FakeLogger:
        def warning(self, message):
            messages.append(str(message))

    monkeypatch.setattr("core.di_container.container", FakeContainer())
    monkeypatch.setattr("utils.logger_loguru.get_logger", lambda *_args, **_kwargs: FakeLogger())

    handler = AIReplyHandler(bot=None)

    assert handler.bot is None
    joined = "\n".join(messages)
    assert "secret-token" not in joined
    assert "api_key=***" in joined


def test_ai_handler_sync_bot_reply_runs_in_thread():
    async def run():
        loop_thread_id = threading.get_ident()

        class SyncBot:
            def __init__(self):
                self.reply_thread_id = None

            def reply(self, query, context):
                self.reply_thread_id = threading.get_ident()
                return Reply(ReplyType.TEXT, f"reply:{query}")

        bot = SyncBot()
        handler = AIReplyHandler(bot=bot)
        context = Context.create_pinduoduo_context(
            content="你好",
            msg_id="msg-1",
            from_uid="buyer-1",
            user_msg_type=ContextType.TEXT,
            shop_id="shop-1",
            user_id="seller-1",
            channel_type=ChannelType.PINDUODUO,
        )

        reply = await handler._get_ai_reply("你好", context)

        assert reply == "reply:你好"
        assert bot.reply_thread_id is not None
        assert bot.reply_thread_id != loop_thread_id

    asyncio.run(run())


def test_bot_default_async_reply_runs_sync_reply_in_thread():
    async def run():
        loop_thread_id = threading.get_ident()

        class SyncOnlyBot(Bot):
            def __init__(self):
                self.reply_thread_id = None

            def reply(self, query, context=None):
                self.reply_thread_id = threading.get_ident()
                return Reply(ReplyType.TEXT, "ok")

        bot = SyncOnlyBot()
        reply = await bot.async_reply("你好")

        assert reply.content == "ok"
        assert bot.reply_thread_id is not None
        assert bot.reply_thread_id != loop_thread_id

    asyncio.run(run())


def test_internal_context_filter_accepts_preprocessed_content_prefix():
    assert is_internal_context_only_message("内容：上一条客户问题：好的") is True
    assert is_internal_context_only_message("内容: 上次客户消息：这个多少钱") is True


def test_internal_context_filter_does_not_match_normal_customer_text():
    assert is_internal_context_only_message("我想问下上一条客户问题是什么意思") is False
    assert is_internal_context_only_message("内容：我想问下上一条客户问题是什么意思") is False


def test_internal_context_skip_log_does_not_include_raw_content():
    async def run():
        handler = AIReplyHandler(bot=None)
        logs = []

        class FakeLogger:
            def info(self, message):
                logs.append(str(message))

            def warning(self, message):
                logs.append(str(message))

            def exception(self, message):
                logs.append(str(message))

        handler.logger = FakeLogger()
        context = Context.create_pinduoduo_context(
            content="上一条客户问题：token=secret-token",
            msg_id="msg-1",
            from_uid="buyer-1",
            user_msg_type=ContextType.TEXT,
            shop_id="shop-1",
            user_id="seller-1",
            channel_type=ChannelType.PINDUODUO,
        )

        assert await handler.handle(context, {}) is True

        joined = "\n".join(logs)
        assert "secret-token" not in joined
        assert "token=***" not in joined
        assert "content_chars=" in joined

    asyncio.run(run())


def test_fallback_exception_returns_false_for_queue_retry(monkeypatch):
    async def run():
        handler = AIReplyHandler(bot=None)

        async def boom(*_args, **_kwargs):
            raise RuntimeError("send failed")

        monkeypatch.setattr(handler, "_send_reply", boom)
        context = Context.create_pinduoduo_context(
            content="你好",
            msg_id="msg-1",
            from_uid="buyer-1",
            user_msg_type=ContextType.TEXT,
            shop_id="shop-1",
            user_id="seller-1",
            channel_type=ChannelType.PINDUODUO,
        )

        assert await handler._handle_fallback(
            context,
            {"shop_id": "shop-1", "user_id": "seller-1", "from_uid": "buyer-1"},
        ) is False

    asyncio.run(run())


def test_send_reply_rejects_non_dict_metadata():
    async def run():
        handler = AIReplyHandler(bot=None)
        context = Context.create_pinduoduo_context(
            content="你好",
            msg_id="msg-1",
            from_uid="buyer-1",
            user_msg_type=ContextType.TEXT,
            shop_id="shop-1",
            user_id="seller-1",
            channel_type=ChannelType.PINDUODUO,
        )

        assert await handler._send_reply(context, "回复", ["bad"]) is False

    asyncio.run(run())


def test_send_reply_treats_string_false_success_as_failure(monkeypatch):
    async def run():
        class FakeSendMessage:
            def __init__(self, shop_id, user_id):
                self.shop_id = shop_id
                self.user_id = user_id

            def send_text(self, from_uid, reply):
                return {"success": "false", "error_msg": "platform rejected"}

        monkeypatch.setattr(
            "Channel.pinduoduo.utils.API.send_message.SendMessage",
            FakeSendMessage,
        )
        handler = AIReplyHandler(bot=None)
        context = Context.create_pinduoduo_context(
            content="你好",
            msg_id="msg-1",
            from_uid="buyer-1",
            user_msg_type=ContextType.TEXT,
            shop_id="shop-1",
            user_id="seller-1",
            channel_type=ChannelType.PINDUODUO,
        )

        assert await handler._send_reply(
            context,
            "回复",
            {"shop_id": "shop-1", "user_id": "seller-1", "from_uid": "buyer-1"},
        ) is False

    asyncio.run(run())
