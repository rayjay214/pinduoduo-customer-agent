from Message.core.handlers import CatchAllHandler
from Message.handlers.base import BaseHandler
from bridge.context import Context, ContextType


class DummyHandler(BaseHandler):
    def can_handle(self, context):
        return False

    async def handle(self, context, metadata):
        return False


def test_get_user_info_accepts_mapping_kwargs():
    handler = DummyHandler()
    context = Context(
        type=ContextType.TEXT,
        content="hello",
        kwargs={"from_uid": "buyer-1", "username": "Alice"},
    )

    assert handler._get_user_info(context) == "用户:Alice(buyer-1)"


def test_log_message_masks_sensitive_content_and_extra_info():
    async def run():
        handler = DummyHandler()
        logs = []

        class FakeLogger:
            def info(self, message):
                logs.append(str(message))

        handler.logger = FakeLogger()
        context = Context(
            type=ContextType.TEXT,
            content="请查一下 token=secret-token api_key=secret-api",
            kwargs={"from_uid": "buyer-1"},
        )

        await handler.log_message(context, "测试动作", "password=hunter2")

        joined = "\n".join(logs)
        assert "secret-token" not in joined
        assert "secret-api" not in joined
        assert "hunter2" not in joined
        assert "token=***" not in joined
        assert "api_key=***" not in joined
        assert "password=***" in joined
        assert "content_chars=" in joined

    import asyncio

    asyncio.run(run())


def test_log_message_does_not_include_raw_content_preview():
    async def run():
        handler = DummyHandler()
        logs = []

        class FakeLogger:
            def info(self, message):
                logs.append(str(message))

        handler.logger = FakeLogger()
        context = Context(
            type=ContextType.TEXT,
            content="我的手机号是13800138000，地址是测试小区1号楼",
            kwargs={"from_uid": "buyer-1"},
        )

        await handler.log_message(context, "测试动作")

        joined = "\n".join(logs)
        assert "13800138000" not in joined
        assert "测试小区" not in joined
        assert "content_chars=" in joined

    import asyncio

    asyncio.run(run())


def test_catch_all_handler_masks_sensitive_content_preview():
    async def run():
        handler = CatchAllHandler()
        logs = []

        class FakeLogger:
            def info(self, message):
                logs.append(str(message))

        handler.logger = FakeLogger()
        context = Context(
            type=ContextType.TEXT,
            content="兜底记录 token=secret-token api_key=secret-api",
        )

        assert await handler.handle(context, {"user_id": "user-1", "message_id": "msg-1"}) is False

        joined = "\n".join(logs)
        assert "secret-token" not in joined
        assert "secret-api" not in joined
        assert "token=***" not in joined
        assert "api_key=***" not in joined
        assert "content_chars=" in joined

    import asyncio

    asyncio.run(run())


def test_catch_all_handler_does_not_include_raw_content_preview():
    async def run():
        handler = CatchAllHandler()
        logs = []

        class FakeLogger:
            def info(self, message):
                logs.append(str(message))

        handler.logger = FakeLogger()
        context = Context(
            type=ContextType.TEXT,
            content="我的手机号是13800138000，地址是测试小区1号楼",
        )

        assert await handler.handle(context, {"user_id": "user-1", "message_id": "msg-1"}) is False

        joined = "\n".join(logs)
        assert "13800138000" not in joined
        assert "测试小区" not in joined
        assert "content_chars=" in joined

    import asyncio

    asyncio.run(run())
