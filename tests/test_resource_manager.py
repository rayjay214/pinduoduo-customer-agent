import asyncio
import weakref

from utils.resource_manager import WebSocketResourceManager


class DummyWebSocket:
    pass


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


def test_reference_cleanup_without_running_loop_does_not_raise():
    manager = WebSocketResourceManager()
    websocket = DummyWebSocket()
    ref = weakref.ref(websocket)
    manager._connections.add(ref)

    manager._schedule_reference_cleanup(ref)

    assert ref not in manager._connections


def test_reference_cleanup_with_running_loop_schedules_task():
    async def run():
        manager = WebSocketResourceManager()
        websocket = DummyWebSocket()
        ref = weakref.ref(websocket)
        manager._connections.add(ref)

        manager._schedule_reference_cleanup(ref)
        await asyncio.sleep(0)

        assert ref not in manager._connections

    asyncio.run(run())


def test_cleanup_all_masks_sensitive_close_exception():
    async def run():
        class BrokenWebSocket:
            closed = False

            def close(self):
                raise RuntimeError("token=secret-token")

        manager = WebSocketResourceManager()
        manager.logger = FakeLogger()
        websocket = BrokenWebSocket()
        manager.register_websocket(websocket, "broken")

        await manager.cleanup_all()

        joined = "\n".join(manager.logger.messages)
        assert "清理WebSocket失败" in joined
        assert "secret-token" not in joined
        assert "token=***" in joined

    asyncio.run(run())
