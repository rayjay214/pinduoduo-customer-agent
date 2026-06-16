import asyncio

import pytest

from core.connection_status import ConnectionStatusManager
from Channel.pinduoduo.core.pdd_config import HeartbeatConfig
from Channel.pinduoduo.core.pdd_lifecycle import LifecycleMixin


class HeartbeatHarness(LifecycleMixin):
    def __init__(self, timeout=0.05):
        self.heartbeat_config = HeartbeatConfig(heartbeat_timeout=timeout)
        self.status_manager = ConnectionStatusManager()
        self._heartbeat_tasks = {}
        self.logs = []
        self.logger = type(
            "Logger",
            (),
            {
                "debug": lambda _self, message, *args, **kwargs: self.logs.append(str(message)),
                "warning": lambda _self, message, *args, **kwargs: self.logs.append(str(message)),
                "error": lambda _self, message, *args, **kwargs: self.logs.append(str(message)),
                "exception": lambda _self, message, *args, **kwargs: self.logs.append(str(message)),
            },
        )()


class FakeWebSocket:
    def __init__(self, waiter):
        self.waiter = waiter
        self.ping_called = False

    async def ping(self):
        self.ping_called = True
        return self.waiter


class FailingPingWebSocket:
    async def ping(self):
        raise RuntimeError("anti-content=secret-anti")


def test_send_heartbeat_ping_waits_for_pong():
    async def run():
        pong_waiter = asyncio.Future()
        websocket = FakeWebSocket(pong_waiter)
        task = asyncio.create_task(HeartbeatHarness()._send_heartbeat_ping(websocket))

        await asyncio.sleep(0)
        assert websocket.ping_called is True
        assert task.done() is False

        pong_waiter.set_result(None)
        await task

    asyncio.run(run())


def test_send_heartbeat_ping_times_out_without_pong():
    async def run():
        pong_waiter = asyncio.Future()
        websocket = FakeWebSocket(pong_waiter)

        with pytest.raises(asyncio.TimeoutError):
            await HeartbeatHarness(timeout=0.01)._send_heartbeat_ping(websocket)

    asyncio.run(run())


def test_heartbeat_loop_exits_promptly_when_stop_event_is_set():
    async def run():
        stop_event = asyncio.Event()
        websocket = FakeWebSocket(asyncio.Future())
        harness = HeartbeatHarness(timeout=0.01)
        harness.heartbeat_config.heartbeat_interval = 30

        task = asyncio.create_task(
            harness._heartbeat_loop(websocket, "shop-1", "user-1", "客服", stop_event)
        )
        await asyncio.sleep(0)
        stop_event.set()
        await asyncio.wait_for(task, timeout=0.2)

    asyncio.run(run())


def test_heartbeat_loop_masks_sensitive_ping_exception():
    async def run():
        stop_event = asyncio.Event()
        harness = HeartbeatHarness(timeout=0.01)
        harness.heartbeat_config.heartbeat_interval = 30
        harness.heartbeat_config.heartbeat_timeout = 0.01
        harness.heartbeat_config.max_heartbeat_failures = 1

        await harness._heartbeat_loop(FailingPingWebSocket(), "shop-1", "user-1", "客服", stop_event)

        joined = "\n".join(harness.logs)
        assert "secret-anti" not in joined
        assert "anti-content=***" in joined

    asyncio.run(run())
