import asyncio
import ast
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from Channel.pinduoduo.core.pdd_config import ReconnectConfig
from Channel.pinduoduo.core.pdd_connection import ConnectionMixin
from Channel.pinduoduo.core.pdd_lifecycle import LifecycleMixin
from core.connection_status import ConnectionState, ConnectionStatusManager


class FakeLogger:
    def __init__(self):
        self.messages = []

    def __getattr__(self, _name):
        return lambda *args, **kwargs: self.messages.append(" ".join(str(arg) for arg in args))


class FakeWebSocket:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class ClosingWebSocket:
    def __aiter__(self):
        return self

    async def __anext__(self):
        from websockets import exceptions as ws_exceptions

        raise ws_exceptions.ConnectionClosedError(None, None)


class ExplodingWebSocket:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise RuntimeError("cookies=secret-cookie")


class LifecycleHarness(LifecycleMixin):
    def __init__(self):
        self.logger = FakeLogger()
        self.channel_name = "pinduoduo"
        self.status_manager = ConnectionStatusManager()
        self.reconnect_config = ReconnectConfig(enable_auto_reconnect=False)
        self._stop_event = None
        self._stop_events = {}
        self.ws = None
        self._websockets = {}
        self._reconnect_tasks = {}
        self._heartbeat_tasks = {}
        self.processing_tasks = set()
        self._processing_tasks_by_connection = {}

    async def _safe_close_websocket(self, websocket):
        await websocket.close()

    async def _connect_single_attempt(self, *_args, **_kwargs):
        raise AssertionError("should not connect with malformed account info")


class RetryHarness(ConnectionMixin, LifecycleMixin):
    def __init__(self):
        self.logger = FakeLogger()
        self.status_manager = ConnectionStatusManager()
        self.reconnect_config = ReconnectConfig(
            max_attempts=2,
            initial_delay=0,
            max_delay=0,
            stable_reset_seconds=5,
        )
        self._stop_event = None
        self._stop_events = {}
        self.failures = 0
        self.failure_messages = []
        self.success_calls = 0

    async def _connect_single_attempt(self, shop_id, user_id, username, on_success, on_failure):
        self.status_manager.update_status(shop_id, user_id, username, ConnectionState.CONNECTED)
        status = self.status_manager.get_status(shop_id, user_id)
        if self.success_calls == 0:
            status.last_connect_time = datetime.now() - timedelta(seconds=10)
        else:
            status.last_connect_time = datetime.now()
        self.success_calls += 1
        raise RuntimeError("lost connection")


class FailingRetryHarness(ConnectionMixin, LifecycleMixin):
    def __init__(self):
        self.logger = FakeLogger()
        self.status_manager = ConnectionStatusManager()
        self.reconnect_config = ReconnectConfig(
            max_attempts=1,
            initial_delay=0,
            max_delay=0,
            stable_reset_seconds=5,
        )
        self._stop_event = None
        self._stop_events = {}

    async def _connect_single_attempt(self, *_args, **_kwargs):
        raise RuntimeError("token=secret-token")


def test_stop_account_state_is_isolated_by_connection_key():
    async def run():
        harness = LifecycleHarness()
        ws1 = FakeWebSocket()
        ws2 = FakeWebSocket()
        key1 = harness._connection_key("shop-1", "user-1")
        key2 = harness._connection_key("shop-2", "user-2")
        event1 = harness._get_or_create_stop_event(key1)
        event2 = harness._get_or_create_stop_event(key2)
        harness._set_active_websocket(key1, ws1)
        harness._set_active_websocket(key2, ws2)

        harness.request_stop("shop-1", "user-1")
        await harness._cleanup_resources(
            "pdd_shop-1_user-1",
            cleanup_reconnect_tasks=True,
            cleanup_heartbeat_tasks=True,
            cleanup_all_websockets=False,
            stop_consumer=False,
            connection_key=key1,
        )

        assert event1.is_set() is True
        assert event2.is_set() is False
        assert ws1.closed is True
        assert ws2.closed is False
        assert key1 not in harness._websockets
        assert harness._websockets[key2] is ws2

    asyncio.run(run())


def test_stop_all_connections_cleans_processing_tasks_and_consumers(monkeypatch):
    async def run():
        stopped_consumers = []

        class FakeMessageConsumerManager:
            async def stop_all(self):
                stopped_consumers.append(True)

        import Message

        monkeypatch.setattr(Message, "message_consumer_manager", FakeMessageConsumerManager())

        harness = LifecycleHarness()
        task_started = asyncio.Event()

        async def long_running():
            task_started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(long_running())
        await task_started.wait()
        harness.processing_tasks.add(task)

        await harness.stop_all_connections()

        assert task.cancelled() is True
        assert harness.processing_tasks == set()
        assert stopped_consumers == [True]

    asyncio.run(run())


def test_reconnect_attempts_are_consecutive_after_stable_connection():
    async def run():
        harness = RetryHarness()
        shop_id = "retry-shop"
        user_id = "retry-user"
        harness.status_manager.clear_connection(shop_id, user_id)

        def on_failure(message):
            harness.failure_messages.append(message)

        await harness._connect_with_retry(shop_id, user_id, "客服", lambda: None, on_failure)

        assert len(harness.failure_messages) == 1
        assert "已达到最大重试次数" in harness.failure_messages[0]
        assert harness.success_calls == harness.reconnect_config.max_attempts
        harness.status_manager.clear_connection(shop_id, user_id)

    asyncio.run(run())


def test_connect_with_retry_masks_sensitive_failure_messages():
    async def run():
        harness = FailingRetryHarness()
        failures = []
        messages = []

        class RetryLogger:
            def info(self, message):
                messages.append(str(message))

            def warning(self, message):
                messages.append(str(message))

            def error(self, message):
                messages.append(str(message))

        import Channel.pinduoduo.core.pdd_connection as pdd_connection_module

        pdd_connection_module.get_logger = lambda _name: RetryLogger()

        await harness._connect_with_retry("shop-1", "user-1", "客服", lambda: None, failures.append)

        assert failures
        assert "secret-token" not in failures[0]
        assert "token=***" in failures[0]
        joined = "\n".join(messages)
        assert "secret-token" not in joined
        assert "token=***" in joined

    asyncio.run(run())


def test_websocket_url_encodes_access_token_query_value():
    harness = LifecycleHarness()
    harness.base_url = "wss://m-ws.pinduoduo.com/"
    harness.API_VERSION = "202506091557"

    url = harness._build_websocket_url("abc+/=&token")
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    assert params["access_token"] == ["abc+/=&token"]
    assert params["role"] == ["mall_cs"]
    assert params["client"] == ["web"]
    assert params["version"] == ["202506091557"]
    assert "access_token=abc+/=&token" not in url


def test_start_account_rejects_malformed_account_info(monkeypatch):
    async def run():
        from Channel.pinduoduo.core import pdd_lifecycle

        class FakeDbManager:
            def get_account(self, *_args):
                return "bad account row"

        monkeypatch.setattr(pdd_lifecycle, "db_manager", FakeDbManager())
        harness = LifecycleHarness()
        failures = []

        await harness.start_account("shop-1", "user-1", lambda: None, failures.append)

        assert failures
        assert "账号 user-1 在数据库中不存在" in failures[0]
        assert harness._reconnect_tasks == {}

    asyncio.run(run())


def test_stop_account_ignores_malformed_account_info(monkeypatch):
    async def run():
        from Channel.pinduoduo.core import pdd_lifecycle

        class FakeDbManager:
            def get_account(self, *_args):
                return ["bad account row"]

        monkeypatch.setattr(pdd_lifecycle, "db_manager", FakeDbManager())
        harness = LifecycleHarness()

        await harness.stop_account("shop-1", "user-1")

        assert harness._reconnect_tasks == {}

    asyncio.run(run())


def test_connection_closed_error_handler_precedes_parent_handler():
    source = Path("Channel/pinduoduo/core/pdd_lifecycle.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    found_order = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        handler_names = []
        for handler in node.handlers:
            exc = handler.type
            if isinstance(exc, ast.Attribute):
                handler_names.append(exc.attr)
        if "ConnectionClosedError" in handler_names and "ConnectionClosed" in handler_names:
            found_order = handler_names
            break

    assert found_order is not None
    assert found_order.index("ConnectionClosedError") < found_order.index("ConnectionClosed")


def test_message_loop_propagates_unexpected_connection_close():
    async def run():
        harness = LifecycleHarness()
        stop_event = asyncio.Event()

        try:
            await harness._message_loop(
                ClosingWebSocket(),
                "shop-1",
                "user-1",
                "客服",
                "queue",
                stop_event,
            )
        except RuntimeError as exc:
            assert "WebSocket连接异常关闭" in str(exc)
        else:
            raise AssertionError("expected unexpected websocket close to propagate")

    asyncio.run(run())


def test_message_loop_masks_unexpected_runtime_exception():
    async def run():
        harness = LifecycleHarness()
        stop_event = asyncio.Event()

        try:
            await harness._message_loop(
                ExplodingWebSocket(),
                "shop-1",
                "user-1",
                "客服",
                "queue",
                stop_event,
            )
        except RuntimeError as exc:
            assert "secret-cookie" not in str(exc)
            assert "cookies=***" in str(exc)
        else:
            raise AssertionError("expected unexpected websocket error to propagate")

    asyncio.run(run())


def test_init_masks_sensitive_exception_in_failure_outputs(monkeypatch):
    async def run():
        from Channel.pinduoduo.core import pdd_lifecycle

        class FakeToken:
            def __init__(self, *_args):
                pass

            def get_token(self):
                raise RuntimeError("token=secret-token")

        cleanup_calls = []

        async def fake_cleanup(*_args, **_kwargs):
            cleanup_calls.append(True)

        monkeypatch.setattr(pdd_lifecycle, "GetToken", FakeToken)

        harness = LifecycleHarness()
        harness._cleanup_resources = fake_cleanup
        failures = []

        await harness.init("shop-1", "user-1", "客服", lambda: None, failures.append)

        status = harness.status_manager.get_status("shop-1", "user-1")
        joined_logs = "\n".join(harness.logger.messages)
        assert cleanup_calls == [True]
        assert failures
        assert "secret-token" not in failures[0]
        assert "secret-token" not in str(status.last_error)
        assert "secret-token" not in joined_logs
        assert "token=***" in failures[0]

    asyncio.run(run())


def test_cleanup_reconnect_tasks_masks_wait_exception(monkeypatch):
    async def run():
        harness = LifecycleHarness()
        started = asyncio.Event()

        async def long_running():
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(long_running())
        await started.wait()
        harness._reconnect_tasks["shop-1_user-1"] = task

        async def fake_wait_for(_awaitable, timeout=None):
            raise RuntimeError("token=secret-token")

        monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

        await harness._cleanup_reconnect_tasks("shop-1_user-1")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        joined = "\n".join(harness.logger.messages)
        assert "secret-token" not in joined
        assert "token=***" in joined

    asyncio.run(run())
