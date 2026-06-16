import asyncio
import time

from Message.core.consumer import MessageConsumer, MessageConsumerManager
from Message.core.queue import QueueManager, queue_manager
from Message.core.handlers import MessageHandler
from Message.models.queue_models import QueueConfig
from Message.models.queue_models import MessageWrapper
from bridge.context import ChannelType, Context, ContextType


class ConditionalHandler(MessageHandler):
    def __init__(self, started, release):
        super().__init__()
        self.started = started
        self.release = release
        self.handled = []

    def can_handle(self, context):
        return True

    async def handle(self, context, metadata):
        self.handled.append(context.content)
        if context.kwargs.from_uid == "buyer-1":
            self.started.set()
            await self.release.wait()
        return True


class FastHandler(MessageHandler):
    def __init__(self):
        super().__init__()
        self.handled = []

    def can_handle(self, context):
        return True

    async def handle(self, context, metadata):
        self.handled.append(context.content)
        return True


class MetadataCaptureHandler(MessageHandler):
    def __init__(self):
        super().__init__()
        self.metadata = None

    def can_handle(self, context):
        return True

    async def handle(self, context, metadata):
        self.metadata = dict(metadata)
        return True


class FailingHandler(MessageHandler):
    def can_handle(self, context):
        return True

    async def handle(self, context, metadata):
        return False


class RaisingHandler(MessageHandler):
    def can_handle(self, context):
        return True

    async def handle(self, context, metadata):
        raise RuntimeError("token=secret-token")


def _wrapper(content, from_uid, msg_id):
    context = Context.create_pinduoduo_context(
        content=content,
        msg_id=msg_id,
        from_uid=from_uid,
        user_msg_type=ContextType.TEXT,
        shop_id="shop-1",
        user_id="seller-1",
        channel_type=ChannelType.PINDUODUO,
    )
    return MessageWrapper(message_id=msg_id, context=context, timestamp=time.time())


def test_waiting_same_conversation_does_not_hold_global_concurrency_slot():
    async def run():
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        handler = ConditionalHandler(first_started, release_first)
        consumer = MessageConsumer("queue", max_concurrent=2)
        consumer.add_handler(handler)

        first = asyncio.create_task(consumer._process_message(_wrapper("first", "buyer-1", "msg-1")))
        await first_started.wait()
        second = asyncio.create_task(consumer._process_message(_wrapper("second", "buyer-1", "msg-2")))
        await asyncio.sleep(0)

        third = asyncio.create_task(consumer._process_message(_wrapper("third", "buyer-2", "msg-3")))
        await asyncio.wait_for(third, timeout=0.1)

        assert "third" in handler.handled

        release_first.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=0.2)
        assert handler.handled == ["first", "third", "second"]

    asyncio.run(run())


def test_conversation_locks_are_removed_after_processing():
    async def run():
        consumer = MessageConsumer("queue", max_concurrent=1)
        handler = FastHandler()
        consumer.add_handler(handler)

        await consumer._process_message(_wrapper("hello", "buyer-1", "msg-1"))

        assert consumer._user_locks == {}

    asyncio.run(run())


def test_consumer_normalizes_non_positive_max_concurrent_to_one():
    async def run():
        consumer = MessageConsumer("queue", max_concurrent=0)
        handler = FastHandler()
        consumer.add_handler(handler)

        processed = await asyncio.wait_for(
            consumer._process_message(_wrapper("hello", "buyer-1", "msg-1")),
            timeout=0.1,
        )

        assert processed is True
        assert consumer.max_concurrent == 1
        assert handler.handled == ["hello"]

    asyncio.run(run())


def test_extract_user_id_accepts_mapping_kwargs():
    consumer = MessageConsumer("queue", max_concurrent=1)
    context = Context(
        type=ContextType.TEXT,
        content="hello",
        channel_type=ChannelType.PINDUODUO,
        kwargs={"from_uid": "buyer-1"},
    )

    assert consumer._extract_user_id(context) == "pinduoduo_buyer-1"


def test_process_message_copies_channel_metadata_from_mapping_kwargs():
    async def run():
        consumer = MessageConsumer("queue", max_concurrent=1)
        handler = MetadataCaptureHandler()
        consumer.add_handler(handler)
        context = Context(
            type=ContextType.TEXT,
            content="hello",
            channel_type=ChannelType.PINDUODUO,
            kwargs={
                "shop_id": "shop-1",
                "user_id": "seller-1",
                "from_uid": "buyer-1",
            },
        )
        wrapper = MessageWrapper(message_id="msg-1", context=context, timestamp=time.time())

        assert await consumer._process_message(wrapper) is True
        assert handler.metadata["shop_id"] == "shop-1"
        assert handler.metadata["user_id"] == "seller-1"
        assert handler.metadata["from_uid"] == "buyer-1"

    asyncio.run(run())


def test_metadata_append_exception_logs_are_sanitized():
    async def run():
        class ExplodingKwargs:
            def __getattr__(self, name):
                if name == "shop_id":
                    raise RuntimeError("token=secret-token")
                return None

        consumer = MessageConsumer("queue", max_concurrent=1)
        handler = MetadataCaptureHandler()
        consumer.add_handler(handler)
        context = Context(
            type=ContextType.TEXT,
            content="hello",
            channel_type=ChannelType.PINDUODUO,
            kwargs=ExplodingKwargs(),
        )
        wrapper = MessageWrapper(message_id="msg-1", context=context, timestamp=time.time())
        logs = []

        class FakeLogger:
            def debug(self, message):
                logs.append(str(message))

            def warning(self, message):
                logs.append(str(message))

            def error(self, message):
                logs.append(str(message))

        consumer.logger = FakeLogger()

        assert await consumer._process_message(wrapper) is True
        joined = "\n".join(logs)
        assert "secret-token" not in joined
        assert "token=***" in joined

    asyncio.run(run())


def test_failed_message_is_requeued_before_dead_letter():
    async def run():
        manager = QueueManager()
        queue = manager.get_or_create_queue(
            "queue",
            QueueConfig(max_size=10, enable_deduplication=False, max_processing_retries=1),
        )
        wrapper = _wrapper("needs retry", "buyer-1", "msg-1")
        consumer = MessageConsumer("queue", max_concurrent=1)
        consumer.add_handler(FailingHandler())

        processed = await consumer._process_message(wrapper)
        assert processed is False

        requeued = await queue.retry_or_dead_letter(wrapper)
        assert requeued is True
        assert wrapper.retry_count == 1
        assert queue.size() == 1
        assert queue.get_stats().total_requeued == 1

        retry_wrapper = await queue.get(timeout=0.01)
        processed = await consumer._process_message(retry_wrapper)
        assert processed is False

        requeued = await queue.retry_or_dead_letter(retry_wrapper)
        assert requeued is False
        assert queue.size() == 0
        assert queue.get_stats().total_dead_lettered == 1

    asyncio.run(run())


def test_handler_exception_logs_are_sanitized(monkeypatch):
    async def run():
        consumer = MessageConsumer("queue", max_concurrent=1)
        consumer.add_handler(RaisingHandler())
        wrapper = _wrapper("needs retry", "buyer-1", "msg-1")

        logs = []

        class FakeLogger:
            def debug(self, message):
                logs.append(str(message))

            def warning(self, message):
                logs.append(str(message))

            def error(self, message):
                logs.append(str(message))

        consumer.logger = FakeLogger()

        await consumer._process_message(wrapper)

        joined = "\n".join(logs)
        assert "secret-token" not in joined
        assert "token=***" in joined

    asyncio.run(run())


def test_stop_all_removes_stopped_consumers_from_manager():
    async def run():
        manager = MessageConsumerManager()
        first = manager.create_consumer("queue", max_concurrent=1)

        await manager.stop_all()
        second = manager.create_consumer("queue", max_concurrent=1)

        assert manager.list_consumers() == ["queue"]
        assert second is not first

    asyncio.run(run())


def test_stop_waits_for_retry_task_scheduled_by_processing_callback():
    async def run():
        consumer = MessageConsumer("queue", max_concurrent=1)
        process_release = asyncio.Event()
        retry_started = asyncio.Event()
        retry_release = asyncio.Event()
        retry_finished = asyncio.Event()

        class SlowRetryQueue:
            async def retry_or_dead_letter(self, _wrapper):
                retry_started.set()
                await retry_release.wait()
                retry_finished.set()
                return True

        async def delayed_failure():
            await process_release.wait()
            return False

        wrapper = _wrapper("needs retry", "buyer-1", "msg-1")
        process_task = asyncio.create_task(delayed_failure())
        consumer._tasks.add(process_task)
        process_task.add_done_callback(consumer._tasks.discard)
        process_task.add_done_callback(
            lambda done_task: consumer._schedule_retry_if_needed(SlowRetryQueue(), wrapper, done_task)
        )

        stop_task = asyncio.create_task(consumer.stop())
        await asyncio.sleep(0)
        process_release.set()
        await retry_started.wait()
        await asyncio.sleep(0)

        assert not stop_task.done()

        retry_release.set()
        await asyncio.wait_for(stop_task, timeout=0.2)
        assert retry_finished.is_set()
        assert consumer._tasks == set()

    asyncio.run(run())


def test_consumer_marks_successful_message_done_for_queue_join():
    async def run():
        queue_name = "consumer-join-success"
        queue = queue_manager.recreate_queue(
            queue_name,
            QueueConfig(max_size=10, enable_deduplication=False),
        )
        consumer = MessageConsumer(queue_name, max_concurrent=1)
        handler = FastHandler()
        consumer.add_handler(handler)

        await queue.put(_wrapper("hello", "buyer-1", "msg-1").context)
        await consumer.start()
        try:
            deadline = time.time() + 0.5
            while not handler.handled and time.time() < deadline:
                await asyncio.sleep(0.01)

            assert handler.handled == ["hello"]
            await asyncio.wait_for(queue._queue.join(), timeout=0.1)
        finally:
            await consumer.stop()
            queue.close()
            queue_manager._queues.pop(queue_name, None)

    asyncio.run(run())


def test_consumer_marks_failed_retry_and_dead_letter_messages_done_for_queue_join():
    async def run():
        queue_name = "consumer-join-failure"
        queue = queue_manager.recreate_queue(
            queue_name,
            QueueConfig(max_size=10, enable_deduplication=False, max_processing_retries=1),
        )
        consumer = MessageConsumer(queue_name, max_concurrent=1)
        consumer.add_handler(FailingHandler())

        await queue.put(_wrapper("needs retry", "buyer-1", "msg-1").context)
        await consumer.start()
        try:
            deadline = time.time() + 0.5
            while queue.get_stats().total_dead_lettered < 1 and time.time() < deadline:
                await asyncio.sleep(0.01)

            stats = queue.get_stats()
            assert stats.total_requeued == 1
            assert stats.total_dead_lettered == 1
            await asyncio.wait_for(queue._queue.join(), timeout=0.1)
        finally:
            await consumer.stop()
            queue.close()
            queue_manager._queues.pop(queue_name, None)

    asyncio.run(run())
