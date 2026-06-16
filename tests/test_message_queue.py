import asyncio

from Message.core.queue import QueueManager
from Message.models.queue_models import QueueConfig
from bridge.context import ChannelType, Context, ContextType


def _context(content, msg_id, timestamp=None):
    return Context.create_pinduoduo_context(
        content=content,
        msg_id=msg_id,
        from_uid="buyer-1",
        user_msg_type=ContextType.TEXT,
        shop_id="shop-1",
        user_id="seller-1",
        timestamp=timestamp,
        channel_type=ChannelType.PINDUODUO,
    )


def _mapping_context(content, msg_id, timestamp=None):
    return Context(
        type=ContextType.TEXT,
        content=content,
        channel_type=ChannelType.PINDUODUO,
        kwargs={
            "msg_id": msg_id,
            "shop_id": "shop-1",
            "user_id": "seller-1",
            "from_uid": "buyer-1",
            "timestamp": timestamp,
        },
    )


def test_recreate_queue_migrates_unconsumed_messages():
    async def run():
        manager = QueueManager()
        queue = manager.get_or_create_queue(
            "queue",
            QueueConfig(max_size=10, enable_deduplication=False),
        )
        await queue.put(_context("第一条", "msg-1"))
        await queue.put(_context("第二条", "msg-2"))

        new_queue = manager.recreate_queue(
            "queue",
            QueueConfig(max_size=10, enable_deduplication=False),
        )

        assert new_queue.size() == 2
        first = await new_queue.get(timeout=0.01)
        second = await new_queue.get(timeout=0.01)
        assert first.context.content == "第一条"
        assert second.context.content == "第二条"

    asyncio.run(run())


def test_recreate_queue_preserves_cumulative_stats():
    async def run():
        manager = QueueManager()
        queue = manager.get_or_create_queue(
            "queue",
            QueueConfig(max_size=10, enable_deduplication=False),
        )
        await queue.put(_context("第一条", "msg-1"))
        wrapper = await queue.get(timeout=0.01)
        await queue.retry_or_dead_letter(wrapper)
        retry_wrapper = await queue.get(timeout=0.01)
        retry_wrapper.retry_count = queue.config.max_processing_retries
        await queue.retry_or_dead_letter(retry_wrapper)

        new_queue = manager.recreate_queue(
            "queue",
            QueueConfig(max_size=10, enable_deduplication=False),
        )
        stats = new_queue.get_stats()

        assert stats.total_enqueued == 1
        assert stats.total_dequeued == 2
        assert stats.total_requeued == 1
        assert stats.total_dead_lettered == 1
        assert stats.current_size == 0

    asyncio.run(run())


def test_recreate_queue_preserves_deduplication_cache_for_migrated_messages():
    async def run():
        manager = QueueManager()
        queue = manager.get_or_create_queue(
            "queue",
            QueueConfig(max_size=10, enable_deduplication=True),
        )
        first_id = await queue.put(_context("第一条", "msg-1"))

        new_queue = manager.recreate_queue(
            "queue",
            QueueConfig(max_size=10, enable_deduplication=True),
        )
        second_id = await new_queue.put(_context("第一条", "msg-1"))

        assert second_id == first_id
        assert new_queue.size() == 1
        assert new_queue.get_stats().total_enqueued == 1

    asyncio.run(run())


def test_full_queue_rejection_does_not_increment_enqueued_stats():
    async def run():
        manager = QueueManager()
        queue = manager.get_or_create_queue(
            "queue",
            QueueConfig(max_size=1, enable_deduplication=False),
        )
        await queue.put(_context("第一条", "msg-1"))

        try:
            await queue.put(_context("第二条", "msg-2"))
        except RuntimeError as exc:
            assert "full" in str(exc).lower()
        else:
            raise AssertionError("expected queue full error")

        stats = queue.get_stats()
        assert stats.total_enqueued == 1
        assert stats.current_size == 1

    asyncio.run(run())


def test_missing_msg_id_uses_stable_fallback_dedup_key():
    async def run():
        manager = QueueManager()
        queue = manager.get_or_create_queue(
            "queue",
            QueueConfig(max_size=10, enable_deduplication=True),
        )

        first_id = await queue.put(_context("你好", "", timestamp="2026-06-13 10:00:00"))
        second_id = await queue.put(_context("你好", "", timestamp="2026-06-13 10:00:00"))

        assert second_id == first_id
        assert queue.size() == 1
        assert queue.get_stats().total_enqueued == 1

    asyncio.run(run())


def test_duplicate_message_returns_original_enqueued_message_id():
    async def run():
        manager = QueueManager()
        queue = manager.get_or_create_queue(
            "queue",
            QueueConfig(max_size=10, enable_deduplication=True),
        )

        first_id = await queue.put(_context("你好", "platform-msg-1"))
        second_id = await queue.put(_context("你好", "platform-msg-1"))

        assert second_id == first_id
        assert queue.size() == 1
        wrapper = await queue.get(timeout=0.01)
        assert wrapper.message_id == first_id

    asyncio.run(run())


def test_duplicate_message_is_accepted_when_queue_is_full():
    async def run():
        manager = QueueManager()
        queue = manager.get_or_create_queue(
            "queue",
            QueueConfig(max_size=1, enable_deduplication=True),
        )

        first_id = await queue.put(_context("你好", "platform-msg-1"))
        duplicate_id = await queue.put(_context("你好", "platform-msg-1"))

        assert duplicate_id == first_id
        assert queue.size() == 1
        assert queue.get_stats().total_enqueued == 1

    asyncio.run(run())


def test_full_queue_rejection_does_not_register_dedup_key_for_unenqueued_message():
    async def run():
        manager = QueueManager()
        queue = manager.get_or_create_queue(
            "queue",
            QueueConfig(max_size=1, enable_deduplication=True),
        )
        await queue.put(_context("第一条", "platform-msg-1"))

        try:
            await queue.put(_context("第二条", "platform-msg-2"))
        except RuntimeError as exc:
            assert "full" in str(exc).lower()
        else:
            raise AssertionError("expected queue full error")

        wrapper = await queue.get(timeout=0.01)
        queue.task_done()
        assert wrapper.context.content == "第一条"

        second_id = await queue.put(_context("第二条", "platform-msg-2"))
        assert second_id != wrapper.message_id
        assert queue.size() == 1

        second_wrapper = await queue.get(timeout=0.01)
        assert second_wrapper.context.content == "第二条"

    asyncio.run(run())


def test_duplicate_message_deduplicates_mapping_kwargs():
    async def run():
        manager = QueueManager()
        queue = manager.get_or_create_queue(
            "queue",
            QueueConfig(max_size=10, enable_deduplication=True),
        )

        first_id = await queue.put(_mapping_context("你好", "platform-msg-1"))
        second_id = await queue.put(_mapping_context("你好", "platform-msg-1"))

        assert second_id == first_id
        assert queue.size() == 1

    asyncio.run(run())


def test_missing_msg_id_does_not_drop_same_text_at_different_timestamps():
    async def run():
        manager = QueueManager()
        queue = manager.get_or_create_queue(
            "queue",
            QueueConfig(max_size=10, enable_deduplication=True),
        )

        await queue.put(_context("你好", "", timestamp="2026-06-13 10:00:00"))
        await queue.put(_context("你好", "", timestamp="2026-06-13 10:00:01"))

        assert queue.size() == 2
        assert queue.get_stats().total_enqueued == 2

    asyncio.run(run())


def test_missing_msg_id_and_timestamp_does_not_deduplicate_by_content_only():
    async def run():
        manager = QueueManager()
        queue = manager.get_or_create_queue(
            "queue",
            QueueConfig(max_size=10, enable_deduplication=True),
        )

        first_id = await queue.put(_context("你好", ""))
        second_id = await queue.put(_context("你好", ""))

        assert second_id != first_id
        assert queue.size() == 2
        assert queue.get_stats().total_enqueued == 2

    asyncio.run(run())


def test_close_all_allows_queue_to_be_recreated():
    async def run():
        manager = QueueManager()
        queue = manager.get_or_create_queue(
            "queue",
            QueueConfig(max_size=10, enable_deduplication=False),
        )
        await queue.put(_context("第一条", "msg-1"))

        await manager.close_all()
        new_queue = manager.get_or_create_queue(
            "queue",
            QueueConfig(max_size=10, enable_deduplication=False),
        )
        message_id = await new_queue.put(_context("第二条", "msg-2"))

        assert new_queue is not queue
        assert message_id
        assert new_queue.size() == 1

    asyncio.run(run())


def test_close_all_marks_pending_messages_done_for_queue_join():
    async def run():
        manager = QueueManager()
        queue = manager.get_or_create_queue(
            "queue",
            QueueConfig(max_size=10, enable_deduplication=False),
        )
        await queue.put(_context("第一条", "msg-1"))
        old_inner_queue = queue._queue

        await manager.close_all()

        await asyncio.wait_for(old_inner_queue.join(), timeout=0.01)

    asyncio.run(run())


def test_clear_marks_drained_messages_done_for_queue_join():
    async def run():
        manager = QueueManager()
        queue = manager.get_or_create_queue(
            "queue",
            QueueConfig(max_size=10, enable_deduplication=False),
        )
        await queue.put(_context("第一条", "msg-1"))

        await queue.clear()

        await asyncio.wait_for(queue._queue.join(), timeout=0.01)
        assert queue.size() == 0

    asyncio.run(run())


def test_recreate_queue_marks_old_drained_messages_done_for_queue_join():
    async def run():
        manager = QueueManager()
        queue = manager.get_or_create_queue(
            "queue",
            QueueConfig(max_size=10, enable_deduplication=False),
        )
        await queue.put(_context("第一条", "msg-1"))
        old_inner_queue = queue._queue

        manager.recreate_queue(
            "queue",
            QueueConfig(max_size=10, enable_deduplication=False),
        )

        await asyncio.wait_for(old_inner_queue.join(), timeout=0.01)

    asyncio.run(run())
