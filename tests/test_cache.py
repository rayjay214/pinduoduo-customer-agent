import asyncio

from core.cache import MemoryCache


def test_memory_cache_can_be_created_without_running_loop():
    cache = MemoryCache(cleanup_interval=60)

    assert cache._cleanup_task is None


def test_memory_cache_starts_cleanup_when_used_in_running_loop():
    async def run():
        cache = MemoryCache(cleanup_interval=60)

        await cache.set("key", "value")

        assert cache._cleanup_task is not None
        assert not cache._cleanup_task.done()

        await cache.close()

    asyncio.run(run())


def test_memory_cache_close_does_not_restart_cleanup_task():
    async def run():
        cache = MemoryCache(cleanup_interval=60)

        await cache.set("key", "value")
        await cache.close()

        assert cache._cleanup_task is None

    asyncio.run(run())


def test_memory_cache_stats_tolerate_zero_max_size():
    async def run():
        cache = MemoryCache(max_size=0, cleanup_interval=0)

        stats = await cache.get_stats()

        assert stats["max_size"] == 1
        assert stats["usage_ratio"] == 0

    asyncio.run(run())


def test_memory_cache_set_sanitizes_exception_logs():
    class BrokenTTL:
        def __radd__(self, other):
            raise RuntimeError("Authorization: Bearer secret-token")

    class CapturingLogger:
        def __init__(self):
            self.messages = []

        def debug(self, *args, **kwargs):
            pass

        def error(self, message, *args, **kwargs):
            self.messages.append(message)

    async def run():
        cache = MemoryCache(cleanup_interval=0)
        cache.logger = CapturingLogger()

        result = await cache.set("account token=key-secret", "value", ttl=BrokenTTL())

        assert result is False
        assert cache.logger.messages
        assert "secret-token" not in cache.logger.messages[-1]
        assert "key-secret" not in cache.logger.messages[-1]
        assert "Authorization" in cache.logger.messages[-1]

    asyncio.run(run())
