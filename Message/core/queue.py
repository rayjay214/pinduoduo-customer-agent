"""
简化的消息队列实现
只支持FIFO队列，移除未使用的复杂功能
"""

import asyncio
import hashlib
import time
from collections.abc import Mapping
from typing import Optional, Dict
from utils.logger_loguru import get_logger

from ..models.queue_models import MessageWrapper, QueueStats, QueueConfig
from bridge.context import Context


logger = get_logger(__name__)


class SimpleMessageQueue:
    """简化的消息队列 - 只支持FIFO"""

    def __init__(self, name: str, config: QueueConfig):
        self.name = name
        self.config = config
        self.logger = get_logger(f"Queue.{name}")

        # 基本队列
        self._queue = asyncio.Queue(maxsize=config.max_size)
        self._stats = QueueStats()
        self._closed = False

        # 去重缓存（可选）
        self._deduplication_cache: Dict[str, str] = {} if config.enable_deduplication else None
        self._last_cleanup_time = time.time()

    async def put(self, context: Context) -> str:
        """放入消息"""
        if self._closed:
            raise RuntimeError("Queue is closed")

        message_wrapper = MessageWrapper(
            message_id="",  # 将在__post_init__中生成
            context=context,
            timestamp=time.time()
        )

        dedup_key = self._deduplication_key(message_wrapper)
        if dedup_key and self._deduplication_cache is not None:
            duplicate_message_id = self._deduplication_cache.get(dedup_key)
        else:
            duplicate_message_id = None
        if duplicate_message_id:
            self.logger.debug(
                f"Message deduplicated: new={message_wrapper.message_id}, original={duplicate_message_id}"
            )
            return duplicate_message_id

        if self._queue.full():
            raise RuntimeError("Queue is full")

        await self._queue.put(message_wrapper)
        self._register_deduplication_key(dedup_key, message_wrapper.message_id)
        self._stats.enqueue()
        self.logger.debug(f"Message enqueued: {message_wrapper.message_id}")
        return message_wrapper.message_id

    async def get(self, timeout: Optional[float] = None) -> Optional[MessageWrapper]:
        """获取消息"""
        if self._closed and self._queue.empty():
            return None

        try:
            if timeout:
                wrapper = await asyncio.wait_for(self._queue.get(), timeout)
            else:
                wrapper = await self._queue.get()

            self._stats.dequeue()
            self.logger.debug(f"Message dequeued: {wrapper.message_id}")
            return wrapper

        except asyncio.TimeoutError:
            return None

    def task_done(self) -> None:
        """标记最近一次 get() 取出的消息已完成处理。"""
        self._queue.task_done()

    def size(self) -> int:
        """获取队列大小"""
        return self._queue.qsize()

    def is_empty(self) -> bool:
        """检查队列是否为空"""
        return self._queue.empty()

    def get_stats(self) -> QueueStats:
        """获取统计信息"""
        stats = QueueStats(
            total_enqueued=self._stats.total_enqueued,
            total_dequeued=self._stats.total_dequeued,
            total_requeued=self._stats.total_requeued,
            total_dead_lettered=self._stats.total_dead_lettered,
            current_size=self.size(),
            last_activity=self._stats.last_activity
        )
        return stats

    def close(self):
        """关闭队列"""
        self._closed = True
        self.logger.info(f"Queue {self.name} closed")

    def drain_pending(self) -> list[MessageWrapper]:
        """取出尚未消费的消息，用于队列重建时迁移，避免静默丢消息。"""
        pending: list[MessageWrapper] = []
        while True:
            try:
                pending.append(self._queue.get_nowait())
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        return pending

    async def put_wrapper(self, wrapper: MessageWrapper) -> None:
        """迁移已有消息包装器，不重新生成 message_id 或再次执行去重。"""
        if self._closed:
            raise RuntimeError("Queue is closed")
        if self._queue.full():
            raise RuntimeError("Queue is full")
        await self._queue.put(wrapper)
        self._stats.enqueue()

    async def retry_or_dead_letter(self, wrapper: MessageWrapper) -> bool:
        """处理失败后重入队；超过上限后进入死信记录，避免无限重试或静默丢失。"""
        if self._closed:
            self.logger.warning(f"Queue closed; message dead-lettered: {wrapper.message_id}")
            self._stats.dead_letter()
            return False
        if wrapper.retry_count >= self.config.max_processing_retries:
            self.logger.error(
                f"Message exceeded max processing retries and was dead-lettered: "
                f"{wrapper.message_id}, retries={wrapper.retry_count}"
            )
            self._stats.dead_letter()
            return False
        if self._queue.full():
            self.logger.error(f"Queue full; failed message dead-lettered: {wrapper.message_id}")
            self._stats.dead_letter()
            return False

        wrapper.retry_count += 1
        await self._queue.put(wrapper)
        self._stats.requeue()
        self.logger.warning(
            f"Message requeued after processing failure: {wrapper.message_id}, "
            f"retry_count={wrapper.retry_count}"
        )
        return True

    def _deduplication_key(self, wrapper: MessageWrapper) -> Optional[str]:
        """计算稳定去重键；没有平台标识或时间戳时不做内容去重。"""
        if self._deduplication_cache is None:
            return None

        kwargs = getattr(wrapper.context, "kwargs", None)
        msg_id = str(self._kwarg_value(kwargs, "msg_id", "") or "").strip()
        shop_id = str(self._kwarg_value(kwargs, "shop_id", "") or "").strip()
        user_id = str(self._kwarg_value(kwargs, "user_id", "") or "").strip()
        from_uid = str(self._kwarg_value(kwargs, "from_uid", "") or "").strip()
        timestamp = str(self._kwarg_value(kwargs, "timestamp", "") or "").strip()
        context_type = str(getattr(wrapper.context, "type", "") or "").strip()
        content_str = str(wrapper.context.content) if wrapper.context.content is not None else ""

        # PDD has many customers asking identical short questions ("发什么快递",
        # "有没有运费险"). Deduplicating only by content drops legitimate messages.
        if not msg_id and not timestamp:
            return None
        identity = msg_id if msg_id else f"fallback:{timestamp}:{content_str}"
        dedup_source = "|".join(
            part
            for part in (
                shop_id,
                user_id,
                from_uid,
                context_type,
                identity,
            )
            if part
        )
        return hashlib.md5(dedup_source.encode("utf-8")).hexdigest()

    def _register_deduplication_key(self, dedup_key: Optional[str], message_id: str) -> None:
        """仅在消息成功入队后登记去重缓存，避免队满拒绝污染缓存。"""
        if not dedup_key or self._deduplication_cache is None:
            return
        self._deduplication_cache[dedup_key] = message_id
        self._cleanup_deduplication_cache()

    @staticmethod
    def _kwarg_value(kwargs, key: str, default=None):
        if isinstance(kwargs, Mapping):
            return kwargs.get(key, default)
        return getattr(kwargs, key, default)

    def _cleanup_deduplication_cache(self):
        """清理过期的去重缓存，限制最大条目数避免内存泄漏"""
        current_time = time.time()
        # 定期清空缓存
        if current_time - self._last_cleanup_time > self.config.deduplication_window:
            self._deduplication_cache.clear()
            self._last_cleanup_time = current_time
            self.logger.debug("Deduplication cache cleaned (time window)")
            return

        # 限制最大缓存条目数（高并发保护）
        MAX_DEDUP_ENTRIES = 10000
        if len(self._deduplication_cache) > MAX_DEDUP_ENTRIES:
            # 清空一半以平衡内存和去重效果
            self._deduplication_cache.clear()
            self._last_cleanup_time = current_time
            self.logger.warning(f"Deduplication cache exceeded {MAX_DEDUP_ENTRIES} entries, cleared")

    async def clear(self):
        """清空队列"""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        self.logger.info(f"Queue {self.name} cleared")


class QueueManager:
    """队列管理器 - 简化版"""

    def __init__(self):
        self._queues: Dict[str, SimpleMessageQueue] = {}
        self.logger = get_logger("QueueManager")

    def get_or_create_queue(self, name: str, config: Optional[QueueConfig] = None) -> SimpleMessageQueue:
        """获取或创建队列"""
        if name not in self._queues:
            if config is None:
                config = QueueConfig()
            queue = SimpleMessageQueue(name, config)
            self._queues[name] = queue
            self.logger.debug(f"Created queue: {name}")
        return self._queues[name]

    def get_queue(self, name: str) -> Optional[SimpleMessageQueue]:
        """获取队列"""
        return self._queues.get(name)

    def recreate_queue(self, name: str, config: Optional[QueueConfig] = None) -> SimpleMessageQueue:
        """重新创建队列以绑定当前事件循环"""
        pending_messages = []
        deduplication_cache = None
        last_cleanup_time = time.time()
        previous_stats = None
        try:
            old = self._queues.get(name)
            if old:
                pending_messages = old.drain_pending()
                previous_stats = old.get_stats()
                if old._deduplication_cache is not None:
                    deduplication_cache = dict(old._deduplication_cache)
                    last_cleanup_time = old._last_cleanup_time
                old.close()
                self._queues.pop(name, None)
        except Exception:
            self.logger.exception(f"Drain old queue failed before recreate: {name}")
        if config is None:
            config = QueueConfig()
        required_size = max(config.max_size, len(pending_messages) or config.max_size)
        queue = SimpleMessageQueue(
            name,
            QueueConfig(
                max_size=required_size,
                enable_deduplication=config.enable_deduplication,
                deduplication_window=config.deduplication_window,
                max_processing_retries=config.max_processing_retries,
            ),
        )
        if previous_stats is not None:
            queue._stats.total_enqueued = previous_stats.total_enqueued
            queue._stats.total_dequeued = previous_stats.total_dequeued
            queue._stats.total_requeued = previous_stats.total_requeued
            queue._stats.total_dead_lettered = previous_stats.total_dead_lettered
            queue._stats.last_activity = previous_stats.last_activity
        for wrapper in pending_messages:
            queue._queue.put_nowait(wrapper)
            queue._stats.current_size += 1
        if deduplication_cache is not None and queue._deduplication_cache is not None:
            queue._deduplication_cache = deduplication_cache
            queue._last_cleanup_time = last_cleanup_time
        self._queues[name] = queue
        self.logger.info(f"Recreated queue: {name}, migrated_pending={len(pending_messages)}")
        return queue

    def list_queues(self) -> Dict[str, QueueStats]:
        """列出所有队列及其统计信息"""
        return {name: queue.get_stats() for name, queue in self._queues.items()}

    async def close_all(self):
        """关闭所有队列"""
        for queue in self._queues.values():
            queue.close()
            queue.drain_pending()
        self._queues.clear()
        self.logger.info("All queues closed")


# 全局队列管理器实例
queue_manager = QueueManager()
