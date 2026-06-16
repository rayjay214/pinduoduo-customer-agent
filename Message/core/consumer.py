"""
简化的消息消费者实现
移除复杂的用户隔离机制，保持核心功能
"""

import asyncio
from collections.abc import Mapping
from typing import List, Dict, Any
from utils.logger_loguru import get_logger
from Channel.pinduoduo.utils.base_request import BaseRequest
from bridge.context import Context
from .queue import queue_manager
from .handlers import MessageHandler
from ..models.queue_models import MessageWrapper


logger = get_logger(__name__)


def _sanitize_for_log(value: Any) -> Any:
    if isinstance(value, BaseException):
        text = str(value)
        return f"{type(value).__name__}: {BaseRequest()._sanitize_for_log(text)}" if text else type(value).__name__
    return BaseRequest()._sanitize_for_log(value)


class MessageConsumer:
    """消息消费者 - 简化版"""

    def __init__(self, queue_name: str, max_concurrent: int = 10):
        self.queue_name = queue_name
        self.max_concurrent = max(1, int(max_concurrent or 1))
        self.handlers: List[MessageHandler] = []
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.running = False
        self.consumer_task = None
        self._tasks: set = set()
        self._user_locks: Dict[str, asyncio.Lock] = {}
        self.logger = get_logger(f"Consumer.{queue_name}")

    def add_handler(self, handler: MessageHandler):
        """添加处理器"""
        self.handlers.append(handler)
        self.logger.debug(f"Added handler: {handler.__class__.__name__}")

    def is_running(self) -> bool:
        """检查消费者是否正在运行"""
        return self.running

    async def start(self):
        """启动消费者"""
        if self.running:
            self.logger.warning(f"Consumer {self.queue_name} is already running")
            return

        self.running = True
        self.consumer_task = asyncio.create_task(self._consume_loop())
        self.logger.info(f"Consumer {self.queue_name} started")

    async def _consume_loop(self):
        """消费循环"""
        queue = queue_manager.get_or_create_queue(self.queue_name)

        try:
            while self.running:
                try:
                    wrapper = await queue.get(timeout=1.0)
                    if wrapper:
                        # 使用信号量控制并发数，跟踪任务以便优雅停止
                        task = asyncio.create_task(self._process_message(wrapper))
                        self._tasks.add(task)
                        task.add_done_callback(self._tasks.discard)
                        task.add_done_callback(
                            lambda done_task, q=queue, w=wrapper: self._schedule_retry_if_needed(q, w, done_task)
                        )
                except Exception as e:
                    self.logger.error(f"Consumer error: {_sanitize_for_log(e)}")
                    await asyncio.sleep(0.1)
        finally:
            self.logger.info(f"Consumer {self.queue_name} stopped")

    async def stop(self):
        """停止消费者"""
        self.running = False

        # 取消消费任务
        if getattr(self, "consumer_task", None):
            self.consumer_task.cancel()
            try:
                await self.consumer_task
            except asyncio.CancelledError:
                pass
            finally:
                self.consumer_task = None

        # 等待所有正在处理的任务完成；处理任务完成时可能在回调里追加 retry/dead-letter 任务。
        while self._tasks:
            pending = [task for task in self._tasks if not task.done()]
            if not pending:
                self._tasks.clear()
                break
            await asyncio.gather(*pending, return_exceptions=True)
            await asyncio.sleep(0)

    def _schedule_retry_if_needed(self, queue, wrapper: MessageWrapper, task: asyncio.Task) -> None:
        """处理任务完成后确认原始出队项，失败时先安排重试/死信。"""
        try:
            processed = task.result()
        except asyncio.CancelledError:
            self._mark_queue_task_done(queue, wrapper.message_id)
            return
        except Exception as exc:
            self.logger.error(f"Message task failed unexpectedly: {wrapper.message_id}, {_sanitize_for_log(exc)}")
            processed = False
        if processed:
            self._mark_queue_task_done(queue, wrapper.message_id)
            return
        retry_task = asyncio.create_task(self._retry_or_dead_letter_then_ack(queue, wrapper))
        self._tasks.add(retry_task)
        retry_task.add_done_callback(self._tasks.discard)
        retry_task.add_done_callback(
            lambda done_task, message_id=wrapper.message_id: self._log_retry_task_failure(message_id, done_task)
        )

    async def _retry_or_dead_letter_then_ack(self, queue, wrapper: MessageWrapper) -> bool:
        try:
            return await queue.retry_or_dead_letter(wrapper)
        finally:
            self._mark_queue_task_done(queue, wrapper.message_id)

    def _mark_queue_task_done(self, queue, message_id: str) -> None:
        task_done = getattr(queue, "task_done", None)
        if not callable(task_done):
            return
        try:
            task_done()
        except ValueError as exc:
            self.logger.error(f"Queue task_done failed: {message_id}, {_sanitize_for_log(exc)}")

    def _log_retry_task_failure(self, message_id: str, task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.logger.error(f"Retry/dead-letter task failed: {message_id}, {_sanitize_for_log(exc)}")

    async def _process_message(self, wrapper: MessageWrapper) -> bool:
        """处理单个消息"""
        lock_key = self._extract_conversation_key(wrapper.context)
        lock = self._user_locks.setdefault(lock_key, asyncio.Lock())
        processed = False
        try:
            async with lock:
                async with self.semaphore:
                    metadata = wrapper.to_metadata()
                    # 追加渠道上下文到metadata，供发送使用
                    try:
                        kwargs = getattr(wrapper.context, 'kwargs', None)
                        if kwargs:
                            metadata['shop_id'] = self._kwarg_value(kwargs, 'shop_id')
                            metadata['user_id'] = self._kwarg_value(kwargs, 'user_id')
                            metadata['from_uid'] = self._kwarg_value(kwargs, 'from_uid')
                    except Exception as exc:
                        self.logger.debug(
                            f"Failed to append channel metadata for message {wrapper.message_id}: {_sanitize_for_log(exc)}"
                        )
                    # 保留用于日志的用户键
                    metadata['user_key'] = self._extract_user_id(wrapper.context)

                    for handler in self.handlers:
                        try:
                            if handler.can_handle(wrapper.context):
                                success = await handler.handle(wrapper.context, metadata)
                                if success:
                                    processed = True
                                    self.logger.debug(f"Message {wrapper.message_id} handled by {handler.__class__.__name__}")
                                    break
                        except Exception as e:
                            self.logger.error(f"Handler {handler.__class__.__name__} error: {_sanitize_for_log(e)}")
                            # 尝试下一个处理器
                            continue

                    if not processed:
                        self.logger.warning(f"Message {wrapper.message_id} not processed by any handler")

        except Exception as e:
            self.logger.error(f"Failed to process message {wrapper.message_id}: {_sanitize_for_log(e)}")
            processed = False
        finally:
            waiters = getattr(lock, "_waiters", None) or []
            has_waiters = any(not waiter.done() for waiter in waiters)
            if not lock.locked() and not has_waiters:
                self._user_locks.pop(lock_key, None)
        return processed

    def _extract_conversation_key(self, context: Context) -> str:
        """按店铺账号+顾客 UID 串行化，避免同一会话上下文被并发打乱。"""
        try:
            kwargs = getattr(context, 'kwargs', None)
            channel = context.channel_type
            channel_str = str(channel.value if hasattr(channel, 'value') else channel or "unknown")
            shop_id = str(self._kwarg_value(kwargs, 'shop_id') or "unknown_shop")
            user_id = str(self._kwarg_value(kwargs, 'user_id') or "unknown_user")
            from_uid = str(self._kwarg_value(kwargs, 'from_uid') or "unknown_customer")
            return f"{channel_str}:{shop_id}:{user_id}:{from_uid}"
        except Exception as e:
            self.logger.error(f"Failed to extract conversation key: {_sanitize_for_log(e)}")
            return self._extract_user_id(context)

    @staticmethod
    def _kwarg_value(kwargs, key: str, default=None):
        if isinstance(kwargs, Mapping):
            return kwargs.get(key, default)
        return getattr(kwargs, key, default)

    def _extract_user_id(self, context: Context) -> str:
        """提取用户ID"""
        try:
            kwargs = getattr(context, 'kwargs', None)
            from_uid = self._kwarg_value(kwargs, 'from_uid') if kwargs is not None else None
            channel = context.channel_type

            # 处理可能的None值
            if from_uid is None:
                from_uid = "unknown"
            if channel is None:
                channel = "unknown"

            # 处理channel可能是字符串或枚举对象的情况
            if hasattr(channel, 'value'):
                channel_str = str(channel.value)
            else:
                channel_str = str(channel)

            return f"{channel_str}_{from_uid}"
        except Exception as e:
            self.logger.error(f"Failed to extract user ID: {_sanitize_for_log(e)}")
            return "unknown_unknown"


class MessageConsumerManager:
    """消息消费者管理器"""

    def __init__(self):
        self._consumers: Dict[str, MessageConsumer] = {}
        self.logger = get_logger("ConsumerManager")

    def create_consumer(self, queue_name: str, max_concurrent: int = 10) -> MessageConsumer:
        """创建消费者"""
        if queue_name in self._consumers:
            self.logger.warning(f"Consumer {queue_name} already exists")
            return self._consumers[queue_name]

        consumer = MessageConsumer(queue_name, max_concurrent)
        self._consumers[queue_name] = consumer
        self.logger.info(f"Created consumer: {queue_name}")
        return consumer

    def get_consumer(self, queue_name: str) -> MessageConsumer:
        """获取消费者"""
        return self._consumers.get(queue_name)

    async def start_consumer(self, queue_name: str):
        """启动消费者"""
        consumer = self.get_consumer(queue_name)
        if consumer:
            await consumer.start()
        else:
            self.logger.error(f"Consumer {queue_name} not found")

    async def stop_consumer(self, queue_name: str):
        """停止消费者"""
        consumer = self.get_consumer(queue_name)
        if consumer:
            await consumer.stop()
            self._consumers.pop(queue_name, None)
        else:
            self.logger.error(f"Consumer {queue_name} not found")

    def list_consumers(self) -> List[str]:
        """列出所有消费者"""
        return list(self._consumers.keys())

    async def stop_all(self):
        """停止所有消费者"""
        for consumer in list(self._consumers.values()):
            await consumer.stop()
        self._consumers.clear()
        self.logger.info("All consumers stopped")


# 全局消费者管理器实例
message_consumer_manager = MessageConsumerManager()
