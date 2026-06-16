# 生命周期管理模块
import asyncio
import time
import websockets
from websockets import exceptions as ws_exceptions
from typing import Optional, Any
from urllib.parse import urlencode
from utils.logger_loguru import get_logger
from Channel.pinduoduo.utils.API.get_token import GetToken
from core.base_service import _sanitize_for_log
from config import config


class LifecycleMixin:
    """生命周期管理 Mixin"""

    @staticmethod
    def _sanitize_for_log(value):
        return _sanitize_for_log(value)

    @staticmethod
    def _connection_key(shop_id: str, user_id: str) -> str:
        return f"{shop_id}_{user_id}"

    @staticmethod
    def _build_queue_name(shop_id: str, user_id: str) -> str:
        """按账号隔离消息队列/消费者，避免同店多账号共享同一 asyncio 资源。"""
        return f"pdd_{LifecycleMixin._connection_key(shop_id, user_id)}"

    def _build_websocket_url(self, access_token: str) -> str:
        params = {
            "access_token": access_token,
            "role": "mall_cs",
            "client": "web",
            "version": self.API_VERSION,
        }
        return f"{self.base_url}?{urlencode(params)}"

    def _get_or_create_stop_event(self, connection_key: str) -> asyncio.Event:
        if not hasattr(self, "_stop_events"):
            self._stop_events = {}
        event = self._stop_events.get(connection_key)
        if event is None:
            event = asyncio.Event()
            self._stop_events[connection_key] = event
        self._stop_event = event
        return event

    def _is_stop_requested(self, shop_id: str, user_id: str) -> bool:
        connection_key = self._connection_key(shop_id, user_id)
        event = getattr(self, "_stop_events", {}).get(connection_key)
        if event is not None:
            return event.is_set()
        return bool(getattr(self, "_stop_event", None) and self._stop_event.is_set())

    def _set_active_websocket(self, connection_key: str, websocket) -> None:
        if not hasattr(self, "_websockets"):
            self._websockets = {}
        self._websockets[connection_key] = websocket
        self.ws = websocket

    async def _close_account_websocket(self, connection_key: str) -> None:
        websocket = getattr(self, "_websockets", {}).pop(connection_key, None)
        if websocket is not None:
            await self._safe_close_websocket(websocket)
        if getattr(self, "ws", None) is websocket:
            self.ws = None

    async def start_account(self, shop_id: str, user_id: str, on_success: callable, on_failure: callable):
        """启动指定店铺下账号"""
        account_info = db_manager.get_account(self.channel_name, shop_id, user_id)
        if not isinstance(account_info, dict):
            error_msg = f"账号 {user_id} 在数据库中不存在"
            self.logger.error(error_msg)
            on_failure(error_msg)
            return

        username = account_info.get("username", user_id)
        connection_key = self._connection_key(shop_id, user_id)
        self._get_or_create_stop_event(connection_key).clear()

        self.status_manager.update_status(shop_id, user_id, username, ConnectionState.CONNECTING)

        if connection_key in self._reconnect_tasks:
            self._reconnect_tasks[connection_key].cancel()
            del self._reconnect_tasks[connection_key]

        if self.reconnect_config.enable_auto_reconnect:
            connect_task = asyncio.create_task(
                self._connect_with_retry(shop_id, user_id, username, on_success, on_failure)
            )
        else:
            connect_task = asyncio.create_task(
                self._connect_single_attempt(shop_id, user_id, username, on_success, on_failure)
            )

        self._reconnect_tasks[connection_key] = connect_task

    async def stop_account(self, shop_id: str, user_id: str):
        """停止指定店铺下账号"""
        try:
            account_info = db_manager.get_account(self.channel_name, shop_id, user_id)
            if not isinstance(account_info, dict):
                self.logger.warning(f"账号 {user_id} 不存在，无法停止")
                return

            username = account_info.get("username", user_id)
            connection_key = self._connection_key(shop_id, user_id)

            self.logger.info(f"正在停止店铺 {shop_id} 账号 {username}")

            self._get_or_create_stop_event(connection_key).set()

            if connection_key in self._reconnect_tasks:
                task = self._reconnect_tasks[connection_key]
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=5.0)
                    except asyncio.CancelledError:
                        self.logger.debug(f"重连任务已被取消: {connection_key}")
                    except asyncio.TimeoutError:
                        self.logger.warning(f"重连任务取消超时: {connection_key}")
                    except Exception as task_error:
                        self.logger.error(f"等待重连任务完成时出错: {self._sanitize_for_log(task_error)}")
                del self._reconnect_tasks[connection_key]
                self.logger.debug(f"已清理重连任务: {connection_key}")

            if connection_key in self._heartbeat_tasks:
                task = self._heartbeat_tasks[connection_key]
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=3.0)
                    except asyncio.CancelledError:
                        self.logger.debug(f"心跳任务已被取消: {connection_key}")
                    except asyncio.TimeoutError:
                        self.logger.warning(f"心跳任务取消超时: {connection_key}")
                    except Exception as task_error:
                        self.logger.error(f"等待心跳任务完成时出错: {self._sanitize_for_log(task_error)}")
                del self._heartbeat_tasks[connection_key]
                self.logger.debug(f"已清理心跳任务: {connection_key}")

            self.status_manager.update_status(shop_id, user_id, username, ConnectionState.DISCONNECTED)

            if connection_key in getattr(self, "_websockets", {}):
                await self._close_account_websocket(connection_key)
                self.logger.info(f"已关闭店铺 {shop_id} 账号 {username} 的WebSocket连接")
            else:
                self.logger.warning(f"店铺 {shop_id} 账号 {username} 的WebSocket连接已经关闭或不存在")

            await self.cleanup_processing_tasks(connection_key)

            queue_name = self._build_queue_name(shop_id, user_id)
            await self._cleanup_resources(queue_name, connection_key=connection_key)

            self.logger.info(f"成功停止店铺 {shop_id} 账号 {username}")

        except Exception:
            self.logger.exception(f"停止店铺 {shop_id} 账号 {user_id} 时发生错误")

    async def init(self, shop_id: str, user_id: str, username: str, on_success: callable, on_failure: callable):
        """初始化WebSocket连接和消息处理系统"""
        connection_key = self._connection_key(shop_id, user_id)
        try:
            stop_event = self._get_or_create_stop_event(connection_key)
            stop_event.clear()

            token = GetToken(shop_id, user_id)
            access_token = await asyncio.to_thread(token.get_token)

            queue_name = self._build_queue_name(shop_id, user_id)
            await self._setup_message_consumer(queue_name)

            full_url = self._build_websocket_url(access_token)

            self.logger.debug(f"正在连接到拼多多WebSocket: {shop_id}-{username}")

            async with websockets.connect(
                full_url,
                ping_interval=60,
                ping_timeout=30,
                max_size=10**7,
                compression=None,
                close_timeout=10
            ) as websocket:
                self._set_active_websocket(connection_key, websocket)
                self.resource_manager.register_websocket(
                    websocket,
                    f"PDD WebSocket ({shop_id}-{username})"
                )
                self.logger.debug(f"WebSocket连接已建立: {shop_id}-{username}")

                if self.ws and not self._is_ws_closed(self.ws):
                    self.logger.debug(f"WebSocket连接正常: {shop_id}-{username}")
                else:
                    self.logger.error(f"WebSocket连接异常: {shop_id}-{username}")

                self.status_manager.update_status(shop_id, user_id, username, ConnectionState.CONNECTED)
                self.logger.debug(f"暂时跳过在线状态设置: {shop_id}-{username}")

                on_success()

                heartbeat_task = None
                if self.heartbeat_config.enable_heartbeat:
                    heartbeat_task = asyncio.create_task(
                        self._heartbeat_loop(websocket, shop_id, user_id, username, stop_event)
                    )
                    self._heartbeat_tasks[connection_key] = heartbeat_task
                    self.logger.debug(f"心跳检查已启动: {shop_id}-{username}")

                message_task = asyncio.create_task(
                    self._message_loop(websocket, shop_id, user_id, username, queue_name, stop_event)
                )

                stop_task = asyncio.create_task(stop_event.wait())

                try:
                    tasks = [message_task, stop_task]
                    if heartbeat_task:
                        tasks.append(heartbeat_task)

                    done, pending = await asyncio.wait(
                        tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    should_cleanup = False
                    if stop_task in done:
                        self.logger.debug(f"收到停止信号: {shop_id}-{username}")
                        should_cleanup = True
                    else:
                        self.logger.warning(f"消息循环异常结束: {shop_id}-{username}")
                        should_cleanup = True

                    for task in pending:
                        task.cancel()
                        try:
                            await asyncio.wait_for(task, timeout=3.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError, asyncio.InvalidStateError):
                            pass
                        except Exception as e:
                            self.logger.debug(f"等待任务取消时出错: {self._sanitize_for_log(e)}")

                    if should_cleanup:
                        if stop_task in done:
                            await self._cleanup_resources(
                                self._build_queue_name(shop_id, user_id),
                                cleanup_reconnect_tasks=True,
                                cleanup_heartbeat_tasks=True,
                                cleanup_all_websockets=False,
                                stop_consumer=True,
                                connection_key=connection_key,
                            )
                        else:
                            await self._cleanup_resources(
                                self._build_queue_name(shop_id, user_id),
                                cleanup_reconnect_tasks=False,
                                cleanup_heartbeat_tasks=False,
                                cleanup_all_websockets=False,
                                stop_consumer=False,
                                connection_key=connection_key,
                            )
                            raise RuntimeError(f"message loop ended unexpectedly: {shop_id}-{username}")

                except asyncio.CancelledError:
                    self.logger.debug(f"WebSocket任务被取消: {shop_id}-{username}")
                    message_task.cancel()
                    if heartbeat_task:
                        heartbeat_task.cancel()
                    try:
                        await asyncio.wait_for(message_task, timeout=3.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError, asyncio.InvalidStateError):
                        pass
                    if heartbeat_task:
                        try:
                            await asyncio.wait_for(heartbeat_task, timeout=3.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError, asyncio.InvalidStateError):
                            pass
                    await self._cleanup_resources(self._build_queue_name(shop_id, user_id), connection_key=connection_key)

        except ws_exceptions.ConnectionClosed as e:
            safe_error = self._sanitize_for_log(e)
            self.status_manager.update_status(shop_id, user_id, username, ConnectionState.ERROR, safe_error)
            self.logger.warning(f"WebSocket连接已关闭: {shop_id}-{username}, 错误: {safe_error}")
            await self._cleanup_resources(self._build_queue_name(shop_id, user_id), connection_key=connection_key)
            if self.reconnect_config.enable_auto_reconnect and not self._is_stop_requested(shop_id, user_id):
                raise RuntimeError(f"WebSocket连接已关闭: {safe_error}")
            on_failure(f"WebSocket连接已关闭: {safe_error}")
        except Exception as e:
            safe_error = self._sanitize_for_log(e)
            self.status_manager.update_status(shop_id, user_id, username, ConnectionState.ERROR, safe_error)
            self.logger.error(f"WebSocket连接错误: {shop_id}-{username}, 错误: {safe_error}")
            await self._cleanup_resources(self._build_queue_name(shop_id, user_id), connection_key=connection_key)
            if self.reconnect_config.enable_auto_reconnect and not self._is_stop_requested(shop_id, user_id):
                raise RuntimeError(f"WebSocket连接错误: {safe_error}")
            on_failure(f"WebSocket连接错误: {safe_error}")

    def request_stop(self, shop_id: str = None, user_id: str = None):
        """请求停止WebSocket连接"""
        if shop_id is not None and user_id is not None:
            self._get_or_create_stop_event(self._connection_key(shop_id, user_id)).set()
            return
        for event in getattr(self, "_stop_events", {}).values():
            event.set()
        if self._stop_event:
            self._stop_event.set()

    async def stop_all_connections(self):
        """停止所有连接并清理所有任务"""
        try:
            self.logger.info("正在停止所有连接...")

            for event in getattr(self, "_stop_events", {}).values():
                event.set()
            if self._stop_event:
                self._stop_event.set()

            for connection_key, task in list(self._reconnect_tasks.items()):
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=5.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        self.logger.debug(f"任务已取消或超时: {connection_key}")
                    except Exception as e:
                        self.logger.error(f"停止任务时出错: {connection_key}, {self._sanitize_for_log(e)}")
                del self._reconnect_tasks[connection_key]

            for connection_key, task in list(self._heartbeat_tasks.items()):
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=3.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        self.logger.debug(f"心跳任务已取消或超时: {connection_key}")
                    except Exception as e:
                        self.logger.error(f"停止心跳任务时出错: {connection_key}, {self._sanitize_for_log(e)}")
                del self._heartbeat_tasks[connection_key]

            for connection_key in list(getattr(self, "_websockets", {}).keys()):
                await self._close_account_websocket(connection_key)

            await self.cleanup_processing_tasks()

            try:
                from Message import message_consumer_manager

                await message_consumer_manager.stop_all()
            except asyncio.InvalidStateError:
                self.logger.debug("消息消费者已在其他事件循环中停止")
            except Exception as e:
                self.logger.warning(f"停止全部消息消费者失败: {self._sanitize_for_log(e)}")

            self.logger.info("所有连接已停止")

        except Exception:
            self.logger.exception("停止所有连接时发生错误")

    async def _heartbeat_loop(self, websocket, shop_id: str, user_id: str, username: str, stop_event: asyncio.Event = None):
        """心跳检查循环"""
        connection_key = self._connection_key(shop_id, user_id)
        stop_event = stop_event or self._get_or_create_stop_event(connection_key)
        consecutive_failures = 0

        try:
            while not stop_event.is_set():
                try:
                    start_time = time.time()
                    await self._send_heartbeat_ping(websocket)
                    response_time = time.time() - start_time

                    consecutive_failures = 0
                    self.logger.debug(f"心跳成功: {shop_id}-{username}, 响应时间: {response_time:.3f}s")

                    status = self.status_manager.get_status(shop_id, user_id)
                    if status and status.state == ConnectionState.CONNECTED:
                        pass

                    try:
                        await asyncio.wait_for(
                            stop_event.wait(),
                            timeout=self.heartbeat_config.heartbeat_interval,
                        )
                    except asyncio.TimeoutError:
                        pass

                except asyncio.TimeoutError:
                    consecutive_failures += 1
                    self.logger.warning(f"心跳超时: {shop_id}-{username}, 连续失败: {consecutive_failures}")
                    try:
                        await asyncio.wait_for(
                            stop_event.wait(),
                            timeout=self.heartbeat_config.heartbeat_timeout,
                        )
                    except asyncio.TimeoutError:
                        pass

                except Exception as e:
                    consecutive_failures += 1
                    self.logger.warning(
                        f"心跳失败: {shop_id}-{username}, 错误: {self._sanitize_for_log(e)}, 连续失败: {consecutive_failures}"
                    )

                    if consecutive_failures >= self.heartbeat_config.max_heartbeat_failures:
                        self.logger.error(f"心跳检查失败次数过多，标记连接为错误状态: {shop_id}-{username}")
                        self.status_manager.update_status(
                            shop_id, user_id, username,
                            ConnectionState.ERROR,
                            f"心跳检查失败: 连续{consecutive_failures}次失败"
                        )
                        break

                    try:
                        await asyncio.wait_for(
                            stop_event.wait(),
                            timeout=self.heartbeat_config.heartbeat_timeout,
                        )
                    except asyncio.TimeoutError:
                        pass

        except asyncio.CancelledError:
            self.logger.debug(f"心跳循环被取消: {shop_id}-{username}")
        except Exception:
            self.logger.exception(f"心跳循环异常: {shop_id}-{username}")
        finally:
            if connection_key in self._heartbeat_tasks:
                del self._heartbeat_tasks[connection_key]
            self.logger.debug(f"心跳循环已结束: {shop_id}-{username}")

    async def _send_heartbeat_ping(self, websocket) -> None:
        """发送 ping 并等待对应 pong，避免只发送不确认连接健康。"""
        pong_waiter = websocket.ping()
        if asyncio.iscoroutine(pong_waiter):
            pong_waiter = await pong_waiter
        if pong_waiter is not None:
            await asyncio.wait_for(pong_waiter, timeout=self.heartbeat_config.heartbeat_timeout)

    async def _message_loop(self, websocket, shop_id: str, user_id: str, username: str, queue_name: str, stop_event: asyncio.Event = None):
        """消息接收循环"""
        connection_key = self._connection_key(shop_id, user_id)
        stop_event = stop_event or self._get_or_create_stop_event(connection_key)
        try:
            self.logger.info(f"消息循环开始: {shop_id}-{username}")

            async for message in websocket:
                if stop_event.is_set():
                    self.logger.info(f"停止事件已设置，退出消息循环: {shop_id}-{username}")
                    break
                task = asyncio.create_task(
                    self._process_websocket_message_concurrent(
                        message, shop_id, user_id, username, queue_name
                    )
                )

                self.processing_tasks.add(task)
                if not hasattr(self, "_processing_tasks_by_connection"):
                    self._processing_tasks_by_connection = {}
                self._processing_tasks_by_connection.setdefault(connection_key, set()).add(task)
                task.add_done_callback(self.processing_tasks.discard)
                task.add_done_callback(
                    lambda done_task, key=connection_key: self._processing_tasks_by_connection.get(key, set()).discard(done_task)
                )

        except ws_exceptions.ConnectionClosedError as cce:
            safe_error = self._sanitize_for_log(cce)
            self.logger.error(f"WebSocket连接异常关闭: {shop_id}-{username}, 错误: {safe_error}")
            if not stop_event.is_set():
                raise RuntimeError(f"WebSocket连接异常关闭: {shop_id}-{username}, {safe_error}") from cce
        except ws_exceptions.ConnectionClosed as cc:
            self.logger.warning(f"WebSocket连接正常关闭: {shop_id}-{username}, 代码: {cc.code}")
            if not stop_event.is_set():
                raise RuntimeError(f"WebSocket连接关闭: {shop_id}-{username}, code={cc.code}") from cc
        except Exception as e:
            safe_error = self._sanitize_for_log(e)
            self.logger.error(f"消息循环错误: {shop_id}-{username}, 错误: {safe_error}")
            if not stop_event.is_set():
                raise RuntimeError(f"消息循环错误: {shop_id}-{username}, {safe_error}") from e

    async def _process_websocket_message_concurrent(self, message: str, shop_id: str, user_id: str, username: str, queue_name: str):
        """并发处理WebSocket消息"""
        async with self.message_semaphore:
            try:
                await self._process_websocket_message(message, shop_id, user_id, username, queue_name)
            except Exception:
                self.logger.exception("并发处理消息失败")

    async def cleanup_processing_tasks(self, connection_key: str = None):
        """清理所有处理任务"""
        if connection_key:
            tasks = set(getattr(self, "_processing_tasks_by_connection", {}).get(connection_key, set()))
        else:
            tasks = set(self.processing_tasks)
        if not tasks:
            return

        self.logger.info(f"清理 {len(tasks)} 个处理任务")
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.error(f"清理任务失败: {self._sanitize_for_log(e)}")

            self.processing_tasks.discard(task)
            if connection_key:
                self._processing_tasks_by_connection.get(connection_key, set()).discard(task)

        if not connection_key:
            self.processing_tasks.clear()
            getattr(self, "_processing_tasks_by_connection", {}).clear()

    async def _cleanup_reconnect_tasks(self, connection_key: str = None):
        """清理所有重连任务"""
        try:
            task_items = (
                [(connection_key, self._reconnect_tasks[connection_key])]
                if connection_key and connection_key in self._reconnect_tasks
                else list(self._reconnect_tasks.items())
            )
            for task_key, task in task_items:
                if task is asyncio.current_task():
                    self._reconnect_tasks.pop(task_key, None)
                    continue
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=5.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                    except asyncio.InvalidStateError:
                        self.logger.debug(f"重连任务在不同的的事件循环中: {task_key}")
                    except Exception as e:
                        self.logger.error(f"清理重连任务失败: {task_key}, {self._sanitize_for_log(e)}")
                self._reconnect_tasks.pop(task_key, None)
        except Exception as e:
            self.logger.error(f"清理重连任务列表失败: {self._sanitize_for_log(e)}")

    async def _cleanup_heartbeat_tasks(self, connection_key: str = None):
        """清理所有心跳任务"""
        try:
            task_items = (
                [(connection_key, self._heartbeat_tasks[connection_key])]
                if connection_key and connection_key in self._heartbeat_tasks
                else list(self._heartbeat_tasks.items())
            )
            for task_key, task in task_items:
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=3.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                    except asyncio.InvalidStateError:
                        self.logger.debug(f"心跳任务在不同的的事件循环中: {task_key}")
                    except Exception as e:
                        self.logger.error(f"清理心跳任务失败: {task_key}, {self._sanitize_for_log(e)}")
                self._heartbeat_tasks.pop(task_key, None)
        except Exception as e:
            self.logger.error(f"清理心跳任务列表失败: {self._sanitize_for_log(e)}")

    async def _cleanup_resources(
        self,
        queue_name: str,
        cleanup_reconnect_tasks: bool = True,
        cleanup_heartbeat_tasks: bool = True,
        cleanup_all_websockets: bool = True,
        stop_consumer: bool = True,
        connection_key: str = None,
    ):
        """清理资源"""
        from Message import message_consumer_manager

        try:
            await self.cleanup_processing_tasks(connection_key)
            if cleanup_reconnect_tasks:
                await self._cleanup_reconnect_tasks(connection_key)
            if cleanup_heartbeat_tasks:
                await self._cleanup_heartbeat_tasks(connection_key)
            if cleanup_all_websockets:
                await self.resource_manager.cleanup_all()
                getattr(self, "_websockets", {}).clear()
                self.ws = None
            elif connection_key:
                await self._close_account_websocket(connection_key)
            elif self.ws:
                await self._safe_close_websocket(self.ws)
                self.ws = None

            if stop_consumer:
                try:
                    await message_consumer_manager.stop_consumer(queue_name)
                    self.logger.debug(f"已停止消息消费者: {queue_name}")
                except asyncio.InvalidStateError:
                    self.logger.debug(f"消息消费者已在其他事件循环中停止: {queue_name}")
                except Exception as e:
                    self.logger.warning(f"停止消息消费者失败: {queue_name}, {self._sanitize_for_log(e)}")

        except Exception:
            self.logger.exception("清理资源失败")


# 延迟导入避免循环依赖
from database import db_manager
from core.connection_status import ConnectionState

__all__ = ['LifecycleMixin']
