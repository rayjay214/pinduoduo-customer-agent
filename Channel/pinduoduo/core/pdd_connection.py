# 连接管理模块
import asyncio
import websockets
from websockets import exceptions as ws_exceptions
from typing import Optional, Any
from datetime import datetime
from utils.logger_loguru import get_logger
from core.base_service import _sanitize_for_log


class ConnectionMixin:
    """连接管理 Mixin"""

    base_url: str = "wss://m-ws.pinduoduo.com/"

    async def _connect_with_retry(self, shop_id: str, user_id: str, username: str, on_success: callable, on_failure: callable):
        """带重连机制的WebSocket连接"""
        logger = get_logger("PDDChannel")

        consecutive_failures = 0
        last_reset_connect_time = None

        while consecutive_failures < self.reconnect_config.max_attempts:
            if self._is_stop_requested(shop_id, user_id):
                logger.info(f"收到停止信号，取消重连: {shop_id}-{username}")
                self.status_manager.update_status(shop_id, user_id, username, ConnectionState.DISCONNECTED)
                return

            try:
                if consecutive_failures > 0:
                    self.status_manager.update_status(shop_id, user_id, username, ConnectionState.RECONNECTING)
                    logger.info(f"尝试重连 ({consecutive_failures + 1}/{self.reconnect_config.max_attempts}): {shop_id}-{username}")

                await self._connect_single_attempt(shop_id, user_id, username, on_success, on_failure)
                consecutive_failures = 0
                return

            except Exception as e:
                if self._is_stop_requested(shop_id, user_id):
                    logger.info(f"连接被停止信号中断: {shop_id}-{username}")
                    self.status_manager.update_status(shop_id, user_id, username, ConnectionState.DISCONNECTED)
                    return

                stable_connect_time = self._stable_connect_time(shop_id, user_id)
                if stable_connect_time is not None and stable_connect_time != last_reset_connect_time:
                    consecutive_failures = 0
                    last_reset_connect_time = stable_connect_time

                consecutive_failures += 1
                safe_error = _sanitize_for_log(e)
                if consecutive_failures >= self.reconnect_config.max_attempts:
                    self.status_manager.update_status(shop_id, user_id, username, ConnectionState.ERROR, safe_error)
                    logger.error(f"连接失败，已达到最大重试次数: {shop_id}-{username}, 错误: {safe_error}")
                    on_failure(f"连接失败，已达到最大重试次数: {safe_error}")
                    return

                delay = min(
                    self.reconnect_config.initial_delay * (self.reconnect_config.backoff_factor ** (consecutive_failures - 1)),
                    self.reconnect_config.max_delay
                )

                logger.warning(f"连接失败，{delay:.1f}秒后重试: {shop_id}-{username}, 错误: {safe_error}")

                try:
                    for _ in range(int(delay * 10)):
                        if self._is_stop_requested(shop_id, user_id):
                            logger.info(f"重连延迟被停止信号中断: {shop_id}-{username}")
                            self.status_manager.update_status(shop_id, user_id, username, ConnectionState.DISCONNECTED)
                            return
                        await asyncio.sleep(0.1)
                except (asyncio.CancelledError, RuntimeError):
                    logger.info(f"重连延迟被中断或事件循环关闭: {shop_id}-{username}")
                    self.status_manager.update_status(shop_id, user_id, username, ConnectionState.DISCONNECTED)
                    return

    def _stable_connect_time(self, shop_id: str, user_id: str):
        """返回已稳定连接的时间戳；同一段稳定连接只用于重置一次失败计数。"""
        status = self.status_manager.get_status(shop_id, user_id)
        last_connect_time = getattr(status, "last_connect_time", None)
        if last_connect_time is None:
            return None
        stable_seconds = max(0.0, float(getattr(self.reconnect_config, "stable_reset_seconds", 300.0)))
        if (datetime.now() - last_connect_time).total_seconds() >= stable_seconds:
            return last_connect_time
        return None

    async def _connect_single_attempt(self, shop_id: str, user_id: str, username: str, on_success: callable, on_failure: callable):
        """单次WebSocket连接尝试"""
        await self.init(shop_id, user_id, username, on_success, on_failure)

    def _is_ws_closed(self, ws: Any) -> bool:
        """检查WebSocket是否已关闭"""
        try:
            closed = getattr(ws, "closed", None)
            if isinstance(closed, bool):
                return closed
            return False
        except Exception:
            return False

    async def _safe_close_websocket(self, ws: Any):
        """安全关闭WebSocket"""
        try:
            close_fn = getattr(ws, "close", None)
            if close_fn:
                result = close_fn()
                if asyncio.iscoroutine(result):
                    await result
        except Exception as e:
            self.logger.debug(f"关闭WebSocket失败: {_sanitize_for_log(e)}")


# 延迟导入避免循环依赖
from core.connection_status import ConnectionState
__all__ = ['ConnectionMixin']
