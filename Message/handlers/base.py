"""
处理器基类和通用工具
"""
import json
from typing import Dict, Any, Optional
from utils.logger_loguru import get_logger
from bridge.context import Context
from ..core.handlers import MessageHandler
from core.base_service import _sanitize_for_log



class BaseHandler(MessageHandler):
    """处理器基类，提供通用功能"""

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name or self.__class__.__name__

    async def log_message(self, context: Context, action: str, extra_info: str = ""):
        """统一的日志记录（不记录完整内容以保护隐私）"""
        user_info = self._get_user_info(context)
        content_chars = len(str(context.content or ""))
        safe_extra_info = _sanitize_for_log(extra_info)
        self.logger.info(f"{self.name} {action} - {user_info} - content_chars={content_chars} {safe_extra_info}")

    def _get_user_info(self, context: Context) -> str:
        """提取用户信息"""
        try:
            if hasattr(context, 'kwargs') and context.kwargs:
                from_uid = self._get_kwarg(context.kwargs, 'from_uid')
                username = self._get_kwarg(context.kwargs, 'username')
                if username:
                    return f"用户:{username}({from_uid})"
                elif from_uid:
                    return f"用户:{from_uid}"
            return "用户:unknown"
        except Exception:
            return "用户:unknown"

    @staticmethod
    def _get_kwarg(kwargs: Any, key: str, default: Any = None) -> Any:
        if isinstance(kwargs, dict):
            return kwargs.get(key, default)
        return getattr(kwargs, key, default)
