"""
基础服务类 - 提供统一的服务基础功能
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from utils.logger_loguru import get_logger
import time
import re


def _sanitize_for_log(value: Any) -> Any:
    if isinstance(value, BaseException):
        text = str(value)
        return f"{type(value).__name__}: {_sanitize_for_log(text)}" if text else type(value).__name__

    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            key_text = str(key or "")
            normalized_key = re.sub(r"[^a-z0-9]", "", key_text.lower())
            if key_text.lower() in _SENSITIVE_TEXT_KEYS or normalized_key in _SENSITIVE_KEY_NORMALIZED:
                result[key] = "***"
            else:
                result[key] = _sanitize_for_log(item)
        return result

    if isinstance(value, list):
        return [_sanitize_for_log(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_sanitize_for_log(item) for item in value)

    if not isinstance(value, str):
        return value

    text = str(value or "")
    text = re.sub(
        r'(?i)((?:authorization)\s*[:=]\s*bearer\s+)([^\s&;,"\']+)',
        r'\1***',
        text,
    )
    for key in _SENSITIVE_TEXT_KEYS:
        pattern = re.compile(
            rf"(?i)((?:[\"']?){re.escape(key)}(?:[\"']?)\s*[:=]\s*[\"']?)([^\s&;,\"']+)"
        )
        text = pattern.sub(r"\1***", text)
    return text


_SENSITIVE_TEXT_KEYS = (
    "authorization",
    "cookie",
    "cookies",
    "token",
    "api_key",
    "api-key",
    "x-api-key",
    "access_token",
    "accessToken",
    "anti-content",
    "anti_content",
    "antiContent",
    "password",
)

_SENSITIVE_KEY_NORMALIZED = {
    re.sub(r"[^a-z0-9]", "", key.lower()) for key in _SENSITIVE_TEXT_KEYS
}

class BaseService(ABC):
    """服务基类 - 提供通用功能"""

    def __init__(self, logger=None):
        self.logger = logger or get_logger(self.__class__.__name__)
        self._initialized = False
        self._disposed = False
        self._start_time = time.time()

    @abstractmethod
    def initialize(self) -> bool:
        """初始化服务"""
        pass

    @abstractmethod
    def dispose(self):
        """释放服务资源"""
        pass

    @property
    def is_initialized(self) -> bool:
        """服务是否已初始化"""
        return self._initialized

    @property
    def is_disposed(self) -> bool:
        """服务是否已释放"""
        return self._disposed

    @property
    def uptime(self) -> float:
        """服务运行时间（秒）"""
        return time.time() - self._start_time

    def handle_exception(self, e: Exception, context: str = "") -> bool:
        """
        统一异常处理

        Args:
            e: 异常对象
            context: 异常上下文信息

        Returns:
            bool: 是否已处理异常
        """
        error_msg = f"{self.__class__.__name__}"
        if context:
            error_msg += f" - {context}"
        error_msg += f": {_sanitize_for_log(str(e))}"

        self.logger.error(error_msg, exc_info=True)
        return False

    def validate_state(self, operation: str = "") -> bool:
        """验证服务状态"""
        if self._disposed:
            raise RuntimeError(f"服务已被释放，无法执行{operation}")
        if not self._initialized:
            raise RuntimeError(f"服务未初始化，无法执行{operation}")
        return True

    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息"""
        return {
            "name": self.__class__.__name__,
            "initialized": self._initialized,
            "disposed": self._disposed,
            "uptime": self.uptime
        }


class ConfigurableService(BaseService):
    """可配置服务基类"""

    def __init__(self, config_service=None, logger=None):
        super().__init__(logger)
        self.config_service = config_service
        self._config = {}

    def load_config(self, config_key: str = None, default_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """加载配置"""
        if default_config:
            self._config = default_config

        if self.config_service and config_key:
            try:
                service_config = self.config_service.get(config_key, {})
                if service_config:
                    self._config.update(service_config)
            except Exception as e:
                self.handle_exception(e, f"加载配置失败: {config_key}")

        return self._config

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return self._config.get(key, default)

    def set_config(self, key: str, value: Any):
        """设置配置项"""
        self._config[key] = value


class AsyncService(BaseService):
    """异步服务基类"""

    async def initialize_async(self) -> bool:
        """异步初始化服务"""
        try:
            result = self.initialize()
            self._initialized = result
            return result
        except Exception as e:
            self.handle_exception(e, "异步初始化失败")
            return False

    async def dispose_async(self):
        """异步释放服务资源"""
        try:
            self.dispose()
        except Exception as e:
            self.handle_exception(e, "异步释放资源失败")


class HealthCheckable(BaseService):
    """支持健康检查的服务基类"""

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            Dict[str, Any]: 健康状态信息
            {
                "status": "healthy" | "degraded" | "unhealthy",
                "details": {...},
                "timestamp": float
            }
        """
        pass

    async def get_health_status(self) -> str:
        """获取健康状态"""
        try:
            health_info = await self.health_check()
            return health_info.get("status", "unknown")
        except Exception:
            return "unhealthy"
