"""
Agent 配置模块。
管理默认配置和运行时配置参数。
"""
from dataclasses import dataclass, field

from config import get_config
from utils.config_values import as_bool, as_float, as_int
from utils.logger_loguru import get_logger

logger = get_logger("AgentConfig")

# 默认参数
DEFAULT_DB_PATH = "./temp/agent.db"
DEFAULT_TOKEN_WINDOW = 18432
DEFAULT_COMPRESS_RATIO = 16384 / 18432
DEFAULT_RETAIN_COUNT = 10
DEFAULT_MAX_LOOPS = 5
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 512
DEFAULT_TOOL_CALL_MAX_TOKENS = 512
DEFAULT_LLM_TIMEOUT_SECONDS = 20.0
DEFAULT_LLM_MAX_CONCURRENT_REQUESTS = 2
DEFAULT_DISABLE_THINKING = True
DEFAULT_DISABLE_THINKING_API_BASE_PATTERNS = (
    "127.0.0.1",
    "localhost",
    "xiaomimimo.com",
    "siliconflow.cn",
)
DEFAULT_DISABLE_THINKING_MODEL_PREFIXES = (
    "mimo-",
    "glm-",
    "qwen/",
    "nex-agi/",
)


def _get_int_config(key: str, default: int) -> int:
    value = get_config(key, default)
    result = as_int(value, default)
    if result == default and value != default:
        logger.warning(f"配置 {key}={value!r} 不是有效整数，使用默认值 {default}")
    return result


def _get_float_config(key: str, default: float) -> float:
    value = get_config(key, default)
    result = as_float(value, default)
    if result == default and value != default:
        logger.warning(f"配置 {key}={value!r} 不是有效数字，使用默认值 {default}")
    return result


@dataclass
class AgentConfig:
    """Agent 配置数据类。"""

    db_path: str = field(default_factory=lambda: get_config("db_path", DEFAULT_DB_PATH))
    token_window: int = field(default_factory=lambda: _get_int_config("agent.token_window", DEFAULT_TOKEN_WINDOW))
    compress_ratio: float = field(default_factory=lambda: _get_float_config("agent.compress_ratio", DEFAULT_COMPRESS_RATIO))
    retain_count: int = field(default_factory=lambda: _get_int_config("agent.retain_count", DEFAULT_RETAIN_COUNT))
    max_loops: int = field(default_factory=lambda: _get_int_config("agent.max_loops", DEFAULT_MAX_LOOPS))
    temperature: float = field(default_factory=lambda: _get_float_config("agent.temperature", DEFAULT_TEMPERATURE))
    max_tokens: int = field(default_factory=lambda: _get_int_config("llm.max_tokens", DEFAULT_MAX_TOKENS))
    tool_call_max_tokens: int = field(default_factory=lambda: _get_int_config("llm.tool_call_max_tokens", DEFAULT_TOOL_CALL_MAX_TOKENS))
    request_timeout_seconds: float = field(default_factory=lambda: _get_float_config("llm.request_timeout_seconds", DEFAULT_LLM_TIMEOUT_SECONDS))
    max_concurrent_requests: int = field(default_factory=lambda: _get_int_config("llm.max_concurrent_requests", DEFAULT_LLM_MAX_CONCURRENT_REQUESTS))
    disable_thinking: bool = field(default_factory=lambda: as_bool(get_config("llm.disable_thinking", DEFAULT_DISABLE_THINKING), DEFAULT_DISABLE_THINKING))
    disable_thinking_api_base_patterns: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            str(item).lower()
            for item in get_config("llm.disable_thinking_api_base_patterns", list(DEFAULT_DISABLE_THINKING_API_BASE_PATTERNS))
            if str(item).strip()
        )
    )
    disable_thinking_model_prefixes: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            str(item).lower()
            for item in get_config("llm.disable_thinking_model_prefixes", list(DEFAULT_DISABLE_THINKING_MODEL_PREFIXES))
            if str(item).strip()
        )
    )

    model_name: str = field(default_factory=lambda: get_config("llm.model_name", "gpt-3.5-turbo"))
    api_key: str = field(default_factory=lambda: get_config("llm.api_key", ""))
    api_base: str = field(default_factory=lambda: get_config("llm.api_base", ""))
    fallback_enabled: bool = field(default_factory=lambda: as_bool(get_config("llm.fallback.enabled", False), False))
    fallback_model_name: str = field(default_factory=lambda: get_config("llm.fallback.model_name", ""))
    fallback_api_key: str = field(default_factory=lambda: get_config("llm.fallback.api_key", ""))
    fallback_api_base: str = field(default_factory=lambda: get_config("llm.fallback.api_base", ""))
    fallback_timeout_seconds: float = field(default_factory=lambda: _get_float_config("llm.fallback.timeout_seconds", DEFAULT_LLM_TIMEOUT_SECONDS))

    @classmethod
    def load_from_config(cls) -> "AgentConfig":
        """从配置文件加载配置。"""
        config = cls()
        logger.debug("Agent 配置加载完成")
        return config

    def validate(self) -> bool:
        """验证配置有效性。"""
        if not self.api_key:
            logger.error("LLM API 密钥未配置")
            return False
        return True
