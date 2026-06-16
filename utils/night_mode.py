"""
Night-mode handoff helpers.

During night mode we keep replying, but do not move conversations to human CS.
"""
from __future__ import annotations

from datetime import datetime, time
from threading import Lock
import time as monotonic_time
from typing import Dict, Tuple

from core.base_service import _sanitize_for_log
from utils.logger_loguru import get_logger


NIGHT_MODE_START_HOUR = 23
NIGHT_MODE_END_HOUR = 8

NIGHT_MODE_TRANSFER_RESULT_PREFIX = "夜间不转人工"
DEFAULT_NIGHT_MODE_REPLY_TEMPLATES = (
    "亲，当前问题需要高级客服为您处理，高级客服上班时间为{work_time_text}，建议您晚点联系这边由高级客服为您处理哦！",
    "亲，专业的高级客服下班了，还没上班，{resume_text}后联系这边，会为您妥善处理的，您耐心等待下。",
    "亲，您的问题这边已经收到啦，目前高级客服不在线，{resume_text}后会有专人继续帮您处理，请您先放心。",
    "亲，您先别着急，夜间无法转接高级客服，{resume_text}后客服上班会继续为您核实处理的。",
    "亲，您反馈的情况我已经了解，目前夜间只能先为您记录，高级客服上班后会优先处理。",
    "亲，现在是夜间值守时段，高级客服暂时不在线，您可以先把情况补充完整，{resume_text}后会继续处理。",
    "亲，已经帮您记录诉求了，当前时段无法转人工，{resume_text}后高级客服会接着为您处理。",
    "亲，您连续发的消息我这边都收到了，请您先耐心等一下，高级客服上班后会为您处理。",
)

_state_lock = Lock()
_reply_stages: Dict[str, Tuple[int, float]] = {}
logger = get_logger(__name__)
NIGHT_MODE_REPLY_STATE_TTL_SECONDS = 24 * 60 * 60
NIGHT_MODE_REPLY_STATE_MAX_ENTRIES = 5000


def is_night_mode(now: datetime | None = None) -> bool:
    current = now or datetime.now()
    start, end = get_night_mode_time_range()
    current_time = current.time().replace(second=0, microsecond=0)
    if start == end:
        return False
    if start < end:
        return start <= current_time < end
    return current_time >= start or current_time < end


def get_night_mode_time_range() -> tuple[time, time]:
    start_text = f"{NIGHT_MODE_START_HOUR:02d}:00"
    end_text = f"{NIGHT_MODE_END_HOUR:02d}:00"
    try:
        from config import config

        start_text = str(config.get("night_mode.start", start_text) or start_text)
        end_text = str(config.get("night_mode.end", end_text) or end_text)
    except Exception as exc:
        logger.debug(f"读取夜间模式时间配置失败，使用默认时间段: {_sanitize_for_log(exc)}")

    try:
        start = datetime.strptime(start_text, "%H:%M").time()
        end = datetime.strptime(end_text, "%H:%M").time()
        return start, end
    except Exception:
        return time(NIGHT_MODE_START_HOUR, 0), time(NIGHT_MODE_END_HOUR, 0)


def format_time_range(start: time, end: time) -> str:
    return f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"


def format_resume_time(end: time) -> str:
    hour = end.hour
    minute = end.minute
    if minute:
        return end.strftime("%H:%M")
    if hour < 12:
        return f"早上{hour}点"
    if hour == 12:
        return "中午12点"
    if hour < 18:
        return f"下午{hour - 12}点"
    return f"晚上{hour - 12}点"


def get_night_mode_prompt_values() -> Dict[str, str]:
    start, end = get_night_mode_time_range()
    return {
        "range_text": format_time_range(start, end),
        "resume_text": format_resume_time(end),
        "work_time_text": format_time_range(end, start),
    }


def get_night_mode_replies() -> tuple[str, ...]:
    values = get_night_mode_prompt_values()
    return tuple(_format_reply_template(template, values) for template in _get_night_mode_reply_templates())


def _get_night_mode_reply_templates() -> tuple[str, ...]:
    try:
        from config import config

        configured = config.get("night_mode.reply_templates", list(DEFAULT_NIGHT_MODE_REPLY_TEMPLATES))
    except Exception:
        configured = DEFAULT_NIGHT_MODE_REPLY_TEMPLATES

    if not isinstance(configured, (list, tuple)):
        configured = DEFAULT_NIGHT_MODE_REPLY_TEMPLATES

    templates = []
    for item in configured:
        text = str(item or "").strip()
        if text:
            templates.append(text)
    return tuple(templates) or DEFAULT_NIGHT_MODE_REPLY_TEMPLATES


def _format_reply_template(template: str, values: Dict[str, str]) -> str:
    try:
        return str(template).format(**values)
    except Exception:
        return str(template)


def build_night_mode_key(shop_id: object = None, user_id: object = None, recipient_uid: object = None) -> str:
    return f"{shop_id or ''}:{user_id or ''}:{recipient_uid or ''}"


def _prune_reply_stages(now: float) -> None:
    expired_before = now - NIGHT_MODE_REPLY_STATE_TTL_SECONDS
    for key, (_stage, updated_at) in list(_reply_stages.items()):
        if updated_at < expired_before:
            _reply_stages.pop(key, None)

    overflow = len(_reply_stages) - NIGHT_MODE_REPLY_STATE_MAX_ENTRIES
    if overflow > 0:
        oldest_keys = sorted(_reply_stages, key=lambda item: _reply_stages[item][1])[:overflow]
        for key in oldest_keys:
            _reply_stages.pop(key, None)


def get_night_mode_reply(key: str | None = None) -> str:
    replies = get_night_mode_replies()
    if not key:
        return replies[0]

    with _state_lock:
        now = monotonic_time.monotonic()
        _prune_reply_stages(now)
        stage, _updated_at = _reply_stages.get(key, (0, now))
        _reply_stages[key] = (stage + 1, now)
        _prune_reply_stages(now)

    return replies[stage % len(replies)]


def reset_night_mode_reply_state() -> None:
    with _state_lock:
        _reply_stages.clear()
