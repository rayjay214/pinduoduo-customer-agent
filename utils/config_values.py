"""Small helpers for parsing loosely typed config values."""
from __future__ import annotations

import math
from typing import Any


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        try:
            numeric = float(value)
        except OverflowError:
            return default
        if not math.isfinite(numeric):
            return default
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on", "开启", "启用"}:
            return True
        if text in {"0", "false", "no", "n", "off", "关闭", "禁用"}:
            return False
    return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        numeric = int(value)
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return numeric
    except (TypeError, ValueError, OverflowError):
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return numeric if math.isfinite(numeric) else default
