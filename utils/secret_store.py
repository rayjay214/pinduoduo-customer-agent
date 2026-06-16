from __future__ import annotations

import base64
import ctypes
import json
import os
from ctypes import wintypes
from typing import Any, Optional


_PREFIX = "dpapi:v1:"
_ENTROPY = b"Customer-Agent:account-secret:v1"


class SecretStoreError(RuntimeError):
    """Raised when the local OS secret protection API cannot protect data."""


def is_protected_secret(value: Optional[str]) -> bool:
    return isinstance(value, str) and value.startswith(_PREFIX)


def _normalize_secret_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def protect_secret(value: Any) -> Optional[str]:
    """Protect a secret for storage using the current Windows user profile."""
    value = _normalize_secret_value(value)
    if value is None or value == "" or is_protected_secret(value):
        return value
    protected = _crypt_protect(value.encode("utf-8"))
    return _PREFIX + base64.b64encode(protected).decode("ascii")


def reveal_secret(value: Optional[str]) -> Optional[str]:
    """Reveal a protected secret; plain legacy values are returned unchanged."""
    if value is None or value == "" or not is_protected_secret(value):
        return value
    encoded = value[len(_PREFIX):]
    try:
        protected = base64.b64decode(encoded.encode("ascii"), validate=True)
    except Exception as exc:
        raise SecretStoreError("Invalid protected secret payload") from exc
    return _crypt_unprotect(protected).decode("utf-8")


def _require_windows() -> None:
    if os.name != "nt":
        raise SecretStoreError("DPAPI secret protection is only available on Windows")


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _blob_from_bytes(data: bytes) -> tuple[_DataBlob, ctypes.Array]:
    buffer = ctypes.create_string_buffer(data)
    return _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), buffer


def _raise_last_error(action: str) -> None:
    error_code = ctypes.get_last_error()
    raise SecretStoreError(f"{action} failed with Windows error {error_code}")


def _crypt_protect(data: bytes) -> bytes:
    _require_windows()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    data_blob, data_buffer = _blob_from_bytes(data)
    entropy_blob, entropy_buffer = _blob_from_bytes(_ENTROPY)
    out_blob = _DataBlob()

    # Keep buffers alive for the duration of the API call.
    _ = (data_buffer, entropy_buffer)
    ok = crypt32.CryptProtectData(
        ctypes.byref(data_blob),
        "Customer-Agent account secret",
        ctypes.byref(entropy_blob),
        None,
        None,
        0x01,  # CRYPTPROTECT_UI_FORBIDDEN
        ctypes.byref(out_blob),
    )
    if not ok:
        _raise_last_error("CryptProtectData")
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        kernel32.LocalFree(out_blob.pbData)


def _crypt_unprotect(data: bytes) -> bytes:
    _require_windows()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    data_blob, data_buffer = _blob_from_bytes(data)
    entropy_blob, entropy_buffer = _blob_from_bytes(_ENTROPY)
    out_blob = _DataBlob()

    _ = (data_buffer, entropy_buffer)
    ok = crypt32.CryptUnprotectData(
        ctypes.byref(data_blob),
        None,
        ctypes.byref(entropy_blob),
        None,
        None,
        0x01,  # CRYPTPROTECT_UI_FORBIDDEN
        ctypes.byref(out_blob),
    )
    if not ok:
        _raise_last_error("CryptUnprotectData")
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        kernel32.LocalFree(out_blob.pbData)
