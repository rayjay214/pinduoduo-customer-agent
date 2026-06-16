from __future__ import annotations

from urllib.parse import urlparse


_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".avif")
_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".3gp")
_TRAILING_URL_PUNCTUATION = " \t\r\n，。；、！？,.!?;:)]）}\"'"
_TRAILING_ENCODED_QUOTES = ("%22", "%27", "%60", "%e2%80%9c", "%e2%80%9d", "%e2%80%98", "%e2%80%99")


def normalize_media_url(media_url: str) -> str:
    """Remove wrapper punctuation that chat payloads sometimes attach to media URLs."""
    url = str(media_url or "").strip()
    while url:
        before = url
        url = url.rstrip(_TRAILING_URL_PUNCTUATION)
        lowered = url.lower()
        for suffix in _TRAILING_ENCODED_QUOTES:
            if lowered.endswith(suffix):
                url = url[: -len(suffix)]
                break
        if url == before:
            break
    return url


def infer_media_type_from_url(media_url: str) -> str:
    """Infer media type from stable URL shape, not arbitrary words in a link."""
    url = normalize_media_url(media_url).lower()
    if not url:
        return ""
    if url.startswith("data:image/"):
        return "image"
    if url.startswith("data:video/"):
        return "video"

    parsed = urlparse(url)
    path = parsed.path or url
    if any(path.endswith(ext) for ext in _IMAGE_EXTENSIONS):
        return "image"
    if any(path.endswith(ext) for ext in _VIDEO_EXTENSIONS):
        return "video"

    path_parts = [part for part in path.split("/") if part]
    if any(part in {"chat-img", "image", "images", "img", "photo", "photos"} for part in path_parts):
        return "image"
    if any(part in {"chat-video", "video", "videos"} for part in path_parts):
        return "video"
    return ""
