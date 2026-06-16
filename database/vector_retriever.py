from __future__ import annotations

import hashlib
import asyncio
import json
import math
import os
import time
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests

from core.base_service import _sanitize_for_log
from utils.logger_loguru import get_logger


logger = get_logger("VectorRetriever")


def _coerce_finite_float(value: Any) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"non-finite float: {value!r}")
    return numeric


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return _coerce_finite_float(value)
    except ValueError:
        logger.warning(f"环境变量 {name}={_sanitize_for_log(value)!r} 不是有效数字，使用默认值 {default}")
        return default


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"环境变量 {name}={_sanitize_for_log(value)!r} 不是有效整数，使用默认值 {default}")
        return default


@dataclass(frozen=True)
class VectorItem:
    item_id: str
    text: str
    payload: Any


class VectorRetriever:
    def __init__(self) -> None:
        self.embedding_url = os.getenv(
            "CUSTOMER_AGENT_EMBEDDING_URL",
            os.getenv("CALLBACK_SERVER_EMBEDDING_URL", "http://127.0.0.1:8081/v1/embeddings"),
        )
        self.embedding_model = os.getenv(
            "CUSTOMER_AGENT_EMBEDDING_MODEL",
            os.getenv("CALLBACK_SERVER_EMBEDDING_MODEL", "bge-large-zh-v1.5-q8_0.gguf"),
        )
        self.timeout_seconds = _float_env("CUSTOMER_AGENT_EMBEDDING_TIMEOUT", 5.0)
        self.score_threshold = _float_env("CUSTOMER_AGENT_VECTOR_SCORE_THRESHOLD", 0.25)
        self.max_text_chars = _int_env("CUSTOMER_AGENT_EMBEDDING_MAX_TEXT_CHARS", 450)
        self.disable_seconds = _float_env("CUSTOMER_AGENT_EMBEDDING_DISABLE_SECONDS", 60.0)
        self._disabled_until = 0.0
        self.cache_dir = Path(
            os.getenv(
                "CUSTOMER_AGENT_VECTOR_INDEX_DIR",
                str(Path(__file__).resolve().parents[1] / "temp" / "vector_index"),
            )
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def rank(
        self,
        namespace: str,
        shop_id: int | str,
        query: str,
        items: Sequence[VectorItem],
        limit: int,
    ) -> list[Any]:
        clean_query = query.strip()
        if not clean_query or not items:
            return []
        if time.monotonic() < self._disabled_until:
            return []

        try:
            query_vector = self._embed(clean_query)
            vectors = self._load_or_build_vectors(namespace, shop_id, items)
        except Exception as exc:
            self._disabled_until = time.monotonic() + self.disable_seconds
            logger.warning(f"向量检索不可用，回退关键词检索: namespace={namespace}, error={_sanitize_for_log(exc)}")
            return []

        item_map = {item.item_id: item for item in items}
        scored: list[tuple[float, VectorItem]] = []
        for item_id, vector in vectors.items():
            item = item_map.get(item_id)
            if item is None:
                continue
            score = self._cosine_similarity(query_vector, vector)
            if score >= self.score_threshold:
                scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item.payload for _, item in scored[:limit]]

    async def rank_async(
        self,
        namespace: str,
        shop_id: int | str,
        query: str,
        items: Sequence[VectorItem],
        limit: int,
    ) -> list[Any]:
        """Async-safe wrapper for callers already running on an event loop."""
        return await asyncio.to_thread(
            self.rank,
            namespace,
            shop_id,
            query,
            items,
            limit,
        )

    async def embed_async(self, text: str) -> list[float]:
        """Async-safe wrapper around the synchronous embedding HTTP request."""
        return await asyncio.to_thread(self._embed, text)

    def _load_or_build_vectors(
        self,
        namespace: str,
        shop_id: int | str,
        items: Sequence[VectorItem],
    ) -> dict[str, list[float]]:
        cache_path = self._cache_path(namespace, shop_id)
        content_hash = self._content_hash(items)
        cached = self._load_cache(cache_path)
        if (
            cached.get("content_hash") == content_hash
            and cached.get("model") == self.embedding_model
            and isinstance(cached.get("vectors"), dict)
        ):
            try:
                return {
                    str(item_id): [_coerce_finite_float(value) for value in vector]
                    for item_id, vector in cached["vectors"].items()
                    if isinstance(vector, list)
                }
            except (TypeError, ValueError) as exc:
                logger.warning(f"向量缓存内容异常，丢弃并重建: path={cache_path}, error={_sanitize_for_log(exc)}")

        vectors = {}
        failed_count = 0
        for item in items:
            try:
                vectors[item.item_id] = self._embed(item.text)
            except Exception as exc:
                failed_count += 1
                logger.warning(f"单条知识向量化失败，已跳过: item_id={item.item_id}, error={_sanitize_for_log(exc)}")
        if failed_count:
            logger.warning(
                f"知识向量化部分失败，跳过完整缓存写入: namespace={namespace}, "
                f"shop_id={shop_id}, failed={failed_count}, total={len(items)}"
            )
            return vectors
        try:
            cache_path.write_text(
                json.dumps(
                    {
                        "model": self.embedding_model,
                        "content_hash": content_hash,
                        "vectors": vectors,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(f"写入向量缓存失败，继续使用本次内存向量: path={cache_path}, error={_sanitize_for_log(exc)}")
        return vectors

    def _embed(self, text: str) -> list[float]:
        input_text = text[: self.max_text_chars]
        response = requests.post(
            self.embedding_url,
            json={"model": self.embedding_model, "input": input_text},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError(f"embedding response missing vector: {body!r}")
        vector = body.get("embedding")
        data = body.get("data")
        if vector is None and isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                vector = first.get("embedding")
        embeddings = body.get("embeddings")
        if vector is None and isinstance(embeddings, list) and embeddings:
            vector = embeddings[0]
        if not isinstance(vector, list):
            raise RuntimeError(f"embedding response missing vector: {body!r}")
        try:
            return [_coerce_finite_float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"embedding response has non-finite vector value: {body!r}") from exc

    def _cache_path(self, namespace: str, shop_id: int | str) -> Path:
        key = hashlib.sha1(f"{shop_id}:{namespace}".encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{key}.json"

    def _content_hash(self, items: Sequence[VectorItem]) -> str:
        digest = hashlib.sha1()
        digest.update(self.embedding_model.encode("utf-8"))
        digest.update(str(self.max_text_chars).encode("utf-8"))
        for item in items:
            digest.update(item.item_id.encode("utf-8"))
            digest.update(item.text.encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _load_cache(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
        left_values = list(left)
        right_values = list(right)
        if not left_values or not right_values or len(left_values) != len(right_values):
            return 0.0
        numerator = sum(a * b for a, b in zip(left_values, right_values))
        left_norm = sqrt(sum(value * value for value in left_values))
        right_norm = sqrt(sum(value * value for value in right_values))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)
