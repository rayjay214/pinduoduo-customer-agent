from database import vector_retriever
from database.vector_retriever import VectorRetriever
from database.vector_retriever import VectorItem
import asyncio
import json
import pytest


def test_vector_retriever_invalid_numeric_env_falls_back_to_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("CUSTOMER_AGENT_EMBEDDING_TIMEOUT", "bad-timeout")
    monkeypatch.setenv("CUSTOMER_AGENT_VECTOR_SCORE_THRESHOLD", "bad-threshold")
    monkeypatch.setenv("CUSTOMER_AGENT_EMBEDDING_MAX_TEXT_CHARS", "bad-max")
    monkeypatch.setenv("CUSTOMER_AGENT_EMBEDDING_DISABLE_SECONDS", "bad-disable")
    monkeypatch.setenv("CUSTOMER_AGENT_VECTOR_INDEX_DIR", str(tmp_path))

    retriever = VectorRetriever()

    assert retriever.timeout_seconds == 5.0
    assert retriever.score_threshold == 0.25
    assert retriever.max_text_chars == 450
    assert retriever.disable_seconds == 60.0


def test_vector_retriever_invalid_numeric_env_masks_sensitive_values(monkeypatch, tmp_path):
    messages = []

    class FakeLogger:
        def warning(self, message):
            messages.append(str(message))

    monkeypatch.setattr(vector_retriever, "logger", FakeLogger())
    monkeypatch.setenv("CUSTOMER_AGENT_EMBEDDING_TIMEOUT", "token=secret-token")
    monkeypatch.setenv("CUSTOMER_AGENT_EMBEDDING_MAX_TEXT_CHARS", "api_key=secret-key")
    monkeypatch.setenv("CUSTOMER_AGENT_VECTOR_INDEX_DIR", str(tmp_path))

    retriever = VectorRetriever()

    assert retriever.timeout_seconds == 5.0
    assert retriever.max_text_chars == 450
    joined = "\n".join(messages)
    assert "secret-token" not in joined
    assert "secret-key" not in joined
    assert "token=***" in joined
    assert "api_key=***" in joined


def test_vector_retriever_non_finite_numeric_env_falls_back_to_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("CUSTOMER_AGENT_EMBEDDING_TIMEOUT", "nan")
    monkeypatch.setenv("CUSTOMER_AGENT_VECTOR_SCORE_THRESHOLD", "inf")
    monkeypatch.setenv("CUSTOMER_AGENT_EMBEDDING_MAX_TEXT_CHARS", "inf")
    monkeypatch.setenv("CUSTOMER_AGENT_EMBEDDING_DISABLE_SECONDS", "nan")
    monkeypatch.setenv("CUSTOMER_AGENT_VECTOR_INDEX_DIR", str(tmp_path))

    retriever = VectorRetriever()

    assert retriever.timeout_seconds == 5.0
    assert retriever.score_threshold == 0.25
    assert retriever.max_text_chars == 450
    assert retriever.disable_seconds == 60.0


def test_vector_retriever_non_finite_integer_env_does_not_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("CUSTOMER_AGENT_EMBEDDING_MAX_TEXT_CHARS", "inf")
    monkeypatch.setenv("CUSTOMER_AGENT_VECTOR_INDEX_DIR", str(tmp_path))

    retriever = VectorRetriever()

    assert retriever.max_text_chars == 450


def test_partial_vector_build_is_not_cached_as_complete(monkeypatch, tmp_path):
    monkeypatch.setenv("CUSTOMER_AGENT_VECTOR_INDEX_DIR", str(tmp_path))
    retriever = VectorRetriever()
    items = [
        VectorItem(item_id="good", text="可向量化", payload="good"),
        VectorItem(item_id="bad", text="临时失败", payload="bad"),
    ]

    def flaky_embed(text):
        if text == "临时失败":
            raise RuntimeError("temporary embedding failure")
        return [1.0, 0.0]

    monkeypatch.setattr(retriever, "_embed", flaky_embed)
    first_vectors = retriever._load_or_build_vectors("scene", "shop-1", items)
    assert set(first_vectors) == {"good"}
    assert not list(tmp_path.glob("*.json"))

    monkeypatch.setattr(retriever, "_embed", lambda _text: [1.0, 0.0])
    second_vectors = retriever._load_or_build_vectors("scene", "shop-1", items)
    assert set(second_vectors) == {"good", "bad"}


def test_rank_masks_sensitive_embedding_error_logs(monkeypatch, tmp_path):
    monkeypatch.setenv("CUSTOMER_AGENT_VECTOR_INDEX_DIR", str(tmp_path))
    retriever = VectorRetriever()
    monkeypatch.setattr(retriever, "_embed", lambda _text: (_ for _ in ()).throw(RuntimeError("token=secret-token")))
    messages = []

    class FakeLogger:
        def warning(self, message):
            messages.append(str(message))

    monkeypatch.setattr(vector_retriever, "logger", FakeLogger())

    result = retriever.rank(
        "scene",
        "shop-1",
        "query",
        [VectorItem(item_id="item-1", text="知识", payload="payload")],
        1,
    )

    joined = "\n".join(messages)
    assert result == []
    assert "secret-token" not in joined
    assert "token=***" in joined


def test_rank_async_runs_rank_in_background_thread(monkeypatch, tmp_path):
    monkeypatch.setenv("CUSTOMER_AGENT_VECTOR_INDEX_DIR", str(tmp_path))
    retriever = VectorRetriever()
    calls = []

    def fake_rank(namespace, shop_id, query, items, limit):
        calls.append((namespace, shop_id, query, items, limit))
        return ["payload"]

    monkeypatch.setattr(retriever, "rank", fake_rank)

    result = asyncio.run(
        retriever.rank_async(
            "scene",
            "shop-1",
            "query",
            [VectorItem(item_id="item-1", text="知识", payload="payload")],
            1,
        )
    )

    assert result == ["payload"]
    assert calls and calls[0][0] == "scene"


def test_embed_async_wraps_sync_embed(monkeypatch, tmp_path):
    monkeypatch.setenv("CUSTOMER_AGENT_VECTOR_INDEX_DIR", str(tmp_path))
    retriever = VectorRetriever()
    monkeypatch.setattr(retriever, "_embed", lambda text: [float(len(text)), 1.0])

    assert asyncio.run(retriever.embed_async("测试")) == [2.0, 1.0]


def test_invalid_cached_vector_values_are_rebuilt(monkeypatch, tmp_path):
    monkeypatch.setenv("CUSTOMER_AGENT_VECTOR_INDEX_DIR", str(tmp_path))
    retriever = VectorRetriever()
    items = [VectorItem(item_id="item-1", text="测试知识", payload="payload")]
    cache_path = retriever._cache_path("scene", "shop-1")
    cache_path.write_text(
        json.dumps(
            {
                "model": retriever.embedding_model,
                "content_hash": retriever._content_hash(items),
                "vectors": {"item-1": ["not-a-float"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    calls = []

    def fake_embed(text):
        calls.append(text)
        return [1.0, 0.0]

    monkeypatch.setattr(retriever, "_embed", fake_embed)

    vectors = retriever._load_or_build_vectors("scene", "shop-1", items)

    assert vectors == {"item-1": [1.0, 0.0]}
    assert calls == ["测试知识"]


def test_embed_empty_embedding_lists_raise_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("CUSTOMER_AGENT_VECTOR_INDEX_DIR", str(tmp_path))
    retriever = VectorRetriever()

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [], "embeddings": []}

    monkeypatch.setattr(
        "database.vector_retriever.requests.post",
        lambda *_args, **_kwargs: FakeResponse(),
    )

    with pytest.raises(RuntimeError, match="missing vector"):
        retriever._embed("测试")


def test_embed_non_dict_response_raises_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("CUSTOMER_AGENT_VECTOR_INDEX_DIR", str(tmp_path))
    retriever = VectorRetriever()

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return ["bad"]

    monkeypatch.setattr(
        "database.vector_retriever.requests.post",
        lambda *_args, **_kwargs: FakeResponse(),
    )

    with pytest.raises(RuntimeError, match="missing vector"):
        retriever._embed("测试")


def test_embed_non_finite_vector_values_raise_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("CUSTOMER_AGENT_VECTOR_INDEX_DIR", str(tmp_path))
    retriever = VectorRetriever()

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"embedding": [1.0, "nan"]}

    monkeypatch.setattr(
        "database.vector_retriever.requests.post",
        lambda *_args, **_kwargs: FakeResponse(),
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        retriever._embed("测试")
