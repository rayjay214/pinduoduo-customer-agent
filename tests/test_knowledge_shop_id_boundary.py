import ast
from pathlib import Path

from database.knowledge_service import KnowledgeService


def test_resolve_shop_id_does_not_query_internal_primary_key_for_external_string(monkeypatch):
    calls = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def scalar(self, _stmt):
            return None

        def get(self, model, key):
            calls.append((model, key))
            return None

    service = object.__new__(KnowledgeService)
    service.get_session = lambda: FakeSession()

    assert service._resolve_shop_id("shop-1") == "shop-1"
    assert calls == []


def test_resolve_shop_id_allows_numeric_internal_primary_key_fallback(monkeypatch):
    class ShopRow:
        id = 7

    calls = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def scalar(self, _stmt):
            return None

        def get(self, model, key):
            calls.append((model, key))
            return ShopRow()

    service = object.__new__(KnowledgeService)
    service.get_session = lambda: FakeSession()

    assert service._resolve_shop_id("7") == 7
    assert calls[0][1] == 7


def test_sync_worker_preserves_external_shop_id_string():
    source = Path("ui/Knowledge_ui.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    offending_calls = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "int":
            continue
        if not node.args:
            continue
        arg = node.args[0]
        if (
            isinstance(arg, ast.Attribute)
            and arg.attr == "pdd_shop_id"
            and isinstance(arg.value, ast.Name)
            and arg.value.id == "self"
        ):
            offending_calls.append(node)

    assert offending_calls == []
