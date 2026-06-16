import ast
from pathlib import Path


def _method_source(path: str, class_name: str, method_name: str) -> str:
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return ast.get_source_segment(source, child) or ""
    raise AssertionError(f"{class_name}.{method_name} not found")


def test_login_thread_closes_event_loop_in_finally():
    method = _method_source("ui/user_ui.py", "LoginThread", "run")

    assert "finally:" in method
    assert "loop.close()" in method


def test_sync_worker_closes_event_loop_in_finally():
    method = _method_source("ui/Knowledge_ui.py", "SyncWorker", "run")

    assert "finally:" in method
    assert "loop.close()" in method


def test_auto_reply_thread_handles_missing_loop_during_cleanup():
    method = _method_source("ui/auto_reply/threads.py", "AutoReplyThread", "run")

    assert 'getattr(self, "loop", None)' in method
    assert "loop.close()" in method
