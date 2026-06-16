import ast
from pathlib import Path


def test_init_fetches_token_via_to_thread():
    source = Path("Channel/pinduoduo/core/pdd_lifecycle.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Await):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "to_thread"
            and isinstance(func.value, ast.Name)
            and func.value.id == "asyncio"
            and call.args
            and isinstance(call.args[0], ast.Attribute)
            and call.args[0].attr == "get_token"
        ):
            found = True
            break

    assert found is True
