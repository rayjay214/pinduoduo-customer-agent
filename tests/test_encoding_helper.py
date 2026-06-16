from utils import encoding_helper
from utils.encoding_helper import EncodingConverter


def test_ensure_utf8_masks_sensitive_conversion_exception(monkeypatch):
    messages = []

    class FakeLogger:
        def warning(self, message):
            messages.append(str(message))

    def broken_open(*_args, **_kwargs):
        raise RuntimeError("api_key=secret-token")

    monkeypatch.setattr(encoding_helper, "logger", FakeLogger())
    monkeypatch.setattr("builtins.open", broken_open)

    path, encoding = EncodingConverter.ensure_utf8("input.txt")

    joined = "\n".join(messages)
    assert path == "input.txt"
    assert encoding == "unknown"
    assert "编码转换失败" in joined
    assert "secret-token" not in joined
    assert "api_key=***" in joined
