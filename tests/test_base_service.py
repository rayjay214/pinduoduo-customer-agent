from core.base_service import BaseService


class CaptureLogger:
    def __init__(self):
        self.messages = []

    def error(self, message, **_kwargs):
        self.messages.append(str(message))


class DemoService(BaseService):
    def initialize(self) -> bool:
        return True

    def dispose(self):
        return None


def test_base_service_handle_exception_masks_sensitive_values():
    logger = CaptureLogger()
    service = DemoService(logger=logger)

    handled = service.handle_exception(RuntimeError("token=secret-token"), "启动失败")

    joined = "\n".join(logger.messages)
    assert handled is False
    assert "启动失败" in joined
    assert "secret-token" not in joined
    assert "token=***" in joined


def test_base_service_sanitizer_masks_common_sensitive_header_formats():
    from core.base_service import _sanitize_for_log

    sanitized = _sanitize_for_log(
        "Authorization: Bearer secret-bearer; Cookie: PASS_ID=secret-cookie; api_key=secret-api"
    )

    assert "secret-bearer" not in sanitized
    assert "secret-cookie" not in sanitized
    assert "secret-api" not in sanitized
    assert "Bearer" not in sanitized or "Bearer ***" in sanitized


def test_base_service_sanitizer_masks_nested_structures_and_exceptions():
    from core.base_service import _sanitize_for_log

    sanitized = _sanitize_for_log(
        {
            "items": [
                {
                    "password": "secret-password",
                    "meta": {"access_token": "secret-access"},
                    "message": RuntimeError("token=secret-token"),
                }
            ],
            "error": "cookies=secret-cookie",
        }
    )

    text = str(sanitized)
    assert "secret-password" not in text
    assert "secret-access" not in text
    assert "secret-token" not in text
    assert "secret-cookie" not in text
    assert sanitized["items"][0]["password"] == "***"
    assert sanitized["items"][0]["meta"]["access_token"] == "***"
    assert "RuntimeError: token=***" in sanitized["items"][0]["message"]
