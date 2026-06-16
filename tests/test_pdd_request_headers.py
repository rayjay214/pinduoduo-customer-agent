import json

from Channel.pinduoduo.utils import base_request as base_request_module
from Channel.pinduoduo.utils.base_request import BaseRequest


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class FakeInvalidJsonResponse(FakeResponse):
    def json(self):
        raise json.JSONDecodeError("bad json", self.text, 0)


class RequestSequence:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.responses.pop(0)


def test_mms_headers_do_not_read_anti_content_from_cookies():
    request = BaseRequest()
    request.cookies = {
        "anti_content": "stale-cookie-value",
        "anti-content": "other-stale-cookie-value",
    }

    headers = request._build_mms_browser_headers(
        url="https://mms.pinduoduo.com/plateau/message/send/mallGoodsCard",
        payload={"goods_id": 1},
        require_anti_content=True,
    )

    assert "anti-content" not in headers


def test_mms_headers_use_injected_anti_content_provider():
    request = BaseRequest()
    seen = {}

    def provider(url, payload):
        seen["url"] = url
        seen["payload"] = payload
        return "dynamic-token"

    request.set_anti_content_provider(provider)

    headers = request._build_mms_browser_headers(
        url="https://mms.pinduoduo.com/plateau/message/send/mallGoodsCard",
        payload={"goods_id": 1},
        require_anti_content=True,
    )

    assert headers["anti-content"] == "dynamic-token"
    assert seen == {
        "url": "https://mms.pinduoduo.com/plateau/message/send/mallGoodsCard",
        "payload": {"goods_id": 1},
    }


def test_mms_headers_read_browser_fingerprint_from_config(monkeypatch):
    def fake_get_config(key, default=None):
        overrides = {
            "pinduoduo.request.user_agent": "Custom UA",
            "pinduoduo.request.sec_ch_ua": '"Custom";v="1"',
            "pinduoduo.request.sec_ch_ua_mobile": "?1",
            "pinduoduo.request.sec_ch_ua_platform": '"Android"',
        }
        return overrides.get(key, default)

    import config

    monkeypatch.setattr(config, "get_config", fake_get_config)

    request = BaseRequest()
    headers = request._merge_headers(
        request._build_mms_browser_headers(
            url="https://mms.pinduoduo.com/latitude/goods/recommendGoods",
            payload={},
        )
    )

    assert headers["User-Agent"] == "Custom UA"
    assert headers["sec-ch-ua"] == '"Custom";v="1"'
    assert headers["sec-ch-ua-mobile"] == "?1"
    assert headers["sec-ch-ua-platform"] == '"Android"'


def test_header_helpers_ignore_non_mapping_overrides():
    request = BaseRequest()

    merged = request._merge_headers(["bad-header"])
    built = request._build_mms_browser_headers(
        url="https://mms.pinduoduo.com/latitude/goods/recommendGoods",
        payload={},
        extra_headers=["bad-header"],
    )

    assert "User-Agent" in merged
    assert built["Accept"] == "application/json, text/plain, */*"


def test_init_account_info_sanitizes_database_exception_logs(monkeypatch):
    class BrokenDBManager:
        def get_account(self, *args):
            raise RuntimeError(
                "cookies=secret-cookie Authorization: Bearer secret-token"
            )

    class CapturingLogger:
        def __init__(self):
            self.messages = []

        def error(self, message, *args, **kwargs):
            self.messages.append(message)

        def warning(self, *args, **kwargs):
            pass

    logger = CapturingLogger()

    monkeypatch.setattr(base_request_module, "db_manager", BrokenDBManager())
    monkeypatch.setattr(base_request_module, "get_logger", lambda *_args, **_kwargs: logger)

    BaseRequest(shop_id="shop-1", user_id="user-1")

    assert logger.messages
    log_text = "\n".join(logger.messages)
    assert "secret-cookie" not in log_text
    assert "secret-token" not in log_text
    assert "cookies=***" in log_text
    assert "Authorization: Bearer ***" in log_text


def test_apply_new_cookies_rejects_invalid_json_without_persisting(monkeypatch):
    persisted = []

    class FakeDBManager:
        def update_account_cookies(self, *args):
            persisted.append(args)
            return True

    monkeypatch.setattr(base_request_module, "db_manager", FakeDBManager())

    request = BaseRequest()
    request.channel_name = "pinduoduo"
    request.shop_id = "shop-1"
    request.user_id = "user-1"
    request.cookies = {"existing": "cookie"}

    assert request._apply_new_cookies("{bad json") is False
    assert request.cookies == {"existing": "cookie"}
    assert persisted == []


def test_apply_new_cookies_keeps_memory_when_persist_fails(monkeypatch):
    class FakeDBManager:
        def update_account_cookies(self, *args):
            return False

    monkeypatch.setattr(base_request_module, "db_manager", FakeDBManager())

    request = BaseRequest()
    request.channel_name = "pinduoduo"
    request.shop_id = "shop-1"
    request.user_id = "user-1"
    request.cookies = {"existing": "cookie"}

    assert request._apply_new_cookies('{"fresh":"cookie"}') is False
    assert request.cookies == {"existing": "cookie"}


def test_relogin_returns_false_when_refreshed_cookies_cannot_be_applied(monkeypatch):
    request = BaseRequest()
    request.account_name = "demo"
    request._get_account_credentials = lambda: ("demo", "password")
    request._run_async_login_func = lambda _func, *_args: {"cookies": {"fresh": "cookie"}}
    request._apply_new_cookies = lambda _cookies: False

    class FakeLoginModule:
        async def refresh_pdd_cookies(self, *_args):
            return {"cookies": {"fresh": "cookie"}}

        async def login_pdd(self, *_args):
            return {"cookies": {"fresh": "cookie"}}

    monkeypatch.setattr(
        base_request_module.importlib,
        "import_module",
        lambda _name: FakeLoginModule(),
    )

    assert request._relogin_and_update_cookies() is False


def test_force_refresh_returns_false_when_cookies_cannot_be_applied(monkeypatch):
    request = BaseRequest()
    request.shop_id = "shop-1"
    request.user_id = "user-1"
    request.account_name = "demo"
    request._get_account_credentials = lambda: ("demo", "password")
    request._run_async_login_func = lambda _func, *_args: {"cookies": {"fresh": "cookie"}}
    request._apply_new_cookies = lambda _cookies: False

    class FakeLoginModule:
        async def refresh_pdd_cookies(self, *_args):
            return {"cookies": {"fresh": "cookie"}}

    monkeypatch.setattr(
        base_request_module.importlib,
        "import_module",
        lambda _name: FakeLoginModule(),
    )

    assert request.force_refresh_cookies() is False


def test_handle_response_rejects_json_array_payload():
    request = BaseRequest()

    result = request._handle_response(
        FakeResponse(status_code=200, payload=[], text="[]"),
        expect_json=True,
    )

    assert result is None


def test_session_expired_detection_accepts_nested_result_error():
    request = BaseRequest()

    assert request._is_session_expired(
        {
            "success": False,
            "result": {
                "error_code": BaseRequest.SESSION_EXPIRED_ERROR_CODE,
                "error": "会话已过期，请重新登录",
            },
        }
    ) is True


def test_session_expired_detection_accepts_camelcase_fields():
    request = BaseRequest()

    assert request._is_session_expired(
        {
            "errorCode": BaseRequest.SESSION_EXPIRED_ERROR_CODE,
            "errorMsg": "会话已过期",
        }
    ) is True


def test_session_expired_detection_ignores_non_dict_payload():
    request = BaseRequest()

    assert request._is_session_expired(["bad"]) is False


def test_sanitize_for_log_masks_nested_sensitive_values_in_lists():
    request = BaseRequest()

    sanitized = request._sanitize_for_log(
        {
            "items": [
                [
                    {
                        "password": "secret",
                        "meta": {"access_token": "token-value"},
                    }
                ]
            ],
            1: {"token": "numeric-key-token"},
        }
    )

    assert sanitized["items"][0][0]["password"] == "***"
    assert sanitized["items"][0][0]["meta"]["access_token"] == "***"
    assert sanitized[1]["token"] == "***"


def test_sanitize_for_log_masks_sensitive_values_embedded_in_plain_strings():
    request = BaseRequest()

    sanitized = request._sanitize_for_log(
        {
            "error_msg": "platform rejected token=secret-token",
            "details": ["cookies=secret-cookie", {"message": "anti-content: secret-anti"}],
        }
    )

    text = str(sanitized)
    assert "secret-token" not in text
    assert "secret-cookie" not in text
    assert "secret-anti" not in text
    assert sanitized["error_msg"] == "platform rejected token=***"


def test_sanitize_for_log_masks_sensitive_plain_text_inside_values():
    request = BaseRequest()

    sanitized = request._sanitize_for_log(
        {
            "error_msg": "platform rejected: token=secret-token; cookies=secret-cookie",
            "items": ["anti-content=secret-anti-content"],
        }
    )

    assert "secret-token" not in sanitized["error_msg"]
    assert "secret-cookie" not in sanitized["error_msg"]
    assert "secret-anti-content" not in sanitized["items"][0]
    assert "token=***" in sanitized["error_msg"]
    assert "cookies=***" in sanitized["error_msg"]
    assert sanitized["items"][0] == "anti-content=***"


def test_handle_response_masks_sensitive_json_response_text_in_logs():
    messages = []
    request = BaseRequest()
    request.logger = type(
        "Logger",
        (),
        {"error": lambda _self, message: messages.append(str(message))},
    )()

    result = request._handle_response(
        FakeResponse(status_code=500, text='{"token":"secret-token","error":"failed"}'),
        expect_json=True,
    )

    assert result is None
    assert messages
    assert "secret-token" not in "\n".join(messages)


def test_handle_response_does_not_log_raw_error_response_text():
    messages = []
    request = BaseRequest()
    request.logger = type(
        "Logger",
        (),
        {"error": lambda _self, message: messages.append(str(message))},
    )()

    result = request._handle_response(
        FakeResponse(status_code=500, text="客户手机号13800138000，地址测试小区1号楼"),
        expect_json=True,
    )

    joined = "\n".join(messages)
    assert result is None
    assert "13800138000" not in joined
    assert "测试小区" not in joined
    assert "response_chars=" in joined


def test_handle_response_does_not_log_raw_invalid_json_text():
    messages = []
    request = BaseRequest()
    request.logger = type(
        "Logger",
        (),
        {"error": lambda _self, message: messages.append(str(message))},
    )()

    result = request._handle_response(
        FakeInvalidJsonResponse(
            status_code=200,
            text="不是JSON，客户手机号13800138000，地址测试小区1号楼",
        ),
        expect_json=True,
    )

    joined = "\n".join(messages)
    assert result is None
    assert "13800138000" not in joined
    assert "测试小区" not in joined
    assert "response_chars=" in joined


def test_handle_response_does_not_log_raw_non_object_json_text():
    messages = []
    request = BaseRequest()
    request.logger = type(
        "Logger",
        (),
        {"error": lambda _self, message: messages.append(str(message))},
    )()

    result = request._handle_response(
        FakeResponse(
            status_code=200,
            payload=["客户手机号13800138000"],
            text='["客户手机号13800138000"]',
        ),
        expect_json=True,
    )

    joined = "\n".join(messages)
    assert result is None
    assert "13800138000" not in joined
    assert "response_type=list" in joined
    assert "response_chars=" in joined


def test_sanitize_text_for_log_masks_sensitive_plain_text_values():
    request = BaseRequest()

    sanitized = request._sanitize_text_for_log(
        "token=secret-token&cookies=secret-cookie&error=failed"
    )

    assert "secret-token" not in sanitized
    assert "secret-cookie" not in sanitized
    assert "token=***" in sanitized
    assert "cookies=***" in sanitized


def test_sanitize_text_for_log_masks_sensitive_malformed_json_fragments():
    request = BaseRequest()

    sanitized = request._sanitize_text_for_log(
        '{"shop_id": "shop-1", "token": "secret-token",'
    )

    assert "secret-token" not in sanitized
    assert '"token": "***"' in sanitized


def test_sanitize_for_log_masks_common_sensitive_key_aliases():
    request = BaseRequest()

    sanitized = request._sanitize_for_log(
        {
            "headers": {
                "Authorization": "Bearer secret-bearer",
                "Cookie": "PASS_ID=secret-cookie",
                "accessToken": "secret-access",
                "antiContent": "secret-anti",
                "x-api-key": "secret-api-key",
            },
            "message": (
                "Authorization: Bearer secret-bearer; "
                "Cookie: PASS_ID=secret-cookie; "
                "accessToken=secret-access; "
                "antiContent: secret-anti; "
                "x-api-key=secret-api-key"
            ),
        }
    )

    text = str(sanitized)
    assert "secret-bearer" not in text
    assert "secret-cookie" not in text
    assert "secret-access" not in text
    assert "secret-anti" not in text
    assert "secret-api-key" not in text


def test_execute_with_retry_masks_sensitive_exception_text_in_logs(monkeypatch):
    messages = []
    request = BaseRequest(max_retries=1)
    request.logger = type(
        "Logger",
        (),
        {
            "warning": lambda _self, message: messages.append(str(message)),
            "error": lambda _self, message: messages.append(str(message)),
        },
    )()
    monkeypatch.setattr(request, "_calculate_retry_delay", lambda _attempt: 0)

    def failing_request():
        raise base_request_module.requests.ConnectionError("token=secret-token")

    result = request._execute_with_retry(failing_request)

    assert result is None
    assert messages
    assert "secret-token" not in "\n".join(messages)


def test_execute_with_retry_retries_after_nested_session_expired(monkeypatch):
    request = BaseRequest(shop_id=None, user_id=None)
    request.shop_id = "shop-1"
    request.user_id = "user-1"
    relogin_calls = []
    request._relogin_and_update_cookies = lambda: relogin_calls.append(True) or True
    monkeypatch.setattr(request, "_calculate_retry_delay", lambda _attempt: 0)

    sequence = RequestSequence(
        FakeResponse(
            payload={
                "success": False,
                "result": {
                    "error_code": BaseRequest.SESSION_EXPIRED_ERROR_CODE,
                    "error": "会话已过期，请重新登录",
                },
            },
            text="expired",
        ),
        FakeResponse(payload={"success": True, "result": {"ok": True}}, text="ok"),
    )

    result = request._execute_with_retry(sequence)

    assert result == {"success": True, "result": {"ok": True}}
    assert relogin_calls == [True]
    assert sequence.calls == 2


def test_get_account_credentials_rejects_non_dict_account_info(monkeypatch):
    class FakeDBManager:
        def get_account(self, *_args):
            return ["bad"]

    monkeypatch.setattr(base_request_module, "db_manager", FakeDBManager())

    request = BaseRequest(shop_id=None, user_id=None)
    request.channel_name = "pinduoduo"
    request.shop_id = "shop-1"
    request.user_id = "user-1"

    assert request._get_account_credentials() is None
