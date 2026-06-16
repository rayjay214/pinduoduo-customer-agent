from Channel.pinduoduo.utils import base_request as base_request_module
from Channel.pinduoduo.utils.API.Set_up_online import AccountMonitor
from Channel.pinduoduo.utils.API.get_shop_info import GetShopInfo
from Channel.pinduoduo.utils.API.get_user_info import GetUserInfo


class _DummyLogger:
    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


class CaptureLogger:
    def __init__(self):
        self.messages = []

    def warning(self, message):
        self.messages.append(str(message))

    def error(self, message):
        self.messages.append(str(message))


class DummyGetShopInfo(GetShopInfo):
    def __init__(self, post_result):
        self._post_result = post_result
        self.logger = _DummyLogger()

    def post(self, *args, **kwargs):
        return self._post_result


class DummyGetUserInfo(GetUserInfo):
    def __init__(self, post_result):
        self._post_result = post_result
        self.logger = _DummyLogger()

    def post(self, *args, **kwargs):
        return self._post_result


class DummyAccountMonitor(AccountMonitor):
    def __init__(self, post_result):
        self._post_result = post_result
        self.logger = _DummyLogger()

    def post(self, *args, **kwargs):
        return self._post_result


def test_api_helpers_invalid_cookies_do_not_overwrite_existing_cookies():
    helpers = [
        GetShopInfo(),
        GetUserInfo(),
        AccountMonitor(),
    ]

    for helper in helpers:
        helper.cookies = {"existing": "cookie"}
        assert helper.update_cookies("{bad json") is False
        assert helper.cookies == {"existing": "cookie"}


def test_api_helpers_constructor_warns_when_cookie_string_is_invalid(monkeypatch):
    warnings = []

    class FakeLogger:
        def warning(self, message):
            warnings.append(message)

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    monkeypatch.setattr(base_request_module, "get_logger", lambda _name: FakeLogger())

    GetShopInfo(cookies="{bad json")
    GetUserInfo(cookies="{bad json")
    AccountMonitor(cookies="{bad json")

    assert len(warnings) == 3
    assert all("cookies 无效" in message for message in warnings)


def test_shop_and_user_info_return_false_for_malformed_success_result():
    assert DummyGetShopInfo({"success": True, "result": None}).get_shop_info() is False
    assert DummyGetShopInfo({"success": True, "result": []}).get_shop_info() is False
    assert DummyGetUserInfo({"success": True, "result": None}).get_user_info() is False
    assert DummyGetUserInfo({"success": True, "result": []}).get_user_info() is False


def test_shop_and_user_info_treat_string_true_success_as_success():
    shop_result = DummyGetShopInfo(
        {
            "success": "true",
            "result": {
                "mallId": "shop-1",
                "mallName": "Demo Shop",
                "mallLogo": "https://example.test/logo.png",
            },
        }
    ).get_shop_info()
    user_result = DummyGetUserInfo(
        {
            "success": 1,
            "result": {
                "id": "user-1",
                "username": "CS",
                "mall_id": "shop-1",
            },
        }
    ).get_user_info()

    assert shop_result == ("shop-1", "Demo Shop", "https://example.test/logo.png")
    assert user_result == ("user-1", "CS", "shop-1")


def test_api_helpers_return_false_for_non_dict_top_level_response():
    assert DummyGetShopInfo(["bad"]).get_shop_info() is False
    assert DummyGetUserInfo("bad").get_user_info() is False
    assert DummyAccountMonitor(["bad"]).set_csstatus("ONLINE") is False


def test_account_monitor_treats_string_true_success_as_success():
    assert DummyAccountMonitor({"success": "true"}).set_csstatus("ONLINE") is True


def test_api_helpers_mask_sensitive_values_in_platform_error_logs():
    helpers = [
        (DummyGetShopInfo({"success": False, "errorMsg": "token=secret-token"}), "get_shop_info", ()),
        (DummyGetUserInfo({"success": False, "errorMsg": "cookies=secret-cookie"}), "get_user_info", ()),
        (DummyAccountMonitor({"success": False, "errorMsg": "anti-content=secret-anti"}), "set_csstatus", ("ONLINE",)),
    ]

    for helper, method_name, args in helpers:
        logger = CaptureLogger()
        helper.logger = logger

        assert getattr(helper, method_name)(*args) is False
        log_text = "\n".join(logger.messages)
        assert "secret-token" not in log_text
        assert "secret-cookie" not in log_text
        assert "secret-anti" not in log_text


def test_api_helpers_do_not_log_raw_platform_error_values():
    helpers = [
        (DummyGetShopInfo({"success": False, "errorMsg": "客户手机号13800138000，地址测试小区1号楼"}), "get_shop_info", ()),
        (DummyGetUserInfo({"success": False, "errorMsg": "客户手机号13800138000，地址测试小区1号楼"}), "get_user_info", ()),
        (DummyAccountMonitor({"success": False, "errorMsg": "客户手机号13800138000，地址测试小区1号楼"}), "set_csstatus", ("ONLINE",)),
    ]

    for helper, method_name, args in helpers:
        logger = CaptureLogger()
        helper.logger = logger

        assert getattr(helper, method_name)(*args) is False
        log_text = "\n".join(logger.messages)
        assert "13800138000" not in log_text
        assert "测试小区" not in log_text
        assert "error_chars=" in log_text
