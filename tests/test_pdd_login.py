import asyncio

import pytest

from Channel.pinduoduo import pdd_login as pdd_login_module


class _FakeContext:
    def __init__(self):
        self.closed = False

    async def new_page(self):
        raise RuntimeError("page creation failed")

    async def close(self):
        self.closed = True


class _CaptureLogger:
    def __init__(self):
        self.messages = []

    def debug(self, message):
        self.messages.append(str(message))

    def info(self, message):
        self.messages.append(str(message))

    def warning(self, message):
        self.messages.append(str(message))

    def error(self, message):
        self.messages.append(str(message))


class _FakeRefreshPage:
    async def goto(self, _url):
        return None

    async def wait_for_url(self, *_args, **_kwargs):
        raise asyncio.TimeoutError()


class _FakeChromium:
    def __init__(self, context):
        self.context = context
        self.launch_kwargs = None

    async def launch_persistent_context(self, *_args, **kwargs):
        self.launch_kwargs = kwargs
        return self.context


class _FakePlaywright:
    def __init__(self, context):
        self.chromium = _FakeChromium(context)
        self.stopped = False

    async def stop(self):
        self.stopped = True


class _FakePlaywrightStarter:
    def __init__(self, playwright):
        self.playwright = playwright

    async def start(self):
        return self.playwright


def test_login_closes_playwright_resources_when_login_flow_fails(monkeypatch):
    context = _FakeContext()
    playwright = _FakePlaywright(context)

    monkeypatch.setattr(
        pdd_login_module,
        "async_playwright",
        lambda: _FakePlaywrightStarter(playwright),
    )

    login = pdd_login_module.PDDLogin(name="demo-shop", password="secret")

    assert asyncio.run(login.login()) is False
    assert context.closed is True
    assert playwright.stopped is True


def test_login_masks_sensitive_values_in_login_flow_error_logs(monkeypatch):
    class SensitiveContext(_FakeContext):
        async def new_page(self):
            raise RuntimeError("token=secret-token")

    context = SensitiveContext()
    playwright = _FakePlaywright(context)

    monkeypatch.setattr(
        pdd_login_module,
        "async_playwright",
        lambda: _FakePlaywrightStarter(playwright),
    )

    logger = _CaptureLogger()
    login = pdd_login_module.PDDLogin(name="demo-shop", password="secret")
    login.logger = logger

    assert asyncio.run(login.login()) is False
    assert "secret-token" not in "\n".join(logger.messages)


def test_login_browser_args_do_not_disable_web_security(monkeypatch):
    context = _FakeContext()
    playwright = _FakePlaywright(context)

    monkeypatch.setattr(
        pdd_login_module,
        "async_playwright",
        lambda: _FakePlaywrightStarter(playwright),
    )

    login = pdd_login_module.PDDLogin(name="demo-shop", password="secret")

    assert asyncio.run(login.login()) is False
    args = playwright.chromium.launch_kwargs["args"]
    assert "--disable-web-security" not in args


def test_login_browser_args_do_not_disable_sandbox(monkeypatch):
    context = _FakeContext()
    playwright = _FakePlaywright(context)

    monkeypatch.setattr(
        pdd_login_module,
        "async_playwright",
        lambda: _FakePlaywrightStarter(playwright),
    )

    login = pdd_login_module.PDDLogin(name="demo-shop", password="secret")

    assert asyncio.run(login.login()) is False
    args = playwright.chromium.launch_kwargs["args"]
    assert "--no-sandbox" not in args


def test_refresh_browser_args_do_not_disable_sandbox(monkeypatch, tmp_path):
    class RefreshContext(_FakeContext):
        async def new_page(self):
            return _FakeRefreshPage()

        async def cookies(self):
            return [{"name": "pdd_user_id", "value": "user-1"}]

    user_dir = tmp_path / "user_data" / "demo-shop"
    user_dir.mkdir(parents=True)
    context = RefreshContext()
    playwright = _FakePlaywright(context)

    monkeypatch.setattr(
        pdd_login_module,
        "async_playwright",
        lambda: _FakePlaywrightStarter(playwright),
    )
    monkeypatch.setattr(pdd_login_module, "app_dir", tmp_path)

    login = pdd_login_module.PDDLogin(name="demo-shop", password="secret")

    assert asyncio.run(login.refresh_cookies()) == '{"pdd_user_id": "user-1"}'
    args = playwright.chromium.launch_kwargs["args"]
    assert "--no-sandbox" not in args
    assert context.closed is True


def test_refresh_cookies_keeps_success_when_context_close_fails(monkeypatch, tmp_path):
    class RefreshContext(_FakeContext):
        async def new_page(self):
            return _FakeRefreshPage()

        async def cookies(self):
            return [{"name": "pdd_user_id", "value": "user-1"}]

        async def close(self):
            raise RuntimeError("close failed")

    user_dir = tmp_path / "user_data" / "demo-shop"
    user_dir.mkdir(parents=True)
    context = RefreshContext()
    playwright = _FakePlaywright(context)

    monkeypatch.setattr(
        pdd_login_module,
        "async_playwright",
        lambda: _FakePlaywrightStarter(playwright),
    )
    monkeypatch.setattr(pdd_login_module, "app_dir", tmp_path)

    login = pdd_login_module.PDDLogin(name="demo-shop", password="secret")

    assert asyncio.run(login.refresh_cookies()) == '{"pdd_user_id": "user-1"}'
    assert playwright.stopped is True


def test_cookies_list_to_json_skips_malformed_cookie_items():
    cookies_json = pdd_login_module._cookies_list_to_json(
        [
            "bad",
            {"name": "", "value": "ignored"},
            {"name": "pdd_user_id", "value": "user-1"},
            {"name": "empty_value"},
        ]
    )

    assert cookies_json == '{"pdd_user_id": "user-1", "empty_value": ""}'


def test_login_pdd_masks_sensitive_values_in_post_login_error_logs(monkeypatch):
    logger = _CaptureLogger()

    class FakeLogin:
        channel_name = "pinduoduo"

        def __init__(self, name, password):
            self.name = name
            self.password = password
            self.logger = logger

        async def login(self):
            return '{"pdd_user_id":"user-1"}'

        def Set_user_info(self, _cookies_json):
            raise RuntimeError("token=secret-token")

    monkeypatch.setattr(pdd_login_module, "PDDLogin", FakeLogin)

    assert asyncio.run(pdd_login_module.login_pdd("demo-shop", "secret")) is False
    assert "secret-token" not in "\n".join(logger.messages)


def test_refresh_pdd_cookies_masks_sensitive_values_in_post_refresh_error_logs(monkeypatch):
    logger = _CaptureLogger()

    class FakeLogin:
        channel_name = "pinduoduo"

        def __init__(self, name, password):
            self.name = name
            self.password = password
            self.logger = logger

        async def refresh_cookies(self):
            return '{"pdd_user_id":"user-1"}'

        def Set_user_info(self, _cookies_json):
            raise RuntimeError("cookies=secret-cookie")

    monkeypatch.setattr(pdd_login_module, "PDDLogin", FakeLogin)

    assert asyncio.run(pdd_login_module.refresh_pdd_cookies("demo-shop", "secret")) is False
    assert "secret-cookie" not in "\n".join(logger.messages)
