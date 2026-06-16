from Channel.pinduoduo.utils.API.get_token import GetToken


class DummyGetToken(GetToken):
    def __init__(self, post_result):
        self._post_result = post_result
        self.account_name = "demo"
        self.logger = type(
            "Logger",
            (),
            {
                "error": lambda *args, **kwargs: None,
            },
        )()

    def post(self, *args, **kwargs):
        return self._post_result


def test_get_token_reads_top_level_token():
    assert DummyGetToken({"token": "top"}).get_token() == "top"


def test_get_token_reads_nested_result_token():
    assert DummyGetToken({"result": {"token": "nested"}}).get_token() == "nested"


def test_get_token_returns_none_for_malformed_result_payload():
    assert DummyGetToken({"result": None}).get_token() is None
    assert DummyGetToken({"result": []}).get_token() is None


def test_get_token_returns_none_for_non_dict_response():
    assert DummyGetToken(["token"]).get_token() is None
    assert DummyGetToken("unexpected").get_token() is None


def test_get_token_ignores_empty_top_level_token_and_reads_nested_token():
    assert DummyGetToken({"token": "", "result": {"token": "nested"}}).get_token() == "nested"


def test_get_token_masks_token_values_in_error_logs():
    messages = []
    sender = DummyGetToken({"token": "", "result": {"token": "", "error": "missing"}})
    sender.logger = type(
        "Logger",
        (),
        {"error": lambda _self, message: messages.append(str(message))},
    )()

    assert sender.get_token() is None
    assert messages
    assert "'token': ''" not in "\n".join(messages)


def test_get_token_masks_plain_text_token_response_in_error_logs():
    messages = []
    sender = DummyGetToken("token=secret-token&error=missing")
    sender.logger = type(
        "Logger",
        (),
        {"error": lambda _self, message: messages.append(str(message))},
    )()

    assert sender.get_token() is None
    assert messages
    assert "secret-token" not in "\n".join(messages)
