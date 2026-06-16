from Channel.pinduoduo.utils.API.send_message import SendMessage


class DummySendMessage(SendMessage):
    def __init__(self, post_result):
        self._post_result = post_result
        self.logger = type(
            "Logger",
            (),
            {
                "error": lambda *args, **kwargs: None,
                "debug": lambda *args, **kwargs: None,
                "info": lambda *args, **kwargs: None,
            },
        )()

    def generate_request_id(self):
        return 1

    def post(self, *args, **kwargs):
        return self._post_result

    def _build_mms_browser_headers(self, **kwargs):
        return {}


class CaptureLogger:
    def __init__(self):
        self.messages = []

    def error(self, message):
        self.messages.append(str(message))

    def debug(self, message):
        self.messages.append(str(message))

    def info(self, message):
        self.messages.append(str(message))


def test_send_text_returns_structured_error_for_platform_business_failure():
    sender = DummySendMessage(
        {
            "success": True,
            "result": {
                "error_code": 10002,
                "error": "会话不可回复",
            },
        }
    )

    result = sender.send_text("buyer-1", "hello")

    assert result == {
        "success": False,
        "error_msg": "会话不可回复",
        "result": {
            "error_code": 10002,
            "error": "会话不可回复",
        },
    }


def test_send_text_masks_sensitive_values_in_business_failure_error_log():
    sender = DummySendMessage(
        {
            "success": True,
            "result": {
                "error_code": 10002,
                "error": "会话不可回复 token=secret-token",
            },
        }
    )
    logger = CaptureLogger()
    sender.logger = logger

    result = sender.send_text("buyer-1", "hello")

    assert result["success"] is False
    assert result["error_msg"] == "会话不可回复 token=secret-token"
    log_text = "\n".join(logger.messages)
    assert "secret-token" not in log_text
    assert "token=***" not in log_text
    assert "error_chars=" in log_text


def test_send_text_returns_structured_error_for_request_failure():
    sender = DummySendMessage(None)

    result = sender.send_text("buyer-1", "hello")

    assert result == {"success": False, "error_msg": "请求失败", "result": None}


def test_send_text_returns_structured_error_for_non_dict_response():
    sender = DummySendMessage("temporary platform error")

    result = sender.send_text("buyer-1", "hello")

    assert result == {
        "success": False,
        "error_msg": "请求失败",
        "result": "temporary platform error",
    }


def test_send_text_treats_string_true_success_as_success():
    sender = DummySendMessage({"success": "true", "result": {"msg_id": "msg-1"}})

    result = sender.send_text("buyer-1", "hello")

    assert result == {"success": "true", "result": {"msg_id": "msg-1"}}


def test_send_text_treats_numeric_false_success_as_failure():
    sender = DummySendMessage({"success": 0, "error_msg": "platform rejected"})

    result = sender.send_text("buyer-1", "hello")

    assert result == {
        "success": False,
        "error_msg": "platform rejected",
        "result": {"success": 0, "error_msg": "platform rejected"},
    }


def test_send_card_returns_structured_error_for_request_failure():
    sender = DummySendMessage(None)

    result = sender.send_mallGoodsCard("buyer-1", 12345)

    assert result == {"success": False, "error_msg": "请求失败", "result": None}


def test_send_card_treats_string_false_success_as_failure():
    sender = DummySendMessage({"success": "false", "error_msg": "anti-content invalid"})

    result = sender.send_mallGoodsCard("buyer-1", 12345)

    assert result["success"] is False
    assert result["error_msg"] == "anti-content invalid"


def test_send_image_returns_structured_error_for_non_dict_response():
    sender = DummySendMessage("temporary platform error")

    result = sender.send_image("buyer-1", "https://example.test/a.png")

    assert result == {
        "success": False,
        "error_msg": "请求失败",
        "result": "temporary platform error",
    }


def test_send_image_treats_string_false_success_as_failure():
    sender = DummySendMessage({"success": "false", "error_msg": "image rejected"})

    result = sender.send_image("buyer-1", "https://example.test/a.png")

    assert result == {
        "success": False,
        "error_msg": "image rejected",
        "result": {"success": "false", "error_msg": "image rejected"},
    }


def test_get_assign_cs_list_returns_none_for_malformed_success_response():
    sender = DummySendMessage({"success": True, "result": {}})

    result = sender.getAssignCsList()

    assert result is None


def test_send_text_handles_non_dict_result_payload():
    sender = DummySendMessage({"success": True, "result": "ok"})

    result = sender.send_text("buyer-1", "hello")

    assert result == {"success": True, "result": "ok"}


def test_get_assign_cs_list_handles_non_dict_result_payload():
    sender = DummySendMessage({"success": False, "result": "timeout"})

    result = sender.getAssignCsList()

    assert result is None


def test_get_assign_cs_list_treats_string_false_success_as_failure():
    sender = DummySendMessage(
        {
            "success": "false",
            "result": {"csList": {"mall_cs": [{"username": "不应使用"}]}},
        }
    )

    result = sender.getAssignCsList()

    assert result is None


def test_get_assign_cs_list_handles_non_dict_top_level_response():
    sender = DummySendMessage("temporary platform error")

    result = sender.getAssignCsList()

    assert result is None


def test_get_assign_cs_list_preserves_grouped_cs_list_payload():
    sender = DummySendMessage(
        {
            "success": True,
            "result": {
                "csList": {
                    "mall_cs": [
                        {"cs_uid": "cs_shop-1_seller-1", "username": "当前客服"},
                        {"cs_uid": "cs_shop-1_seller-2", "username": "客服2"},
                    ]
                }
            },
        }
    )

    result = sender.getAssignCsList()

    assert result == {
        "mall_cs": [
            {"cs_uid": "cs_shop-1_seller-1", "username": "当前客服"},
            {"cs_uid": "cs_shop-1_seller-2", "username": "客服2"},
        ]
    }


def test_send_card_handles_non_dict_response_payload():
    sender = DummySendMessage("temporary platform error")

    result = sender.send_mallGoodsCard("buyer-1", 12345)

    assert result == {
        "success": False,
        "error_msg": "请求失败",
        "result": "temporary platform error",
    }


def test_move_conversation_handles_non_dict_response_payload():
    sender = DummySendMessage("temporary platform error")

    result = sender.move_conversation("buyer-1", "cs-1")

    assert result == {
        "success": False,
        "error_msg": "请求失败",
        "result": "temporary platform error",
    }


def test_move_conversation_treats_string_false_success_as_failure():
    sender = DummySendMessage({"success": "false", "error_msg": "platform rejected"})

    result = sender.move_conversation("buyer-1", "cs-1")

    assert result == {
        "success": False,
        "error_msg": "platform rejected",
        "result": {"success": "false", "error_msg": "platform rejected"},
    }


def test_send_text_masks_sensitive_fields_in_failure_logs():
    sender = DummySendMessage(
        {
            "success": False,
            "error_msg": "platform rejected",
            "result": {
                "token": "secret-token",
                "cookies": {"api_uid": "secret-cookie"},
            },
        }
    )
    logger = CaptureLogger()
    sender.logger = logger

    result = sender.send_text("buyer-1", "hello")

    assert result["success"] is False
    log_text = "\n".join(logger.messages)
    assert "secret-token" not in log_text
    assert "secret-cookie" not in log_text
    assert "'token': '***'" not in log_text
    assert "'cookies': '***'" not in log_text
    assert "result_type=dict" in log_text
    assert "result_keys=" in log_text


def test_send_image_masks_sensitive_fields_in_success_logs():
    sender = DummySendMessage(
        {
            "success": True,
            "result": {
                "token": "secret-token",
                "anti-content": "secret-anti-content",
            },
        }
    )
    logger = CaptureLogger()
    sender.logger = logger

    result = sender.send_image("buyer-1", "https://example.test/a.png")

    assert result["success"] is True
    log_text = "\n".join(logger.messages)
    assert "secret-token" not in log_text
    assert "secret-anti-content" not in log_text
    assert "'token': '***'" not in log_text
    assert "'anti-content': '***'" not in log_text
    assert "result_type=dict" in log_text
    assert "result_keys=" in log_text


def test_send_text_does_not_log_raw_failure_result_values():
    sender = DummySendMessage(
        {
            "success": False,
            "error_msg": "客户手机号13800138000，地址测试小区1号楼",
            "result": {
                "echo": "客户手机号13800138000",
            },
        }
    )
    logger = CaptureLogger()
    sender.logger = logger

    result = sender.send_text("buyer-1", "hello")

    assert result["success"] is False
    log_text = "\n".join(logger.messages)
    assert "13800138000" not in log_text
    assert "测试小区" not in log_text
    assert "result_type=dict" in log_text
