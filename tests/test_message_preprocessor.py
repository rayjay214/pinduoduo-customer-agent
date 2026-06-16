import json

from Message.handlers import preprocessor as preprocessor_module
from Message.handlers.preprocessor import MessagePreprocessor


def test_order_status_codes_are_converted_to_scene_text():
    processed = MessagePreprocessor().process(
        {
            "order_id": "order-1",
            "shipping_status": 1,
            "status": 3,
        }
    )

    assert "当前订单状态：已发货待收货" in processed
    assert "当前业务场景：售中-物流中" in processed


def test_aftersale_status_drives_preprocessor_scene():
    processed = MessagePreprocessor().process(
        {
            "order_id": "order-1",
            "afterSalesStatusStr": "退款处理中",
        }
    )

    assert "售后状态：退款处理中" in processed
    assert "当前订单状态：售后中（退款处理中）" in processed
    assert "当前业务场景：售后倾向" in processed


def test_refunded_aftersale_status_has_terminal_status_label():
    processed = MessagePreprocessor().process(
        {
            "order_id": "order-1",
            "refund_status": "退款成功",
        }
    )

    assert "售后状态：退款成功" in processed
    assert "当前订单状态：已退款" in processed
    assert "当前业务场景：售后倾向" in processed


def test_refunded_text_status_drives_preprocessor_scene_without_aftersale_field():
    processed = MessagePreprocessor().process(
        {
            "order_id": "order-1",
            "status": "退款成功",
        }
    )

    assert "当前订单状态：已退款" in processed
    assert "当前业务场景：售后倾向" in processed


def test_created_order_message_round_trips_order_sn_and_status_text():
    message = MessagePreprocessor.create_order_message(
        "order-1",
        status="已发货",
        goods_name="测试商品",
    )
    processed = MessagePreprocessor().process(message)

    assert "订单：order-1" in processed
    assert "商品：测试商品" in processed
    assert "当前订单状态：已发货" in processed
    assert "订单状态码：已发货" not in processed


def test_empty_aftersale_status_code_does_not_force_preprocessor_aftersale_scene():
    processed = MessagePreprocessor().process(
        {
            "order_id": "order-1",
            "afterSalesStatus": 0,
            "shipping_status": 1,
            "status": 3,
        }
    )

    assert "售后状态" not in processed
    assert "当前订单状态：已发货待收货" in processed
    assert "当前业务场景：售中-物流中" in processed


def test_non_finite_status_code_does_not_break_preprocessing():
    processed = MessagePreprocessor().process(
        {
            "order_id": "order-1",
            "shipping_status": float("inf"),
            "status": 3,
        }
    )

    assert "消息处理失败" not in processed
    assert "订单：order-1" in processed


def test_missing_content_preprocesses_to_empty_string():
    assert MessagePreprocessor().process(None) == ""


def test_process_masks_sensitive_exception_logs(monkeypatch):
    messages = []

    class FakeLogger:
        def error(self, message):
            messages.append(str(message))

    class BrokenContent:
        def __bool__(self):
            raise RuntimeError("token=secret-token")

    monkeypatch.setattr(preprocessor_module, "logger", FakeLogger())

    processed = MessagePreprocessor().process(BrokenContent())

    assert processed == "消息处理失败"
    joined = "\n".join(messages)
    assert "Message preprocessing failed" in joined
    assert "secret-token" not in joined
    assert "token=***" in joined


def test_standard_message_list_preserves_text_and_goods_card_context():
    message = json.dumps(
        [
            {"type": "text", "text": "这个多少钱"},
            {
                "type": "goods_card",
                "goods_name": "测试商品",
                "goods_id": "123456789",
                "price": "1999",
                "spec": "白色",
            },
        ],
        ensure_ascii=False,
    )

    processed = MessagePreprocessor().process(message)

    assert "内容：这个多少钱" in processed
    assert "商品：测试商品" in processed
    assert "商品ID：123456789" in processed
    assert "价格：1999" in processed
    assert "规格：白色" in processed


def test_standard_goods_card_prefers_normalized_non_empty_fields():
    message = json.dumps(
        [
            {
                "type": "goods_card",
                "goods_name": "",
                "name": "真实商品名",
                "goods_price": "",
                "price": "2590",
                "goods_id": "123456789",
            }
        ],
        ensure_ascii=False,
    )

    processed = MessagePreprocessor().process(message)

    assert "商品：真实商品名" in processed
    assert "价格：2590" in processed
    assert "商品ID：123456789" in processed


def test_extract_key_info_returns_empty_for_non_dict_input():
    assert MessagePreprocessor()._extract_key_info(["bad"]) == ""
