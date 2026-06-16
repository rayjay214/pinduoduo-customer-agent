from Channel.pinduoduo.utils import base_request as base_request_module
from Channel.pinduoduo.utils.API.order_manager import (
    OrderManager,
    build_order_context_text,
    _normalize_shipping_status,
)


def test_signed_trace_keywords_keep_default(monkeypatch):
    monkeypatch.setattr(
        "config.get_config",
        lambda key, default=None: default,
    )

    assert _normalize_shipping_status({"traceInfoList": [{"content": "包裹已签收，签收人是本人"}]}) == "signed"


def test_signed_trace_keywords_can_be_configured(monkeypatch):
    monkeypatch.setattr(
        "config.get_config",
        lambda key, default=None: ["投递完成"]
        if key == "pinduoduo.order.signed_trace_keywords"
        else default,
    )

    assert _normalize_shipping_status({"traceInfoList": [{"content": "投递完成，请留意取件"}]}) == "signed"


def test_signed_trace_keywords_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        "config.get_config",
        lambda key, default=None: []
        if key == "pinduoduo.order.signed_trace_keywords"
        else default,
    )

    assert _normalize_shipping_status({"traceInfoList": [{"content": "包裹已签收"}]}) != "signed"


def test_aftersale_status_drives_order_scene_without_order_status():
    context = build_order_context_text(
        [
            {
                "order_id": "order-1",
                "goodsName": "测试商品",
                "afterSalesStatusStr": "退款处理中",
            }
        ]
    )

    assert context["scene_hint"] == "aftersale"
    assert context["business_status"] == "售后中（退款处理中）"
    assert "当前业务场景：售后倾向" in context["text"]


def test_empty_aftersale_status_code_does_not_force_aftersale_scene():
    context = build_order_context_text(
        [
            {
                "order_id": "order-1",
                "goodsName": "测试商品",
                "afterSalesStatus": 0,
                "shipping_status": 1,
                "status": 3,
            }
        ]
    )

    assert context["scene_hint"] == "insale"
    assert context["business_status"] == "已发货待收货"
    assert "售后状态" not in context["text"]


def test_non_finite_status_code_does_not_break_order_context():
    context = build_order_context_text(
        [
            {
                "order_id": "order-1",
                "shipping_status": float("inf"),
                "status": 3,
            }
        ]
    )

    assert context["has_order"] is True
    assert context["order_id"] == "order-1"
    assert context["business_status"] == "未知"


def test_refunded_order_status_uses_aftersale_scene():
    context = build_order_context_text(
        [
            {
                "order_id": "order-1",
                "goodsName": "测试商品",
                "orderStatusStr": "退款成功",
            }
        ]
    )

    assert context["scene_hint"] == "aftersale"
    assert context["business_status"] == "已退款"
    assert "当前业务场景：售后倾向" in context["text"]


def test_order_manager_invalid_cookies_do_not_overwrite_loaded_cookies(monkeypatch):
    class FakeDBManager:
        def get_account(self, *_args):
            return {"username": "demo", "cookies": {"existing": "cookie"}}

    monkeypatch.setattr(base_request_module, "db_manager", FakeDBManager())

    manager = OrderManager(shop_id="shop-1", user_id="user-1", cookies="{bad json")

    assert manager.cookies == {"existing": "cookie"}


def test_order_manager_invalid_page_size_falls_back_to_default(monkeypatch):
    captured = {}

    class FakeDBManager:
        def get_account(self, *_args):
            return {"username": "demo", "cookies": {"existing": "cookie"}}

    monkeypatch.setattr(base_request_module, "db_manager", FakeDBManager())

    manager = OrderManager(shop_id="shop-1", user_id="user-1")

    def fake_post(_url, json_data=None, **_kwargs):
        captured.update(json_data or {})
        return {"result": {"orders": []}}

    monkeypatch.setattr(manager, "post", fake_post)

    assert manager.get_user_orders("buyer-1", page_size="bad") == []
    assert captured["pageSize"] == 10


def test_order_manager_non_dict_result_payload_returns_empty(monkeypatch):
    class FakeDBManager:
        def get_account(self, *_args):
            return {"username": "demo", "cookies": {"existing": "cookie"}}

    monkeypatch.setattr(base_request_module, "db_manager", FakeDBManager())

    manager = OrderManager(shop_id="shop-1", user_id="user-1")
    monkeypatch.setattr(manager, "post", lambda *_args, **_kwargs: {"result": "temporary error"})

    assert manager.get_user_orders("buyer-1") == []
