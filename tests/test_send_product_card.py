from types import SimpleNamespace

from Agent.CustomerAgent.tools import send_product_card as module


class CaptureLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(str(message))

    def warning(self, message):
        self.messages.append(str(message))

    def error(self, message):
        self.messages.append(str(message))


def test_send_product_card_sends_explicit_candidate_index(monkeypatch):
    calls = []

    class FakeProductManager:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def get_product_list(self, page=1, size=10):
            return {
                "success": True,
                "products": [
                    {"goods_id": 111, "goods_name": "A"},
                    {"goods_id": 222, "goods_name": "B"},
                ],
                "total": 2,
            }

    class FakeSendMessage:
        def __init__(self, shop_id, user_id):
            calls.append(("init", shop_id, user_id))

        def send_mallGoodsCard(self, recipient_uid, goods_id, biz_type=2):
            calls.append(("send", recipient_uid, goods_id, biz_type))
            return {"success": True}

    monkeypatch.setattr(module, "ProductManager", FakeProductManager)
    monkeypatch.setattr(module, "SendMessage", FakeSendMessage)

    result = module.send_product_card(
        module.SendProductCardParams(
            shop_id="shop-1",
            user_id="user-1",
            recipient_uid="buyer-1",
            candidate_index=2,
        )
    )

    assert result == "商品卡片发送成功"
    assert calls == [
        ("init", "shop-1", "user-1"),
        ("send", "buyer-1", 222, 2),
    ]


def test_send_product_card_rejects_goods_id_that_matches_candidate_index(monkeypatch):
    calls = []

    class FakeProductManager:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def get_product_list(self, page=1, size=10):
            return {
                "success": True,
                "products": [
                    {"goods_id": 111, "goods_name": "A"},
                    {"goods_id": 222, "goods_name": "B"},
                ],
                "total": 2,
            }

    class FakeSendMessage:
        def __init__(self, shop_id, user_id):
            calls.append(("init", shop_id, user_id))

        def send_mallGoodsCard(self, recipient_uid, goods_id, biz_type=2):
            calls.append(("send", recipient_uid, goods_id, biz_type))
            return {"success": True}

    monkeypatch.setattr(module, "ProductManager", FakeProductManager)
    monkeypatch.setattr(module, "SendMessage", FakeSendMessage)

    result = module.send_product_card(
        module.SendProductCardParams(
            shop_id="shop-1",
            user_id="user-1",
            recipient_uid="buyer-1",
            goods_id=2,
        )
    )

    assert "像候选列表序号" in result
    assert "candidate_index=2" in result
    assert "商品ID：222" in result
    assert calls == []


def test_send_product_card_rejects_out_of_range_candidate_index(monkeypatch):
    class FakeProductManager:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def get_product_list(self, page=1, size=10):
            return {
                "success": True,
                "products": [{"goods_id": 111, "goods_name": "A"}],
                "total": 1,
            }

    monkeypatch.setattr(module, "ProductManager", FakeProductManager)

    result = module.send_product_card(
        module.SendProductCardParams(
            shop_id="shop-1",
            user_id="user-1",
            recipient_uid="buyer-1",
            candidate_index=2,
        )
    )

    assert "候选序号 2 超出范围" in result


def test_send_product_card_rejects_malformed_candidate_index():
    result = module.send_product_card(
        SimpleNamespace(
            shop_id="shop-1",
            user_id="user-1",
            recipient_uid="buyer-1",
            goods_id=None,
            candidate_index="bad",
        )
    )

    assert "候选序号格式异常" in result


def test_send_product_card_rejects_candidate_with_non_finite_goods_id(monkeypatch):
    class FakeProductManager:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def get_product_list(self, page=1, size=10):
            return {
                "success": True,
                "products": [{"goods_id": float("inf"), "goods_name": "A"}],
                "total": 1,
            }

    monkeypatch.setattr(module, "ProductManager", FakeProductManager)

    result = module.send_product_card(
        module.SendProductCardParams(
            shop_id="shop-1",
            user_id="user-1",
            recipient_uid="buyer-1",
            candidate_index=1,
        )
    )

    assert "商品ID格式异常" in result


def test_load_products_uses_product_count_when_total_is_invalid(monkeypatch):
    class FakeProductManager:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def get_product_list(self, page=1, size=10):
            return {
                "success": True,
                "products": [{"goods_id": 123, "goods_name": "测试商品"}],
                "total": "bad-total",
            }

    monkeypatch.setattr(module, "ProductManager", FakeProductManager)

    products, total, error_msg = module._load_products("shop-1", "user-1")

    assert len(products) == 1
    assert total == 1
    assert error_msg == ""


def test_load_products_handles_non_dict_result(monkeypatch):
    class FakeProductManager:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def get_product_list(self, page=1, size=10):
            return "temporary platform error"

    monkeypatch.setattr(module, "ProductManager", FakeProductManager)

    products, total, error_msg = module._load_products("shop-1", "user-1")

    assert products == []
    assert total == 0
    assert "响应格式异常" in error_msg


def test_load_products_treats_string_false_success_as_failure(monkeypatch):
    class FakeProductManager:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def get_product_list(self, page=1, size=10):
            return {
                "success": "false",
                "error_msg": "platform rejected",
                "products": [{"goods_id": 123, "goods_name": "不应展示"}],
                "total": 1,
            }

    monkeypatch.setattr(module, "ProductManager", FakeProductManager)

    products, total, error_msg = module._load_products("shop-1", "user-1")

    assert products == []
    assert total == 0
    assert error_msg == "platform rejected"


def test_load_products_masks_sensitive_error_for_tool_output(monkeypatch):
    class FakeProductManager:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def get_product_list(self, page=1, size=10):
            return {
                "success": False,
                "error_msg": "platform rejected cookies=secret-cookie",
            }

    monkeypatch.setattr(module, "ProductManager", FakeProductManager)

    products, total, error_msg = module._load_products("shop-1", "user-1")

    assert products == []
    assert total == 0
    assert "secret-cookie" not in error_msg
    assert "cookies=***" in error_msg


def test_load_products_filters_non_dict_product_items(monkeypatch):
    class FakeProductManager:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def get_product_list(self, page=1, size=10):
            return {
                "success": True,
                "products": ["bad", {"goods_id": 123, "goods_name": "测试商品"}],
                "total": 2,
            }

    monkeypatch.setattr(module, "ProductManager", FakeProductManager)

    products, total, error_msg = module._load_products("shop-1", "user-1")

    assert products == [{"goods_id": 123, "goods_name": "测试商品"}]
    assert total == 2
    assert error_msg == ""


def test_single_product_with_invalid_goods_id_is_listed_instead_of_crashing(monkeypatch):
    class FakeProductManager:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def get_product_list(self, page=1, size=10):
            return {
                "success": True,
                "products": [{"goods_id": "abc", "goods_name": "测试商品"}],
                "total": 1,
            }

    monkeypatch.setattr(module, "ProductManager", FakeProductManager)

    result = module.send_product_card(
        module.SendProductCardParams(
            shop_id="shop-1",
            user_id="user-1",
            recipient_uid="buyer-1",
        )
    )

    assert "可推荐商品列表" in result
    assert "商品ID：abc" in result


def test_send_card_handles_non_dict_send_result(monkeypatch):
    class FakeSendMessage:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def send_mallGoodsCard(self, recipient_uid, goods_id, biz_type=2):
            return "temporary platform error"

    monkeypatch.setattr(module, "SendMessage", FakeSendMessage)

    result = module._send_card("shop-1", "user-1", "buyer-1", 12345)

    assert result == "商品卡片发送失败: 发送失败"


def test_send_card_treats_string_false_success_as_failure(monkeypatch):
    class FakeSendMessage:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def send_mallGoodsCard(self, recipient_uid, goods_id, biz_type=2):
            return {"success": "false", "error_msg": "platform rejected"}

    monkeypatch.setattr(module, "SendMessage", FakeSendMessage)

    result = module._send_card("shop-1", "user-1", "buyer-1", 12345)

    assert result == "商品卡片发送失败: platform rejected"


def test_send_card_masks_sensitive_error_in_log_and_tool_output(monkeypatch):
    class FakeSendMessage:
        def __init__(self, shop_id, user_id):
            self.shop_id = shop_id
            self.user_id = user_id

        def send_mallGoodsCard(self, recipient_uid, goods_id, biz_type=2):
            return {"success": False, "error_msg": "platform rejected token=secret-token"}

    logger = CaptureLogger()
    monkeypatch.setattr(module, "SendMessage", FakeSendMessage)
    monkeypatch.setattr(module, "logger", logger)

    result = module._send_card("shop-1", "user-1", "buyer-1", 12345)

    assert "secret-token" not in result
    assert "token=***" in result
    assert "secret-token" not in "\n".join(logger.messages)
