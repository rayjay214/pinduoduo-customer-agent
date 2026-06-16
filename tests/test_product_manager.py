from Channel.pinduoduo.utils import base_request as base_request_module
from Channel.pinduoduo.utils.API.product_manager import ProductManager


class DummyProductManager(ProductManager):
    def __init__(self, post_result):
        self._post_result = post_result
        self.logger = type(
            "Logger",
            (),
            {
                "error": lambda *args, **kwargs: None,
                "warning": lambda *args, **kwargs: None,
            },
        )()

    def _build_mms_browser_headers(self, **kwargs):
        return {}

    def post(self, *args, **kwargs):
        return self._post_result


class CaptureLogger:
    def __init__(self):
        self.messages = []

    def error(self, message):
        self.messages.append(str(message))

    def warning(self, message):
        self.messages.append(str(message))


class BrokenMapping:
    def get(self, *_args, **_kwargs):
        raise RuntimeError("token=secret-token cookies=secret-cookie")


def test_parse_product_list_accepts_common_list_shapes():
    manager = ProductManager()
    parsed = manager._parse_product_list(
        {
            "result": {
                "goodsList": [
                    {
                        "goods_id": 123,
                        "goods_name": "测试商品",
                        "thumb_url": "https://example.com/a.jpg",
                        "min_on_sale_group_price": 990,
                        "max_on_sale_group_price": 1290,
                        "sold_quantity": 5,
                    }
                ],
                "total": 1,
            }
        }
    )

    assert parsed["total"] == 1
    assert parsed["products"][0]["goods_id"] == 123
    assert parsed["products"][0]["goods_name"] == "测试商品"
    assert parsed["products"][0]["price"] == "9.90-12.90"


def test_parse_product_detail_accepts_camelcase_fields_and_spec_values():
    manager = ProductManager()
    parsed = manager._parse_product_detail(
        {
            "result": {
                "goodsId": 123,
                "goodsName": "测试商品",
                "skus": [
                    {
                        "specs": [
                            {"parentName": "颜色", "specName": "白色"},
                            {"parent_name": "规格", "spec_name": "标准版"},
                        ]
                    }
                ],
            }
        }
    )

    assert parsed["goods_id"] == 123
    assert parsed["goods_name"] == "测试商品"
    assert "颜色: 白色 | 规格: 标准版" in parsed["specifications"]


def test_parse_product_detail_ignores_non_text_category_values():
    manager = ProductManager()
    parsed = manager._parse_product_detail(
        {
            "result": {
                "goodsId": 123,
                "goodsName": "测试商品",
                "cats": ["家电", 123, None, "风扇"],
            }
        }
    )

    assert parsed["goods_id"] == 123
    assert parsed["goods_name"] == "测试商品"
    assert "商品分类: 家电 > 123 > 风扇" in parsed["specifications"]


def test_parse_product_list_accepts_string_cent_prices():
    manager = ProductManager()
    parsed = manager._parse_product_list(
        {
            "result": {
                "onSaleGoods": [
                    {
                        "goodsId": 123,
                        "goodsName": "测试商品",
                        "minOnSaleGroupPrice": "990",
                        "maxOnSaleGroupPrice": "990",
                    }
                ],
            }
        }
    )

    assert parsed["products"][0]["price"] == "9.90"


def test_parse_product_list_keeps_decimal_yuan_prices():
    manager = ProductManager()
    parsed = manager._parse_product_list(
        {
            "result": {
                "onSaleGoods": [
                    {
                        "goodsId": 123,
                        "goodsName": "测试商品",
                        "minOnSaleGroupPrice": "9.90",
                        "maxOnSaleGroupPrice": "12.90",
                    }
                ],
            }
        }
    )

    assert parsed["products"][0]["price"] == "9.90-12.90"


def test_parse_product_list_keeps_product_when_price_is_non_finite():
    manager = ProductManager()
    parsed = manager._parse_product_list(
        {
            "result": {
                "onSaleGoods": [
                    {
                        "goodsId": 123,
                        "goodsName": "测试商品",
                        "minOnSaleGroupPrice": float("inf"),
                    }
                ],
                "total": 1,
            }
        }
    )

    assert parsed["total"] == 1
    assert parsed["products"][0]["goods_id"] == 123
    assert parsed["products"][0]["price"] is None


def test_parse_product_list_ignores_malformed_optional_tags():
    manager = ProductManager()
    parsed = manager._parse_product_list(
        {
            "result": {
                "onSaleGoods": [
                    {
                        "goodsId": 123,
                        "goodsName": "测试商品",
                        "goodsTag": "unexpected-shape",
                    },
                    {
                        "goodsId": 456,
                        "goodsName": "另一个商品",
                        "goodsTag": {"marketingTags": ["热销", 123]},
                    },
                ],
                "total": 2,
            }
        }
    )

    assert parsed["total"] == 2
    assert [item["goods_id"] for item in parsed["products"]] == [123, 456]
    assert parsed["products"][0]["tag"] == ""
    assert parsed["products"][1]["tag"] == "热销, 123"


def test_product_manager_invalid_cookies_do_not_overwrite_loaded_cookies(monkeypatch):
    class FakeDBManager:
        def get_account(self, *_args):
            return {"username": "demo", "cookies": {"existing": "cookie"}}

    monkeypatch.setattr(base_request_module, "db_manager", FakeDBManager())

    manager = ProductManager(shop_id="shop-1", user_id="user-1", cookies="{bad json")

    assert manager.cookies == {"existing": "cookie"}


def test_product_list_malformed_success_result_is_failure():
    result = DummyProductManager({"success": True, "result": None}).get_product_list()

    assert result["success"] is False
    assert result["products"] == []
    assert result["total"] == 0
    assert "result 格式异常" in result["error_msg"]


def test_product_detail_malformed_success_result_is_failure():
    result = DummyProductManager({"success": True, "result": []}).get_product_detail(123)

    assert result["success"] is False
    assert "result 格式异常" in result["error_msg"]


def test_product_list_treats_string_true_success_as_success():
    result = DummyProductManager(
        {
                "success": "true",
                "result": {
                    "goodsList": [
                        {"goodsId": 123, "goodsName": "Demo Product", "minOnSaleGroupPrice": 990}
                    ],
                    "total": 1,
                },
        }
    ).get_product_list()

    assert result["success"] is True
    assert result["products"][0]["goods_id"] == 123


def test_product_detail_treats_numeric_true_success_as_success():
    result = DummyProductManager(
        {
            "success": 1,
            "result": {
                "goodsId": 123,
                "goodsName": "测试商品",
            },
        }
    ).get_product_detail(123)

    assert result["success"] is True
    assert result["product_info"]["goods_id"] == 123


def test_product_list_non_dict_response_is_failure():
    result = DummyProductManager("temporary platform error").get_product_list()

    assert result["success"] is False
    assert result["products"] == []
    assert result["total"] == 0
    assert result["error_msg"] == "获取商品列表失败"


def test_product_detail_non_dict_response_is_failure():
    result = DummyProductManager("temporary platform error").get_product_detail(123)

    assert result["success"] is False
    assert result["error_msg"] == "获取商品详情失败"


def test_product_manager_masks_sensitive_values_in_platform_error_logs():
    cases = [
        (DummyProductManager({"success": False, "errorMsg": "token=secret-token"}), "get_product_list", ()),
        (DummyProductManager({"success": False, "errorMsg": "cookies=secret-cookie"}), "get_product_detail", (123,)),
    ]

    for manager, method_name, args in cases:
        logger = CaptureLogger()
        manager.logger = logger

        result = getattr(manager, method_name)(*args)

        assert result["success"] is False
        log_text = "\n".join(logger.messages)
        assert "secret-token" not in log_text
        assert "secret-cookie" not in log_text


def test_product_manager_does_not_log_raw_platform_error_values():
    cases = [
        (DummyProductManager({"success": False, "errorMsg": "客户手机号13800138000，地址测试小区1号楼"}), "get_product_list", ()),
        (DummyProductManager({"success": False, "errorMsg": "客户手机号13800138000，地址测试小区1号楼"}), "get_product_detail", (123,)),
    ]

    for manager, method_name, args in cases:
        logger = CaptureLogger()
        manager.logger = logger

        result = getattr(manager, method_name)(*args)

        assert result["success"] is False
        log_text = "\n".join(logger.messages)
        assert "13800138000" not in log_text
        assert "测试小区" not in log_text
        assert "error_chars=" in log_text


def test_product_manager_sanitizes_parse_exception_logs():
    cases = [
        ("_parse_product_list", "解析商品列表失败", "products"),
        ("_parse_product_detail", "解析商品详情失败", "goods_name"),
    ]

    for method_name, expected_message, result_key in cases:
        manager = ProductManager()
        logger = CaptureLogger()
        manager.logger = logger

        result = getattr(manager, method_name)(BrokenMapping())

        if result_key == "products":
            assert result["products"] == []
        else:
            assert result["goods_name"] == "解析失败"
        log_text = "\n".join(logger.messages)
        assert expected_message in log_text
        assert "secret-token" not in log_text
        assert "secret-cookie" not in log_text
        assert "token=***" in log_text or "cookies=***" in log_text
