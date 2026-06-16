import asyncio
from types import SimpleNamespace

from database import product_sync
from database.product_sync import ProductSyncService


class FakeKnowledgeService:
    def __init__(self):
        self.saved = []

    def get_product_by_goods_id(self, _shop_id, _goods_id):
        return None

    def add_or_update_product(self, **kwargs):
        self.saved.append(kwargs)


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(str(message))

    def warning(self, message):
        self.messages.append(str(message))

    def error(self, message):
        self.messages.append(str(message))

    def debug(self, message):
        self.messages.append(str(message))


class FakeProductManager:
    detail_calls = 0

    def __init__(self, *_args, **_kwargs):
        pass

    def get_product_list(self, page=1, size=10):
        if page == 1 and size == 20:
            return {"success": True, "total": 1, "products": []}
        if page == 1:
            return {
                "success": True,
                "total": 1,
                "products": [
                    {
                        "goods_id": 123,
                        "goods_name": "测试商品",
                        "price": "9.90",
                        "thumb_url": "https://example.com/a.jpg",
                    }
                ],
            }
        return {"success": True, "total": 1, "products": []}

    def get_product_detail(self, _goods_id):
        FakeProductManager.detail_calls += 1
        raise AssertionError("sync_shop should not fetch product detail")


def test_sync_shop_only_saves_basic_product_info(monkeypatch):
    async def run():
        monkeypatch.setattr(product_sync, "ProductManager", FakeProductManager)
        knowledge = FakeKnowledgeService()
        service = ProductSyncService(knowledge, request_delay=0)

        async def fail_extract(*_args, **_kwargs):
            raise AssertionError("sync_shop should not call LLM extraction")

        service._extract_product_knowledge = fail_extract

        progress = await service.sync_shop(
            shop_id="external-shop",
            shop_db_id=1,
            user_id="user-1",
            is_full_sync=False,
        )

        assert progress.total == 1
        assert progress.success == 1
        assert progress.failed == 0
        assert FakeProductManager.detail_calls == 0
        assert knowledge.saved[0]["goods_id"] == 123
        assert knowledge.saved[0]["extracted_content"] is None

    asyncio.run(run())


def test_extract_product_knowledge_falls_back_when_llm_choices_missing(monkeypatch):
    async def run():
        class FakeCompletions:
            async def create(self, **_kwargs):
                return SimpleNamespace(choices=[])

        class FakeChat:
            completions = FakeCompletions()

        class FakeAsyncOpenAI:
            def __init__(self, **_kwargs):
                self.chat = FakeChat()

        def fake_get_config(key, default=None):
            values = {
                "llm.api_key": "key",
                "llm.model_name": "model",
                "llm.api_base": "http://llm",
            }
            return values.get(key, default)

        monkeypatch.setattr(product_sync, "AsyncOpenAI", FakeAsyncOpenAI)
        monkeypatch.setattr(product_sync, "get_config", fake_get_config)

        service = ProductSyncService(FakeKnowledgeService(), request_delay=0)
        result = await service._extract_product_knowledge(
            {"goods_name": "测试商品", "price": "9.90", "sold_quantity": 3},
            {"specifications": ["红色"]},
        )

        assert "# 测试商品" in result
        assert "**规格信息**" in result

    asyncio.run(run())


def test_extract_product_knowledge_skips_malformed_faq_items(monkeypatch):
    async def run():
        class FakeCompletions:
            async def create(self, **_kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content='{"description":"轻便好用","faq":["bad",{"question":"怎么用","answer":"按说明使用"}]}'
                            )
                        )
                    ]
                )

        class FakeChat:
            completions = FakeCompletions()

        class FakeAsyncOpenAI:
            def __init__(self, **_kwargs):
                self.chat = FakeChat()

        def fake_get_config(key, default=None):
            values = {
                "llm.api_key": "key",
                "llm.model_name": "model",
                "llm.api_base": "http://llm",
            }
            return values.get(key, default)

        monkeypatch.setattr(product_sync, "AsyncOpenAI", FakeAsyncOpenAI)
        monkeypatch.setattr(product_sync, "get_config", fake_get_config)

        service = ProductSyncService(FakeKnowledgeService(), request_delay=0)
        result = await service._extract_product_knowledge(
            {"goods_name": "测试商品", "price": "9.90", "sold_quantity": 3},
            {"specifications": ["红色"]},
        )

        assert "## 产品描述" in result
        assert "轻便好用" in result
        assert "**Q:** 怎么用" in result
        assert "**A:** 按说明使用" in result

    asyncio.run(run())


def test_extract_product_knowledge_falls_back_when_llm_returns_non_json(monkeypatch):
    async def run():
        class FakeCompletions:
            async def create(self, **_kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="抱歉，我无法输出 JSON。token=secret-token"
                            )
                        )
                    ]
                )

        class FakeChat:
            completions = FakeCompletions()

        class FakeAsyncOpenAI:
            def __init__(self, **_kwargs):
                self.chat = FakeChat()

        def fake_get_config(key, default=None):
            values = {
                "llm.api_key": "key",
                "llm.model_name": "model",
                "llm.api_base": "http://llm",
            }
            return values.get(key, default)

        monkeypatch.setattr(product_sync, "AsyncOpenAI", FakeAsyncOpenAI)
        monkeypatch.setattr(product_sync, "get_config", fake_get_config)

        service = ProductSyncService(FakeKnowledgeService(), request_delay=0)
        result = await service._extract_product_knowledge(
            {"goods_name": "测试商品", "price": "9.90", "sold_quantity": 3},
            {"specifications": ["红色"]},
        )

        assert result.startswith("# 测试商品")
        assert "抱歉" not in result
        assert "secret-token" not in result

    asyncio.run(run())


def test_extract_product_knowledge_non_json_log_does_not_include_raw_llm_content(monkeypatch):
    async def run():
        class FakeCompletions:
            async def create(self, **_kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="抱歉，我无法输出 JSON。客户隐私 token=secret-token"
                            )
                        )
                    ]
                )

        class FakeChat:
            completions = FakeCompletions()

        class FakeAsyncOpenAI:
            def __init__(self, **_kwargs):
                self.chat = FakeChat()

        def fake_get_config(key, default=None):
            values = {
                "llm.api_key": "key",
                "llm.model_name": "model",
                "llm.api_base": "http://llm",
            }
            return values.get(key, default)

        logger = FakeLogger()
        monkeypatch.setattr(product_sync, "AsyncOpenAI", FakeAsyncOpenAI)
        monkeypatch.setattr(product_sync, "get_config", fake_get_config)
        monkeypatch.setattr(product_sync, "logger", logger)

        service = ProductSyncService(FakeKnowledgeService(), request_delay=0)
        result = await service._extract_product_knowledge(
            {"goods_name": "测试商品", "price": "9.90", "sold_quantity": 3},
            {"specifications": ["红色"]},
        )

        joined = "\n".join(logger.messages)
        assert result.startswith("# 测试商品")
        assert "secret-token" not in joined
        assert "token=***" not in joined
        assert "抱歉" not in joined
        assert "LLM输出长度" in joined

    asyncio.run(run())


def test_sync_shop_handles_malformed_first_page_response(monkeypatch):
    async def run():
        class MalformedProductManager:
            def __init__(self, *_args, **_kwargs):
                pass

            def get_product_list(self, page=1, size=10):
                return {"error_msg": "bad response"}

        monkeypatch.setattr(product_sync, "ProductManager", MalformedProductManager)
        service = ProductSyncService(FakeKnowledgeService(), request_delay=0)

        progress = await service.sync_shop(
            shop_id="external-shop",
            shop_db_id=1,
            user_id="user-1",
            is_full_sync=False,
        )

        assert progress.failed == 1
        assert progress.success == 0

    asyncio.run(run())


def test_sync_shop_masks_sensitive_error_message_in_logs(monkeypatch):
    async def run():
        class FailedProductManager:
            def __init__(self, *_args, **_kwargs):
                pass

            def get_product_list(self, page=1, size=10):
                return {
                    "success": False,
                    "error_msg": "token=secret-token",
                    "products": [],
                    "total": 0,
                }

        monkeypatch.setattr(product_sync, "ProductManager", FailedProductManager)
        logger = FakeLogger()
        monkeypatch.setattr(product_sync, "logger", logger)

        service = ProductSyncService(FakeKnowledgeService(), request_delay=0)
        progress = await service.sync_shop(
            shop_id="external-shop",
            shop_db_id=1,
            user_id="user-1",
            is_full_sync=False,
        )

        joined = "\n".join(logger.messages)
        assert progress.failed == 1
        assert "secret-token" not in joined
        assert "token=***" not in joined
        assert "error_chars=" in joined

    asyncio.run(run())


def test_sync_shop_does_not_log_raw_product_list_error_message(monkeypatch):
    async def run():
        class FailedProductManager:
            def __init__(self, *_args, **_kwargs):
                pass

            def get_product_list(self, page=1, size=10):
                return {
                    "success": False,
                    "error_msg": "客户手机号13800138000，地址测试小区1号楼",
                    "products": [],
                    "total": 0,
                }

        monkeypatch.setattr(product_sync, "ProductManager", FailedProductManager)
        logger = FakeLogger()
        monkeypatch.setattr(product_sync, "logger", logger)

        service = ProductSyncService(FakeKnowledgeService(), request_delay=0)
        progress = await service.sync_shop(
            shop_id="external-shop",
            shop_db_id=1,
            user_id="user-1",
            is_full_sync=False,
        )

        joined = "\n".join(logger.messages)
        assert progress.failed == 1
        assert "13800138000" not in joined
        assert "测试小区" not in joined
        assert "error_chars=" in joined

    asyncio.run(run())


def test_sync_shop_treats_string_false_success_as_failure(monkeypatch):
    async def run():
        class FailedProductManager:
            def __init__(self, *_args, **_kwargs):
                pass

            def get_product_list(self, page=1, size=10):
                return {
                    "success": "false",
                    "error_msg": "platform rejected request",
                    "products": [{"goods_id": 123, "goods_name": "不应同步"}],
                    "total": 1,
                }

        monkeypatch.setattr(product_sync, "ProductManager", FailedProductManager)
        knowledge = FakeKnowledgeService()
        service = ProductSyncService(knowledge, request_delay=0)

        progress = await service.sync_shop(
            shop_id="external-shop",
            shop_db_id=1,
            user_id="user-1",
            is_full_sync=False,
        )

        assert progress.failed == 1
        assert progress.success == 0
        assert knowledge.saved == []

    asyncio.run(run())


def test_sync_shop_ignores_malformed_product_page(monkeypatch):
    async def run():
        class MalformedProductManager:
            def __init__(self, *_args, **_kwargs):
                pass

            def get_product_list(self, page=1, size=10):
                if page == 1 and size == 20:
                    return {"success": True, "total": "bad-total", "products": []}
                return {"success": True, "total": 1, "products": "not-a-list"}

        monkeypatch.setattr(product_sync, "ProductManager", MalformedProductManager)
        knowledge = FakeKnowledgeService()
        service = ProductSyncService(knowledge, request_delay=0)

        progress = await service.sync_shop(
            shop_id="external-shop",
            shop_db_id=1,
            user_id="user-1",
            is_full_sync=False,
        )

        assert progress.total == 0
        assert progress.success == 0
        assert progress.failed == 0
        assert knowledge.saved == []

    asyncio.run(run())


def test_format_basic_info_treats_string_specifications_as_single_item():
    service = ProductSyncService(FakeKnowledgeService(), request_delay=0)

    result = service._format_basic_info(
        {"goods_name": "测试商品"},
        {"specifications": "红色"},
    )

    assert "- 红色" in result
    assert "- 红\n- 色" not in result


def test_extract_product_knowledge_tolerates_non_dict_inputs(monkeypatch):
    async def run():
        monkeypatch.setattr(
            product_sync,
            "get_config",
            lambda key, default=None: "" if key == "llm.api_key" else default,
        )
        service = ProductSyncService(FakeKnowledgeService(), request_delay=0)

        result = await service._extract_product_knowledge("bad-list-product", "bad-detail-product")

        assert result == "# None"

    asyncio.run(run())


def test_extract_product_knowledge_masks_exception_logs(monkeypatch):
    async def run():
        class FakeCompletions:
            async def create(self, **_kwargs):
                raise RuntimeError("cookies=secret-cookie")

        class FakeChat:
            completions = FakeCompletions()

        class FakeAsyncOpenAI:
            def __init__(self, **_kwargs):
                self.chat = FakeChat()

        def fake_get_config(key, default=None):
            values = {
                "llm.api_key": "key",
                "llm.model_name": "model",
                "llm.api_base": "http://llm",
            }
            return values.get(key, default)

        logger = FakeLogger()
        monkeypatch.setattr(product_sync, "AsyncOpenAI", FakeAsyncOpenAI)
        monkeypatch.setattr(product_sync, "get_config", fake_get_config)
        monkeypatch.setattr(product_sync, "logger", logger)

        service = ProductSyncService(FakeKnowledgeService(), request_delay=0)
        result = await service._extract_product_knowledge(
            {"goods_name": "测试商品", "price": "9.90", "sold_quantity": 3},
            {"specifications": ["红色"]},
        )

        joined = "\n".join(logger.messages)
        assert result.startswith("# 测试商品")
        assert "secret-cookie" not in joined
        assert "cookies=***" in joined

    asyncio.run(run())


def test_extract_product_knowledge_does_not_log_raw_llm_content(monkeypatch):
    async def run():
        class FakeCompletions:
            async def create(self, **_kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=(
                                    '{"brand":"token=secret-token",'
                                    '"origin":"api_key=secret-api",'
                                    '"description":"安全描述"}'
                                )
                            )
                        )
                    ]
                )

        class FakeChat:
            completions = FakeCompletions()

        class FakeAsyncOpenAI:
            def __init__(self, **_kwargs):
                self.chat = FakeChat()

        def fake_get_config(key, default=None):
            values = {
                "llm.api_key": "key",
                "llm.model_name": "model",
                "llm.api_base": "http://llm",
            }
            return values.get(key, default)

        logger = FakeLogger()
        monkeypatch.setattr(product_sync, "AsyncOpenAI", FakeAsyncOpenAI)
        monkeypatch.setattr(product_sync, "get_config", fake_get_config)
        monkeypatch.setattr(product_sync, "logger", logger)

        service = ProductSyncService(FakeKnowledgeService(), request_delay=0)
        result = await service._extract_product_knowledge(
            {"goods_name": "测试商品", "price": "9.90", "sold_quantity": 3},
            {"specifications": ["红色"]},
        )

        assert "token=secret-token" in result
        joined = "\n".join(logger.messages)
        assert "secret-token" not in joined
        assert "secret-api" not in joined
        assert "token=***" not in joined
        assert "api_key=***" not in joined
        assert "LLM输出长度" in joined

    asyncio.run(run())
