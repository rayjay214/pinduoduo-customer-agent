from Agent.CustomerAgent.tools import search_knowledge as module
from Agent.CustomerAgent.custom.tool_decorator import execute_tool, get_tools_for_llm
from database.knowledge_service import KnowledgeService
from types import SimpleNamespace


def test_search_knowledge_accepts_string_shop_id(monkeypatch):
    class FakeKnowledgeService:
        def search_scene_knowledge(self, scene, shop_id, goods_id, query, limit):
            assert shop_id == "shop-1"
            assert goods_id == 123456789
            return [
                {
                    "answer": "标准答案",
                    "section_title": "参数",
                    "sub_intent": "续航",
                }
            ]

        def format_scene_results(self, results):
            return results[0]["answer"]

    monkeypatch.setattr(module, "_knowledge_service", lambda: FakeKnowledgeService())

    result = execute_tool(
        "search_knowledge",
        '{"query":"续航多久","shop_id":"shop-1","goods_id":123456789,"scene":"售前"}',
        {},
    )

    assert "客户当前问题：续航多久" in result
    assert "当前场景：presale" in result
    assert "先判断客户当前问题主题" in result
    assert "标准答案" in result


def test_search_knowledge_result_includes_candidate_selection_context(monkeypatch):
    class FakeKnowledgeService:
        def search_scene_knowledge(self, scene, shop_id, goods_id, query, limit):
            return [
                {
                    "answer": "页面价格以当前优惠为准",
                    "section_title": "价格优惠",
                    "sub_intent": "价格",
                    "score": 120,
                    "match_type": "keyword",
                },
                {
                    "answer": "风力以页面参数为准",
                    "section_title": "风力参数",
                    "sub_intent": "风力",
                    "score": 90,
                    "match_type": "keyword",
                },
            ]

        def format_scene_results(self, results):
            return "\n".join(
                f"{item['section_title']}：{item['answer']}" for item in results
            )

    monkeypatch.setattr(module, "_knowledge_service", lambda: FakeKnowledgeService())

    result = execute_tool(
        "search_knowledge",
        '{"query":"风力大吗","shop_id":"shop-1","goods_id":123456789,"scene":"售前"}',
        {},
    )

    assert "客户当前问题：风力大吗" in result
    assert "当前场景：presale" in result
    assert "当前商品ID：123456789" in result
    assert "不要只因为某条候选排序靠前" in result
    assert "价格优惠" in result
    assert "风力参数" in result


def test_search_knowledge_without_goods_id_transfers_to_human(monkeypatch):
    transfer_calls = []

    class FakeKnowledgeService:
        def search_scene_knowledge(self, *_args, **_kwargs):
            raise AssertionError("无商品ID时不应查询商品场景知识")

        def search_knowledge(self, *_args, **_kwargs):
            raise AssertionError("不应查询老客服知识库")

    def fake_transfer(params):
        transfer_calls.append((params.shop_id, params.user_id, params.recipient_uid))
        return "会话转接成功"

    monkeypatch.setattr(module, "_knowledge_service", lambda: FakeKnowledgeService())
    monkeypatch.setattr(module, "_transfer_to_human", fake_transfer)

    result = execute_tool(
        "search_knowledge",
        '{"query":"发什么快递","shop_id":"shop-1","scene":"售前"}',
        {"user_id": "user-1", "recipient_uid": "buyer-1"},
    )

    assert transfer_calls == [("shop-1", "user-1", "buyer-1")]
    assert result == "会话转接成功"


def test_search_knowledge_transfers_when_scene_results_empty(monkeypatch):
    calls = []
    transfer_calls = []

    class FakeKnowledgeService:
        def search_scene_knowledge(self, scene, shop_id, goods_id, query, limit):
            calls.append(("scene", scene, shop_id, goods_id, query, limit))
            return []

        def search_knowledge(self, *_args, **_kwargs):
            raise AssertionError("不应查询老商品知识库")

    def fake_transfer(params):
        transfer_calls.append((params.shop_id, params.user_id, params.recipient_uid))
        return "会话转接成功"

    monkeypatch.setattr(module, "_knowledge_service", lambda: FakeKnowledgeService())
    monkeypatch.setattr(module, "_transfer_to_human", fake_transfer)

    result = execute_tool(
        "search_knowledge",
        '{"query":"快递员什么时候联系我","shop_id":"shop-1","goods_id":123456789,"scene":"售中"}',
        {"user_id": "user-1", "recipient_uid": "buyer-1"},
    )

    assert calls == [
        ("scene", "insale", "shop-1", 123456789, "快递员什么时候联系我", 2),
    ]
    assert transfer_calls == [("shop-1", "user-1", "buyer-1")]
    assert result == "会话转接成功"


def test_search_knowledge_tool_description_uses_configured_query_examples(monkeypatch):
    monkeypatch.setattr(
        module,
        "get_config",
        lambda key, default=None: ["尺码", "材质", "洗涤方式"]
        if key == "agent.search_knowledge_query_examples"
        else default,
    )

    tools = get_tools_for_llm()
    description = next(
        tool["function"]["description"]
        for tool in tools
        if tool["function"]["name"] == "search_knowledge"
    )

    assert "尺码、材质、洗涤方式" in description
    assert "续航" not in description
    assert "充电" not in description


def test_format_scene_results_skips_malformed_items():
    formatted = KnowledgeService().format_scene_results(
        [
            "bad",
            {
                "section_title": "参数",
                "answer": "续航以页面为准",
                "score": 12,
                "match_type": "keyword",
            },
        ]
    )

    assert "参数" in formatted
    assert "续航以页面为准" in formatted


def test_format_scene_results_tolerates_non_text_fields():
    formatted = KnowledgeService().format_scene_results(
        [
            {
                "section_title": 123,
                "sub_intent": None,
                "answer": None,
                "score": ["bad-score"],
                "match_type": None,
            },
        ]
    )

    assert "123" in formatted
    assert "bad-score" in formatted


def test_format_scene_results_includes_candidate_metadata():
    formatted = KnowledgeService().format_scene_results(
        [
            {
                "section_title": "续航参数",
                "sub_intent": "续航时长",
                "answer": "最高档约2小时",
                "tags": "parameter_type:duration,unit:hour",
                "source_type": "product",
                "score": 180,
                "match_type": "keyword",
            },
        ]
    )

    assert "标签：parameter_type:duration,unit:hour" in formatted
    assert "来源：product" in formatted
    assert "匹配：keyword" in formatted


def test_format_search_result_handles_non_dict_result():
    assert KnowledgeService().format_search_result(["bad"]) == "未找到相关知识。"


def test_format_search_result_skips_malformed_list_items():
    formatted = KnowledgeService().format_search_result(
        {
            "product_knowledge": ["bad-product"],
            "customer_service_knowledge": ["bad-cs"],
        }
    )

    assert formatted == "未找到相关知识。"


def test_format_search_result_ignores_malformed_hit_maps():
    product = SimpleNamespace(
        id=1,
        goods_name="测试商品",
        goods_id=123,
        price="9.90",
        extracted_content="基础知识",
    )

    formatted = KnowledgeService().format_search_result(
        {
            "product_knowledge": [product],
            "product_knowledge_hits": ["bad"],
            "product_force_full_content": ["bad"],
        }
    )

    assert "测试商品" in formatted
    assert "基础知识" in formatted


def test_format_search_result_tolerates_none_and_non_text_fields():
    product = SimpleNamespace(
        id=1,
        goods_name="测试商品",
        goods_id=123,
        price=None,
        extracted_content=12345,
    )
    cs = SimpleNamespace(title=123, content=None)

    formatted = KnowledgeService().format_search_result(
        {
            "product_knowledge": [product],
            "customer_service_knowledge": [cs],
        }
    )

    assert "测试商品" in formatted
    assert "12345" in formatted
    assert "123" in formatted


def test_format_search_result_tolerates_partial_result_objects():
    product = SimpleNamespace(
        goods_name="测试商品",
        goods_id=123,
    )
    cs = SimpleNamespace(content="客服答案")

    formatted = KnowledgeService().format_search_result(
        {
            "product_knowledge": [product],
            "customer_service_knowledge": [cs],
        }
    )

    assert "测试商品" in formatted
    assert "客服答案" in formatted
    assert "命中客服知识" in formatted
