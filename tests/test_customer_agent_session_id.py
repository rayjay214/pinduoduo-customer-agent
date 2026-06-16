from Agent.CustomerAgent.custom import customer_agent as customer_agent_module
from Agent.CustomerAgent.custom.customer_agent import CustomerAgent
from bridge.context import ChannelType, Context, ContextType


def test_fallback_session_id_uses_context_identity_not_query_text():
    agent = CustomerAgent()
    buyer_a = Context.create_pinduoduo_context(
        content="在吗",
        msg_id="msg-1",
        from_uid="buyer-a",
        user_msg_type=ContextType.TEXT,
        shop_id="shop-1",
        user_id="seller-1",
        channel_type=ChannelType.PINDUODUO,
    )
    buyer_b = Context.create_pinduoduo_context(
        content="在吗",
        msg_id="msg-2",
        from_uid="buyer-b",
        user_msg_type=ContextType.TEXT,
        shop_id="shop-1",
        user_id="seller-1",
        channel_type=ChannelType.PINDUODUO,
    )

    assert agent._build_fallback_session_id(buyer_a) != agent._build_fallback_session_id(buyer_b)


def test_fallback_session_id_is_stable_for_same_context_identity():
    agent = CustomerAgent()
    context = Context.create_pinduoduo_context(
        content="在吗",
        msg_id="msg-1",
        from_uid="buyer-a",
        user_msg_type=ContextType.TEXT,
        shop_id="shop-1",
        user_id="seller-1",
        channel_type=ChannelType.PINDUODUO,
    )

    assert agent._build_fallback_session_id(context) == agent._build_fallback_session_id(context)


def test_session_id_accepts_mapping_kwargs_without_dependencies():
    agent = CustomerAgent()
    context = Context(
        type=ContextType.TEXT,
        content="在吗",
        channel_type=ChannelType.PINDUODUO,
        kwargs={
            "shop_id": "shop-1",
            "user_id": "seller-1",
            "from_uid": "buyer-1",
        },
    )

    assert agent._build_session_id(context, {}) == "pinduoduo:shop-1:seller-1:buyer-1"


def test_session_id_without_buyer_uid_does_not_share_unknown_session():
    agent = CustomerAgent()
    first = Context(
        type=ContextType.TEXT,
        content="在吗",
        channel_type=ChannelType.PINDUODUO,
        kwargs={
            "msg_id": "msg-1",
            "shop_id": "shop-1",
            "user_id": "seller-1",
        },
    )
    second = Context(
        type=ContextType.TEXT,
        content="在吗",
        channel_type=ChannelType.PINDUODUO,
        kwargs={
            "msg_id": "msg-2",
            "shop_id": "shop-1",
            "user_id": "seller-1",
        },
    )

    first_session = agent._build_session_id(first, {})
    second_session = agent._build_session_id(second, {})

    assert first_session != "pinduoduo:shop-1:seller-1:unknown"
    assert second_session != "pinduoduo:shop-1:seller-1:unknown"
    assert first_session != second_session


def test_session_id_fallback_reads_raw_data_identity_when_kwargs_uid_missing():
    agent = CustomerAgent()
    context = Context(
        type=ContextType.TEXT,
        content="在吗",
        channel_type=ChannelType.PINDUODUO,
        kwargs={
            "shop_id": "shop-1",
            "user_id": "seller-1",
            "raw_data": {
                "message": {
                    "msg_id": "msg-1",
                    "from": {"uid": "buyer-from-raw"},
                    "to": {"uid": "seller-raw"},
                }
            },
        },
    )

    assert agent._build_session_id(context, {}) == agent._build_fallback_session_id(context)


def test_fallback_session_id_accepts_mapping_kwargs():
    agent = CustomerAgent()
    buyer_a = Context(
        type=ContextType.TEXT,
        content="在吗",
        channel_type=ChannelType.PINDUODUO,
        kwargs={
            "msg_id": "msg-1",
            "shop_id": "shop-1",
            "user_id": "seller-1",
            "from_uid": "buyer-a",
        },
    )
    buyer_b = Context(
        type=ContextType.TEXT,
        content="在吗",
        channel_type=ChannelType.PINDUODUO,
        kwargs={
            "msg_id": "msg-2",
            "shop_id": "shop-1",
            "user_id": "seller-1",
            "from_uid": "buyer-b",
        },
    )

    assert agent._build_fallback_session_id(buyer_a) != agent._build_fallback_session_id(buyer_b)


def test_session_goods_id_cache_uses_lru_eviction(monkeypatch):
    monkeypatch.setattr(customer_agent_module, "SESSION_GOODS_ID_CACHE_LIMIT", 3)
    agent = CustomerAgent()

    agent._remember_session_goods_id("s1", 101)
    agent._remember_session_goods_id("s2", 102)
    agent._remember_session_goods_id("s3", 103)

    restored = {}
    agent._restore_session_goods_id("s1", restored)

    assert restored["goods_id"] == 101

    agent._remember_session_goods_id("s4", 104)

    assert "s1" in agent._session_goods_id_cache
    assert "s2" not in agent._session_goods_id_cache
    assert list(agent._session_goods_id_cache.keys()) == ["s3", "s1", "s4"]


def test_pre_retrieved_knowledge_uses_di_knowledge_service(monkeypatch):
    agent = CustomerAgent()
    messages = []
    calls = []

    class FakeKnowledgeService:
        def search_scene_knowledge(self, scene, shop_id, goods_id, query, limit):
            calls.append(
                {
                    "scene": scene,
                    "shop_id": shop_id,
                    "goods_id": goods_id,
                    "query": query,
                    "limit": limit,
                }
            )
            return [
                {
                    "answer": "店铺通用答案",
                    "section_title": "物流",
                    "sub_intent": "快递",
                }
            ]

    fake_service = FakeKnowledgeService()
    monkeypatch.setattr(customer_agent_module.container, "get", lambda _service_type: fake_service)

    agent._inject_pre_retrieved_knowledge(
        messages,
        "发什么快递",
        {"shop_id": "shop-1", "goods_id": 123},
        "presale",
    )

    assert calls
    assert "店铺通用答案" in messages[0]["content"]
