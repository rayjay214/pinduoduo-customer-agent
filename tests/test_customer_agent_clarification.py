import asyncio

from Agent.CustomerAgent.custom import customer_agent
from Agent.CustomerAgent.custom.customer_agent import CustomerAgent
from bridge.context import ChannelType, Context, ContextType


class MemorySessionManager:
    def __init__(self):
        self.messages = []

    def get_history(self, session_id):
        return []

    def should_compress(self, session_id):
        return False

    def add_message(self, session_id, role, content):
        self.messages.append({"session_id": session_id, "role": role, "content": content})


class ReadyAgent(CustomerAgent):
    def __init__(self):
        super().__init__()
        self._is_initialized = True
        self._message_builder = None
        self._session_manager = MemorySessionManager()
        self.loop_called = False

    async def initialize_async(self):
        return True

    async def _refresh_order_context(self, dependencies):
        return None

    def _resolve_customer_scene(self, query, history, dependencies):
        return "presale"

    async def _run_agent_loop(self, messages, dependencies):
        self.loop_called = True
        return "LLM 回复"

    def _append_scene_prompt(self, messages, customer_scene):
        return None

    def _append_night_mode_constraint(self, messages):
        return None

    def _append_order_hard_constraints(self, messages, customer_scene, dependencies, session_id, query=""):
        return None

    def _append_image_grounding_constraint(self, messages, dependencies, session_id):
        return None

    def _inject_pre_retrieved_knowledge(self, messages, query, dependencies, customer_scene):
        return None

    def _append_missing_goods_knowledge_constraint(self, messages, query, dependencies, session_id):
        return None


class FakeMessageBuilder:
    def __init__(self, goods_id=None):
        self.goods_id = goods_id

    def build_dependencies(self, context):
        return {
            "shop_id": "shop-1",
            "user_id": "seller-1",
            "recipient_uid": "buyer-1",
            "channel_type": "pinduoduo",
            "context_type": "text",
            "goods_id": self.goods_id,
        }

    def build_messages(self, query, history, dependencies):
        return [{"role": "system", "content": "base"}, {"role": "user", "content": query}]


def _context(content):
    return Context.create_pinduoduo_context(
        content=content,
        msg_id="msg-1",
        from_uid="buyer-1",
        user_msg_type=ContextType.TEXT,
        shop_id="shop-1",
        user_id="seller-1",
        channel_type=ChannelType.PINDUODUO,
    )


def test_missing_goods_parameter_question_asks_for_product_identity(monkeypatch):
    monkeypatch.setattr(customer_agent, "get_config", lambda key, default=None: default)
    agent = ReadyAgent()
    agent._message_builder = FakeMessageBuilder(goods_id=None)

    reply = asyncio.run(agent.async_reply("这款尺寸多大", _context("这款尺寸多大")))

    assert reply.content == "亲，不同款式/规格参数会不一样，麻烦您发一下具体商品链接或点一下商品卡片，我按对应款式帮您确认哦。"
    assert agent.loop_called is False
    assert [message["role"] for message in agent._session_manager.messages] == ["user", "assistant"]
    assert agent._session_manager.messages[-1]["content"] == reply.content


def test_parameter_question_with_goods_id_continues_to_llm(monkeypatch):
    monkeypatch.setattr(customer_agent, "get_config", lambda key, default=None: default)
    agent = ReadyAgent()
    agent._message_builder = FakeMessageBuilder(goods_id=123456789)

    reply = asyncio.run(agent.async_reply("这款尺寸多大", _context("这款尺寸多大")))

    assert reply.content == "LLM 回复"
    assert agent.loop_called is True


def test_shipping_question_without_goods_id_continues_to_llm(monkeypatch):
    monkeypatch.setattr(customer_agent, "get_config", lambda key, default=None: default)
    agent = ReadyAgent()
    agent._message_builder = FakeMessageBuilder(goods_id=None)

    reply = asyncio.run(agent.async_reply("什么快递", _context("什么快递")))

    assert reply.content == "LLM 回复"
    assert agent.loop_called is True
