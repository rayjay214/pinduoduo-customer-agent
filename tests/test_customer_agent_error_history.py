import asyncio
import threading

from Agent.CustomerAgent.custom.customer_agent import CustomerAgent
from Agent.CustomerAgent.custom.message_builder import MessageBuilder
from Agent.CustomerAgent.custom.session_manager import SessionManager
from Agent.CustomerAgent.custom.llm_client import LLMResponse
from bridge.context import ChannelType, Context, ContextType
from bridge.reply import ReplyType


def test_async_reply_saves_fallback_assistant_when_error_happens_after_user_saved(tmp_path):
    agent = CustomerAgent(db_path=str(tmp_path / "agent.db"))
    agent._is_initialized = True
    agent._message_builder = MessageBuilder()
    agent._session_manager = SessionManager(db_path=str(tmp_path / "agent.db"))

    async def no_order_context(_dependencies):
        return None

    async def failing_agent_loop(_messages, _dependencies):
        raise RuntimeError("llm failed after user saved")

    agent._refresh_order_context = no_order_context
    agent._run_agent_loop = failing_agent_loop

    context = Context.create_pinduoduo_context(
        content="你好",
        msg_id="msg-1",
        user_msg_type=ContextType.TEXT,
        shop_id="shop-1",
        user_id="seller-1",
        from_uid="buyer-1",
        channel_type=ChannelType.PINDUODUO,
    )

    reply = asyncio.run(agent.async_reply("你好", context))

    assert reply.type == ReplyType.TEXT
    assert reply.content == "亲，客服正在为您处理，请稍等片刻哦～"

    session_id = agent._build_session_id(context, agent._message_builder.build_dependencies(context))
    history = agent._session_manager.get_history(session_id)
    assert [(item["role"], item["content"]) for item in history] == [
        ("user", "你好"),
        ("assistant", "亲，客服正在为您处理，请稍等片刻哦～"),
    ]


def test_async_reply_returns_fallback_when_error_happens_before_user_saved(tmp_path):
    agent = CustomerAgent(db_path=str(tmp_path / "agent.db"))
    agent._is_initialized = True
    agent._session_manager = SessionManager(db_path=str(tmp_path / "agent.db"))

    class BrokenMessageBuilder:
        def build_dependencies(self, _context):
            raise RuntimeError("dependency build failed")

    agent._message_builder = BrokenMessageBuilder()

    context = Context.create_pinduoduo_context(
        content="你好",
        msg_id="msg-1",
        user_msg_type=ContextType.TEXT,
        shop_id="shop-1",
        user_id="seller-1",
        from_uid="buyer-1",
        channel_type=ChannelType.PINDUODUO,
    )

    reply = asyncio.run(agent.async_reply("你好", context))

    assert reply.type == ReplyType.TEXT
    assert reply.content == "亲，客服正在为您处理，请稍等片刻哦～"


def test_async_reply_runs_pre_retrieved_knowledge_in_thread(tmp_path):
    async def run():
        loop_thread_id = threading.get_ident()
        agent = CustomerAgent(db_path=str(tmp_path / "agent.db"))
        agent._is_initialized = True
        agent._message_builder = MessageBuilder()
        agent._session_manager = SessionManager(db_path=str(tmp_path / "agent.db"))
        agent._tools = []
        pre_retrieval_thread_ids = []

        class FakeLLM:
            async def chat(self, messages, tool_choice="auto"):
                return LLMResponse(content="好的", tool_calls=None, raw_response=None)

        async def no_order_context(_dependencies):
            return None

        def fake_pre_retrieved_knowledge(_messages, _query, _dependencies, _customer_scene):
            pre_retrieval_thread_ids.append(threading.get_ident())

        agent._llm_client = FakeLLM()
        agent._refresh_order_context = no_order_context
        agent._inject_pre_retrieved_knowledge = fake_pre_retrieved_knowledge

        context = Context.create_pinduoduo_context(
            content="续航多久",
            msg_id="msg-1",
            user_msg_type=ContextType.TEXT,
            shop_id="shop-1",
            user_id="seller-1",
            from_uid="buyer-1",
            raw_data={"message": {"info": {"goodsID": "123456789"}}},
            channel_type=ChannelType.PINDUODUO,
        )

        reply = await agent.async_reply("续航多久", context)

        assert reply.content == "好的"
        assert pre_retrieval_thread_ids
        assert pre_retrieval_thread_ids[-1] != loop_thread_id

    asyncio.run(run())
