import asyncio

from Agent.CustomerAgent.custom.customer_agent import CustomerAgent
from Agent.CustomerAgent.custom.message_builder import MessageBuilder
from Agent.CustomerAgent.custom.session_manager import SessionManager
from bridge.context import ChannelType, Context, ContextType
from bridge.reply import ReplyType


def test_media_only_reply_is_saved_to_session_history(tmp_path):
    agent = CustomerAgent(db_path=str(tmp_path / "agent.db"))
    agent._is_initialized = True
    agent._message_builder = MessageBuilder()
    agent._session_manager = SessionManager(db_path=str(tmp_path / "agent.db"))

    async def no_order_context(_dependencies):
        return None

    agent._refresh_order_context = no_order_context
    context = Context.create_pinduoduo_context(
        content="https://example.com/chat-img/a.jpg",
        user_msg_type=ContextType.IMAGE,
        shop_id="shop-1",
        user_id="seller-1",
        from_uid="buyer-1",
        channel_type=ChannelType.PINDUODUO,
    )

    reply = asyncio.run(agent.async_reply("https://example.com/chat-img/a.jpg", context))

    assert reply.type == ReplyType.TEXT
    assert reply.content == "麻烦您说下具体想确认哪里"

    session_id = agent._build_session_id(context, agent._message_builder.build_dependencies(context))
    history = agent._session_manager.get_history(session_id)
    assert [(item["role"], item["content"]) for item in history] == [
        ("user", "https://example.com/chat-img/a.jpg"),
        ("assistant", "麻烦您说下具体想确认哪里"),
    ]
