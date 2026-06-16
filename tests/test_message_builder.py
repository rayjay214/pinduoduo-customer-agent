import json

from Agent.CustomerAgent.custom import message_builder as message_builder_module
from Agent.CustomerAgent.custom.message_builder import MessageBuilder
from bridge.context import ChannelType, Context, ContextType


def test_extract_goods_id_from_embedded_biz_context_json():
    raw_data = {
        "message": {
            "biz_context": {
                "mallOrderConfirmNewCardCallBackParam": json.dumps(
                    {"goodsId": "123456789"},
                    ensure_ascii=False,
                )
            }
        }
    }
    context = Context.create_pinduoduo_context(
        content="商品咨询",
        user_msg_type=ContextType.GOODS_INQUIRY,
        raw_data=raw_data,
        channel_type=ChannelType.PINDUODUO,
    )

    assert MessageBuilder._extract_goods_id(context) == 123456789


def test_extract_goods_id_from_standard_message_list_content():
    context = Context(
        type=ContextType.GOODS_INQUIRY,
        content=json.dumps([{"type": "goods_card", "goods_id": "123456789"}]),
    )

    assert MessageBuilder._extract_goods_id(context) == 123456789


def test_extract_goods_id_from_raw_order_goods_list():
    raw_data = {
        "message": {
            "orders": [
                {
                    "orderGoodsList": [
                        {"goodsId": "987654321"}
                    ]
                }
            ]
        }
    }
    context = Context.create_pinduoduo_context(
        content="订单咨询",
        user_msg_type=ContextType.ORDER_INFO,
        raw_data=raw_data,
        channel_type=ChannelType.PINDUODUO,
    )

    assert MessageBuilder._extract_goods_id(context) == 987654321


def test_system_prompt_uses_configured_grounded_knowledge_topics(monkeypatch):
    monkeypatch.setattr(
        message_builder_module,
        "get_config",
        lambda key, default=None: ["尺码", "材质", "洗涤方式"]
        if key == "agent.grounded_knowledge_topics"
        else default,
    )
    monkeypatch.setattr(message_builder_module, "build_image_grounding_instruction", lambda: "")
    monkeypatch.setattr(message_builder_module, "build_version_name_instruction", lambda: "")

    builder = MessageBuilder()

    assert "涉及尺码、材质、洗涤方式时" in builder.system_prompt
    assert "制冷" not in builder.system_prompt
    assert "续航" not in builder.system_prompt
    assert "充电时间" not in builder.system_prompt


def test_build_dependencies_preserves_external_shop_id_as_string():
    context = Context.create_pinduoduo_context(
        content="商品咨询",
        user_msg_type=ContextType.TEXT,
        shop_id="591119888",
        user_id="seller-1",
        from_uid="buyer-1",
        channel_type=ChannelType.PINDUODUO,
    )

    deps = MessageBuilder().build_dependencies(context)

    assert deps["shop_id"] == "591119888"


def test_build_dependencies_string_false_disables_turn_context(monkeypatch):
    monkeypatch.setattr(
        message_builder_module,
        "get_config",
        lambda key, default=None: "false" if key == "enable_turn_context" else default,
    )
    context = Context.create_pinduoduo_context(
        content="内容：你好",
        user_msg_type=ContextType.TEXT,
        shop_id="shop-1",
        user_id="seller-1",
        from_uid="buyer-1",
        channel_type=ChannelType.PINDUODUO,
    )

    deps = MessageBuilder().build_dependencies(context)

    assert "turn_context" not in deps


def test_build_dependencies_accepts_mapping_kwargs():
    context = Context(
        type=ContextType.TEXT,
        content="商品咨询",
        channel_type=ChannelType.PINDUODUO,
        kwargs={
            "shop_id": "shop-1",
            "user_id": "seller-1",
            "from_uid": "buyer-1",
            "shop_name": "测试店铺",
            "raw_data": {"message": {"info": {"goodsId": "123456789"}}},
        },
    )

    deps = MessageBuilder().build_dependencies(context)

    assert deps["shop_id"] == "shop-1"
    assert deps["user_id"] == "seller-1"
    assert deps["recipient_uid"] == "buyer-1"
    assert deps["shop_name"] == "测试店铺"
    assert deps["goods_id"] == 123456789


def test_build_messages_ignores_malformed_history_items():
    builder = MessageBuilder()

    messages = builder.build_messages(
        query="当前问题",
        history=[
            None,
            {"role": "assistant"},
            {"content": "missing role"},
            {"role": "tool", "content": "工具结果"},
            {"role": "assistant", "content": {"text": "结构化旧消息"}},
            {"role": "user", "content": "正常历史"},
        ],
    )

    assert {"role": "assistant", "content": "{'text': '结构化旧消息'}"} in messages
    assert {"role": "user", "content": "正常历史"} in messages
    assert messages[-1] == {"role": "user", "content": "当前问题"}


def test_build_messages_treats_non_mapping_dependencies_as_empty_context():
    builder = MessageBuilder()

    messages = builder.build_messages(
        query="当前问题",
        history=[],
        dependencies=object(),
    )

    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "当前问题"}


def test_build_messages_normalizes_dirty_image_url_before_llm_payload():
    builder = MessageBuilder()

    messages = builder.build_messages(
        query="客户发送了图片：https://img.example.com/a.jpeg%22",
        history=[],
        dependencies={
            "context_type": "image",
            "media_type": "image",
            "media_url": "https://img.example.com/a.jpeg%22",
        },
    )

    assert messages[-1]["content"][1]["image_url"]["url"] == "https://img.example.com/a.jpeg"


def test_extract_goods_id_tolerates_non_dict_raw_message():
    context = Context.create_pinduoduo_context(
        content="普通文本",
        user_msg_type=ContextType.TEXT,
        raw_data={"message": "bad payload"},
        channel_type=ChannelType.PINDUODUO,
    )

    assert MessageBuilder._extract_goods_id(context) is None
