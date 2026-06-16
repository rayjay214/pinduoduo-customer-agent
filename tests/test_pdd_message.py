from Channel.pinduoduo.pdd_message import PDDChatMessage
from bridge.context import ContextType


def test_pdd_chat_message_sets_timestamp_from_message_time():
    message = PDDChatMessage(
        {
            "response": "push",
            "message": {
                "msg_id": "msg-1",
                "type": 0,
                "content": "你好",
                "from": {"role": "user", "uid": "buyer-1"},
                "to": {"role": "mall_cs", "uid": "seller-1"},
                "time": "2026-06-13 01:00:00",
            },
        }
    )

    assert message.timestamp == "2026-06-13 01:00:00"
    assert message.user_msg_type == ContextType.TEXT


def test_goods_spec_with_work_order_text_is_not_system_card_without_platform_signal():
    message = PDDChatMessage(
        {
            "response": "push",
            "message": {
                "msg_id": "msg-2",
                "type": 64,
                "content": "工单收纳夹 商品规格",
                "info": {
                    "data": {
                        "title": "工单收纳夹",
                        "goodsID": "123456789",
                        "goodsName": "工单收纳夹",
                    }
                },
                "from": {"role": "user", "uid": "buyer-1"},
                "to": {"role": "mall_cs", "uid": "seller-1"},
                "time": "2026-06-13 01:01:00",
            },
        }
    )

    assert message.user_msg_type == ContextType.GOODS_SPEC
    assert message.content["goods_id"] == "123456789"


def test_goods_spec_platform_work_order_card_is_system_hint():
    message = PDDChatMessage(
        {
            "response": "push",
            "message": {
                "msg_id": "msg-3",
                "type": 64,
                "template_name": "consumer_service_work_order_card",
                "content": "消费者服务工单",
                "info": {
                    "key": "work_order_detail",
                    "data": {"title": "消费者服务工单"},
                },
                "from": {"role": "user", "uid": "buyer-1"},
                "to": {"role": "mall_cs", "uid": "seller-1"},
                "time": "2026-06-13 01:02:00",
            },
        }
    )

    assert message.user_msg_type == ContextType.SYSTEM_HINT
    assert message.content == "消费者服务工单"


def test_order_info_extracts_goods_from_order_goods_list():
    message = PDDChatMessage(
        {
            "response": "push",
            "message": {
                "msg_id": "msg-4",
                "type": 0,
                "sub_type": 1,
                "info": {},
                "orders": [
                    {
                        "orderSequenceNo": "order-123",
                        "shippingStatus": 1,
                        "status": 3,
                        "payStatus": 2,
                        "trackingNumber": "YT123",
                        "shippingTime": "2026-06-13 01:03:00",
                        "afterSalesStatus": 0,
                        "orderGoodsList": [
                            {
                                "goodsId": "987654321",
                                "goodsName": "订单商品",
                                "goodsPrice": 2590,
                                "thumbUrl": "https://example.com/goods.jpg",
                                "spec": "蓝色",
                            }
                        ],
                    }
                ],
                "from": {"role": "user", "uid": "buyer-1"},
                "to": {"role": "mall_cs", "uid": "seller-1"},
                "time": "2026-06-13 01:03:00",
            },
        }
    )

    assert message.user_msg_type == ContextType.ORDER_INFO
    assert message.content["order_id"] == "order-123"
    assert message.content["goods_id"] == "987654321"
    assert message.content["goods_name"] == "订单商品"
    assert message.content["goods_price"] == 2590
    assert message.content["goods_thumb_url"] == "https://example.com/goods.jpg"
    assert message.content["spec"] == "蓝色"
    assert message.content["shipping_status"] == 1
    assert message.content["status"] == 3
    assert message.content["pay_status"] == 2
    assert message.content["tracking_number"] == "YT123"
    assert message.content["shipping_time"] == "2026-06-13 01:03:00"
    assert message.content["afterSalesStatus"] == 0


def test_emotion_message_reads_description_from_message_info():
    message = PDDChatMessage(
        {
            "response": "push",
            "message": {
                "msg_id": "msg-5",
                "type": 5,
                "info": {"description": "[微笑]"},
                "from": {"role": "user", "uid": "buyer-1"},
                "to": {"role": "mall_cs", "uid": "seller-1"},
                "time": "2026-06-13 01:04:00",
            },
        }
    )

    assert message.user_msg_type == ContextType.EMOTION
    assert message.content == "[微笑]"


def test_withdraw_message_reads_hint_from_message_info():
    message = PDDChatMessage(
        {
            "response": "push",
            "message": {
                "msg_id": "msg-6",
                "type": 1002,
                "info": {"withdraw_hint": "买家撤回了一条消息"},
                "from": {"role": "user", "uid": "buyer-1"},
                "to": {"role": "mall_cs", "uid": "seller-1"},
                "time": "2026-06-13 01:05:00",
            },
        }
    )

    assert message.user_msg_type == ContextType.WITHDRAW
    assert message.content == "买家撤回了一条消息"


def test_pdd_chat_message_tolerates_non_dict_from_to_fields():
    message = PDDChatMessage(
        {
            "response": "push",
            "message": {
                "msg_id": "msg-7",
                "type": 0,
                "content": "你好",
                "from": "buyer-1",
                "to": ["seller-1"],
                "time": "2026-06-13 01:06:00",
            },
        }
    )

    assert message.user_msg_type == ContextType.TEXT
    assert message.content == "你好"
    assert message.from_uid is None
    assert message.to_uid is None


def test_pdd_chat_message_accepts_string_type_values():
    message = PDDChatMessage(
        {
            "response": "push",
            "message": {
                "msg_id": "msg-8",
                "type": "0",
                "sub_type": "1",
                "info": {"orderSequenceNo": "order-8"},
                "from": {"role": "user", "uid": "buyer-1"},
                "to": {"role": "mall_cs", "uid": "seller-1"},
                "time": "2026-06-13 01:07:00",
            },
        }
    )

    assert message.user_msg_type == ContextType.ORDER_INFO
    assert message.content["order_id"] == "order-8"


def test_pdd_chat_message_tolerates_non_dict_message_envelope():
    message = PDDChatMessage(
        {
            "response": "push",
            "message": "bad payload",
        }
    )

    assert message.user_msg_type == ContextType.SYSTEM_STATUS
    assert "不支持的消息类型" in message.content


def test_pdd_chat_message_tolerates_non_dict_top_level_payload():
    message = PDDChatMessage("bad payload")

    assert message.user_msg_type == ContextType.SYSTEM_STATUS
    assert "不支持的消息类型" in message.content


def test_pdd_chat_message_tolerates_non_finite_message_type():
    message = PDDChatMessage(
        {
            "response": "push",
            "message": {
                "msg_id": "msg-non-finite",
                "type": float("inf"),
                "content": "异常类型",
                "from": {"role": "user", "uid": "buyer-1"},
                "to": {"role": "mall_cs", "uid": "seller-1"},
            },
        }
    )

    assert message.user_msg_type == ContextType.SYSTEM_STATUS
    assert "不支持的消息类型" in message.content


def test_pdd_chat_message_mall_cs_branch_uses_safe_message_content():
    message = PDDChatMessage(
        {
            "response": "push",
            "message": {
                "type": 0,
                "content": "客服回复",
                "from": {"role": "mall_cs", "uid": "seller-1"},
            },
        }
    )

    assert message.user_msg_type == ContextType.MALL_CS
    assert message.content == "客服回复"
