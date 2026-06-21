"""
交互式测试：初始化一次，后续在控制台输入对话。

用法：
  python test_chat.py                              # 默认调用真实 PDD API 获取订单
  python test_chat.py --scene presale              # 模拟售前（无订单）
  python test_chat.py --scene insale               # 模拟售中（已支付待发货）
  python test_chat.py --scene aftersale            # 模拟售后（已签收）
  
支持场景：
  presale    - 售前，无关联订单
  insale     - 售中，已支付待发货
  aftersale  - 售后，已签收
"""
import asyncio, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config as _app_config
from core.di_container import configure_standard_services
configure_standard_services(_app_config)

from bridge.context import Context, ContextType, ChannelType, PinduoduoKwargs
from Agent.CustomerAgent.custom.customer_agent import CustomerAgent
from database.knowledge_service import KnowledgeService
from database.models import Shop, Account
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from typing import Dict, Any


def dump_shops():
    ks = KnowledgeService()
    with ks.get_session() as session:
        stmt = select(Shop).options(joinedload(Shop.accounts)).order_by(Shop.shop_name.asc())
        shops = list(session.scalars(stmt).unique())
    if not shops:
        print("数据库中没有店铺，请先在 UI 中添加账号")
        return None
    for s in shops:
        print(f"  [{s.id}] DB ID={s.id}  店铺ID={s.shop_id}  名称={s.shop_name}")
        for a in s.accounts:
            print(f"       账号: user_id={a.user_id}  username={a.username}")
    return shops


# ============================================================
# Mock order context data for each scene
# ============================================================
MOCK_ORDER_CONTEXTS = {
    "presale": {},  # no order keys → fallback to presale
    "insale": {
        "order_context_text": (
            "【当前订单上下文】\n"
            "- 当前订单状态：已支付待发货\n"
            "- 当前业务场景：售中\n"
            "- 最近订单号：mock_order_insale_001\n"
            "- 订单商品：测试商品"
        ),
        "order_scene_hint": "insale",
        "order_business_status": "已支付待发货",
        "order_shipping_status": "not_shipped",
        "order_latest_trace": "",
        "order_id": "mock_order_insale_001",
    },
    "aftersale": {
        "order_context_text": (
            "【当前订单上下文】\n"
            "- 当前订单状态：已签收\n"
            "- 当前业务场景：售后\n"
            "- 最近订单号：mock_order_aftersale_001\n"
            "- 订单商品：测试商品\n"
            "- 最新物流状态：您的包裹已签收，签收人：本人"
        ),
        "order_scene_hint": "aftersale",
        "order_business_status": "已签收",
        "order_shipping_status": "signed",
        "order_latest_trace": "【物流跟踪】您的包裹已签收，签收人：本人",
        "order_id": "mock_order_aftersale_001",
    },
}


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="交互式测试聊天机器人")
    parser.add_argument(
        "--scene", "-s",
        choices=["presale", "insale", "aftersale"],
        default=None,
        help="模拟订单场景（不传则调用真实 PDD API 获取订单）",
    )
    args = parser.parse_args()

    shops = dump_shops()
    if not shops:
        return

    # 默认选第一个店铺
    shop = shops[0]
    account = shop.accounts[0] if shop.accounts else None
    print(f"\n使用店铺: {shop.shop_name} (shop_id={shop.shop_id})")
    if account:
        print(f"使用账号: {account.username} (user_id={account.user_id})")

    if args.scene:
        print(f"模拟订单场景: {args.scene}（已绕过真实 PDD API）")
    else:
        print("订单场景: 自动（通过真实 PDD API 获取）")
    print("=" * 50)

    bot = CustomerAgent()

    # 如果指定了模拟场景，劫持 _refresh_order_context 注入 mock 数据
    if args.scene:
        async def _mock_refresh(self, dependencies: Dict[str, Any]) -> None:
            dependencies.update(MOCK_ORDER_CONTEXTS[args.scene])
        bot._refresh_order_context = _mock_refresh.__get__(bot, CustomerAgent)

    context = Context(
        type=ContextType.TEXT,
        content="",
        kwargs=PinduoduoKwargs(
            user_id=account.user_id if account else "test_user",
            shop_id=shop.shop_id,
            from_uid="test_customer_uid",
            to_uid="test_cs_uid",
            nickname="测试用户",
        ),
        channel_type=ChannelType.PINDUODUO,
    )

    print("已就绪，输入消息按回车（Ctrl+C 退出）：")
    while True:
        try:
            query = input("\n>>> ")
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break
        if not query.strip():
            continue
        context.content = query
        reply = await bot.async_reply(query, context)
        print(f"AI: {reply.content}")


if __name__ == "__main__":
    asyncio.run(main())
