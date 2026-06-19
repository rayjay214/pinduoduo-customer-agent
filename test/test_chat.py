"""
交互式测试：初始化一次，后续在控制台输入对话。

用法：
  python test_chat.py
  输入消息回车，Ctrl+C 退出
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


async def main():
    shops = dump_shops()
    if not shops:
        return

    # 默认选第一个店铺
    shop = shops[0]
    account = shop.accounts[0] if shop.accounts else None
    print(f"\n使用店铺: {shop.shop_name} (shop_id={shop.shop_id})")
    if account:
        print(f"使用账号: {account.username} (user_id={account.user_id})")
    print("=" * 50)

    bot = CustomerAgent()

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
