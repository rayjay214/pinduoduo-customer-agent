import inspect
import asyncio

import pytest

from Channel import channel as channel_module
from Channel.channel import Channel
from Channel.pinduoduo.pdd_channel import PDDChannel
from core.connection_status import ConnectionStatusManager


def test_channel_base_class_enforces_abstract_contract():
    with pytest.raises(TypeError):
        Channel()


def test_pdd_channel_satisfies_channel_contract():
    assert not inspect.isabstract(PDDChannel)
    channel = PDDChannel(status_manager=ConnectionStatusManager())

    assert hasattr(channel, "start_account")
    assert hasattr(channel, "stop_account")
    assert hasattr(channel, "stop_all_connections")


class _ConcreteChannel(Channel):
    def __init__(self):
        super().__init__()
        self.channel_name = "pinduoduo"

    async def start_account(self, shop_id, user_id, on_success, on_failure):
        return None

    async def stop_account(self, shop_id, user_id):
        return None

    async def stop_all_connections(self):
        return None


def test_channel_add_shop_passes_logo_and_description_without_argument_shift(monkeypatch):
    calls = []

    class _FakeDB:
        def get_shop(self, channel_name, shop_id):
            return None

        def add_shop(self, channel_name, shop_id, shop_name, shop_logo, description=None):
            calls.append((channel_name, shop_id, shop_name, shop_logo, description))
            return True

    monkeypatch.setattr(channel_module, "db_manager", _FakeDB())

    channel = _ConcreteChannel()
    result = asyncio.run(
        channel.add_shop(
            shop_id="shop-1",
            shop_name="测试店铺",
            shop_logo="https://example.com/logo.png",
            description="店铺描述",
        )
    )

    assert result is True
    assert calls == [
        (
            "pinduoduo",
            "shop-1",
            "测试店铺",
            "https://example.com/logo.png",
            "店铺描述",
        )
    ]
