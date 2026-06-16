from concurrent.futures import ThreadPoolExecutor
from Channel.pinduoduo.core.pdd_utils import (
    get_pdd_connected_count,
    get_pdd_connection_summary,
)
from core.connection_status import ConnectionState, ConnectionStatusManager
from core.di_container import container
from types import SimpleNamespace


def test_pdd_utils_resolve_connection_status_manager_from_container():
    original_services = dict(container._services)
    original_singletons = dict(container._singletons)
    try:
        manager = ConnectionStatusManager()
        manager.update_status("shop-1", "user-1", "客服1", ConnectionState.CONNECTED)
        manager.update_status("shop-2", "user-2", "客服2", ConnectionState.ERROR, error="boom")
        container.register_singleton(ConnectionStatusManager, instance=manager)

        assert get_pdd_connected_count() == 1
        assert get_pdd_connection_summary() == {
            "total": 2,
            "connected": 1,
            "connecting": 0,
            "reconnecting": 0,
            "error": 1,
            "disconnected": 0,
        }
    finally:
        container._services = original_services
        container._singletons = original_singletons


def test_pdd_connection_summary_tolerates_unknown_state_values():
    original_services = dict(container._services)
    original_singletons = dict(container._singletons)
    try:
        manager = SimpleNamespace(
            get_all_status=lambda: [
                SimpleNamespace(state=SimpleNamespace(value="maintenance")),
                SimpleNamespace(state=None),
            ]
        )
        container.register_singleton(ConnectionStatusManager, instance=manager)

        summary = get_pdd_connection_summary()

        assert summary["total"] == 2
        assert summary["unknown"] == 2
    finally:
        container._services = original_services
        container._singletons = original_singletons


def test_connection_status_manager_initialization_is_thread_safe():
    def create_and_update(index):
        manager = ConnectionStatusManager()
        manager.update_status(
            f"shop-{index}",
            f"user-{index}",
            f"客服{index}",
            ConnectionState.CONNECTED,
        )
        return manager.get_status(f"shop-{index}", f"user-{index}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        statuses = list(executor.map(create_and_update, range(20)))

    assert all(status is not None for status in statuses)
    assert all(status.state == ConnectionState.CONNECTED for status in statuses)
