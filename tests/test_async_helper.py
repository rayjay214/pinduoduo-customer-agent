import asyncio
import threading

import pytest

from utils.async_helper import run_async


def test_run_async_returns_result_without_running_loop():
    async def ok():
        return "done"

    assert run_async(ok()) == "done"


def test_run_async_propagates_runtime_error_from_coroutine():
    async def bad():
        raise RuntimeError("domain failure")

    with pytest.raises(RuntimeError, match="domain failure"):
        run_async(bad())


def test_run_async_inside_running_loop_uses_worker_thread():
    async def run():
        loop_thread_id = threading.get_ident()

        async def get_thread_id():
            return threading.get_ident()

        worker_thread_id = run_async(get_thread_id())

        assert worker_thread_id != loop_thread_id

    asyncio.run(run())


def test_run_async_inside_running_loop_propagates_coroutine_runtime_error():
    async def run():
        async def bad():
            raise RuntimeError("worker domain failure")

        with pytest.raises(RuntimeError, match="worker domain failure"):
            run_async(bad())

    asyncio.run(run())
