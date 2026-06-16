import asyncio

import pytest

from core.di_container import DIContainer


class CaptureLogger:
    def __init__(self):
        self.messages = []

    def debug(self, message):
        self.messages.append(str(message))

    def info(self, message):
        self.messages.append(str(message))

    def warning(self, message):
        self.messages.append(str(message))

    def error(self, message):
        self.messages.append(str(message))


class BrokenService:
    def __init__(self):
        raise RuntimeError("token=secret-token")


class DisposableService:
    def dispose(self):
        raise RuntimeError("cookies=secret-cookie")


def test_di_container_masks_sensitive_sync_creation_error():
    container = DIContainer()
    container.logger = CaptureLogger()

    with pytest.raises(RuntimeError):
        container._create_instance(BrokenService)

    joined = "\n".join(container.logger.messages)
    assert "secret-token" not in joined
    assert "token=***" in joined


def test_di_container_masks_sensitive_async_creation_error():
    async def run():
        container = DIContainer()
        container.logger = CaptureLogger()

        with pytest.raises(RuntimeError):
            await container._create_instance_async(BrokenService)

        joined = "\n".join(container.logger.messages)
        assert "secret-token" not in joined
        assert "token=***" in joined

    asyncio.run(run())


def test_di_container_masks_sensitive_dispose_error():
    container = DIContainer()
    container.logger = CaptureLogger()
    container._singletons["demo"] = DisposableService()

    container.dispose()

    joined = "\n".join(container.logger.messages)
    assert "secret-cookie" not in joined
    assert "cookies=***" in joined
