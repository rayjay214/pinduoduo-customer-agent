from datetime import time

from utils import night_mode


def test_night_mode_replies_use_configured_time_text(monkeypatch):
    monkeypatch.setattr(
        night_mode,
        "get_night_mode_time_range",
        lambda: (time(22, 30), time(9, 30)),
    )

    night_mode.reset_night_mode_reply_state()

    first = night_mode.get_night_mode_reply("shop:user:buyer")
    second = night_mode.get_night_mode_reply("shop:user:buyer")

    assert "09:30-22:30" in first
    assert "09:30后" in second
    assert "早上8点" not in first
    assert "早上8点" not in second


def test_night_mode_replies_can_be_configured(monkeypatch):
    class FakeConfig:
        def get(self, key, default=None):
            if key == "night_mode.reply_templates":
                return ["值班结束，请{resume_text}后再联系"]
            return default

    import config

    monkeypatch.setattr(config, "config", FakeConfig())
    monkeypatch.setattr(
        night_mode,
        "get_night_mode_time_range",
        lambda: (time(22, 30), time(9, 30)),
    )

    assert night_mode.get_night_mode_replies() == ("值班结束，请09:30后再联系",)


def test_empty_night_mode_reply_templates_fall_back_to_default(monkeypatch):
    class FakeConfig:
        def get(self, key, default=None):
            if key == "night_mode.reply_templates":
                return []
            return default

    import config

    monkeypatch.setattr(config, "config", FakeConfig())
    monkeypatch.setattr(
        night_mode,
        "get_night_mode_time_range",
        lambda: (time(22, 30), time(9, 30)),
    )

    replies = night_mode.get_night_mode_replies()

    assert len(replies) == len(night_mode.DEFAULT_NIGHT_MODE_REPLY_TEMPLATES)
    assert "09:30-22:30" in replies[0]


def test_night_mode_time_range_falls_back_when_config_read_fails(monkeypatch):
    class BrokenConfig:
        def get(self, _key, _default=None):
            raise RuntimeError("config unavailable")

    import config

    monkeypatch.setattr(config, "config", BrokenConfig())

    assert night_mode.get_night_mode_time_range() == (time(23, 0), time(8, 0))


def test_night_mode_time_range_masks_config_read_error(monkeypatch):
    class BrokenConfig:
        def get(self, _key, _default=None):
            raise RuntimeError("token=secret-token")

    class FakeLogger:
        def __init__(self):
            self.messages = []

        def debug(self, message):
            self.messages.append(str(message))

    import config

    fake_logger = FakeLogger()
    monkeypatch.setattr(config, "config", BrokenConfig())
    monkeypatch.setattr(night_mode, "logger", fake_logger)

    assert night_mode.get_night_mode_time_range() == (time(23, 0), time(8, 0))

    joined = "\n".join(fake_logger.messages)
    assert "secret-token" not in joined
    assert "token=***" in joined


def test_night_mode_reply_state_prunes_expired_entries(monkeypatch):
    night_mode.reset_night_mode_reply_state()
    monkeypatch.setattr(night_mode, "NIGHT_MODE_REPLY_STATE_TTL_SECONDS", 10)
    monkeypatch.setattr(night_mode.monotonic_time, "monotonic", lambda: 100.0)
    night_mode._reply_stages["old"] = (3, 80.0)
    night_mode._reply_stages["fresh"] = (1, 95.0)

    night_mode.get_night_mode_reply("current")

    assert "old" not in night_mode._reply_stages
    assert "fresh" in night_mode._reply_stages
    assert "current" in night_mode._reply_stages


def test_night_mode_reply_state_prunes_over_limit(monkeypatch):
    night_mode.reset_night_mode_reply_state()
    monkeypatch.setattr(night_mode, "NIGHT_MODE_REPLY_STATE_TTL_SECONDS", 1000)
    monkeypatch.setattr(night_mode, "NIGHT_MODE_REPLY_STATE_MAX_ENTRIES", 2)
    monkeypatch.setattr(night_mode.monotonic_time, "monotonic", lambda: 100.0)
    night_mode._reply_stages["oldest"] = (1, 90.0)
    night_mode._reply_stages["middle"] = (1, 95.0)

    night_mode.get_night_mode_reply("newest")

    assert "oldest" not in night_mode._reply_stages
    assert set(night_mode._reply_stages) == {"middle", "newest"}
