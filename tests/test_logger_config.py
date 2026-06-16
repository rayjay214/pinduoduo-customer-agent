from utils.logger_config import SimpleLoggerConfig


def test_logger_flags_parse_common_false_values(monkeypatch):
    monkeypatch.setenv("BUSINESS_LOGGING", "0")
    monkeypatch.setenv("UI_LOGGING", "off")

    config = SimpleLoggerConfig()

    assert config.is_business_logging_enabled() is False
    assert config.is_ui_logging_enabled() is False


def test_invalid_log_retention_days_falls_back(monkeypatch):
    monkeypatch.setenv("LOG_RETENTION_DAYS", "bad")

    config = SimpleLoggerConfig()

    assert config.get_log_retention_days() == 7


def test_log_retention_days_has_positive_floor(monkeypatch):
    monkeypatch.setenv("LOG_RETENTION_DAYS", "0")

    config = SimpleLoggerConfig()

    assert config.get_log_retention_days() == 1
