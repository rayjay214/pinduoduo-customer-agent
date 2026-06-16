import pytest
from pathlib import Path

from config import Config, ConfigError, ConfigValidationError, config_base


def test_default_config_is_deep_copied(tmp_path):
    manager = Config(config_path=tmp_path / "missing.json", auto_create=True)

    manager.set("agent.high_risk_aftersale_transfer_phrases", ["custom phrase"], save=False)

    assert config_base["agent"]["high_risk_aftersale_transfer_phrases"] != ["custom phrase"]
    assert config_base["agent"]["high_risk_aftersale_transfer_phrases"] == []


def test_turn_context_flags_are_part_of_validated_config(tmp_path):
    manager = Config(config_path=tmp_path / "missing.json", auto_create=True)

    manager.set("enable_turn_context", False, save=False)
    manager.set("enable_turn_context_log_only", False, save=False)

    model = manager.get_model()
    assert model.enable_turn_context is False
    assert model.enable_turn_context_log_only is False


def test_intent_specific_score_rules_are_part_of_validated_config(tmp_path):
    manager = Config(config_path=tmp_path / "missing.json", auto_create=True)

    assert manager.get_model().knowledge.intent_specific_score_rules is None

    rules = [
        {
            "query_any": ["缩水"],
            "knowledge_any": ["面料"],
            "score": 77,
        }
    ]
    manager.set("knowledge.intent_specific_score_rules", rules, save=False)

    assert manager.get_model().knowledge.intent_specific_score_rules == rules


def test_agent_rule_lists_are_part_of_validated_config(tmp_path):
    manager = Config(config_path=tmp_path / "missing.json", auto_create=True)

    model = manager.get_model()
    assert "噪音" in model.agent.order_fault_examples
    assert "坏了" in model.agent.night_mode_fault_examples
    assert "退款/退货/赔付" in model.agent.transfer_escalation_examples
    assert "商品参数" in model.agent.search_knowledge_query_examples
    assert "快递" in model.agent.grounded_knowledge_topics
    assert "尺寸" in model.agent.missing_goods_parameter_topics
    assert "参数数值" in model.agent.missing_goods_unverified_fact_examples
    assert "续航" not in model.agent.missing_goods_parameter_topics

    manager.set("agent.grounded_knowledge_topics", ["材质"], save=False)

    assert manager.get_model().agent.grounded_knowledge_topics == ["材质"]


def test_failed_set_rolls_back_invalid_in_memory_value(tmp_path):
    manager = Config(config_path=tmp_path / "missing.json", auto_create=True)
    original_start = manager.get("business_hours.start")

    with pytest.raises(ConfigValidationError):
        manager.set("business_hours.start", "not-a-time", save=False)

    assert manager.get("business_hours.start") == original_start


def test_failed_update_rolls_back_nested_invalid_value(tmp_path):
    manager = Config(config_path=tmp_path / "missing.json", auto_create=True)
    original_start = manager.get("business_hours.start")

    with pytest.raises(ConfigValidationError):
        manager.update({"business_hours": {"start": "not-a-time"}}, save=False)

    assert manager.get("business_hours.start") == original_start


def test_failed_update_does_not_mutate_nested_existing_config(tmp_path):
    manager = Config(config_path=tmp_path / "missing.json", auto_create=True)
    original_prompt = list(manager.get("prompt.instructions"))

    with pytest.raises(ConfigValidationError):
        manager.update(
            {
                "prompt": {"instructions": ["mutated"]},
                "business_hours": {"start": "not-a-time"},
            },
            save=False,
        )

    assert manager.get("prompt.instructions") == original_prompt


def test_get_returns_copy_for_mutable_values(tmp_path):
    manager = Config(config_path=tmp_path / "missing.json", auto_create=True)

    instructions = manager.get("prompt.instructions")
    instructions.append("mutated outside config manager")
    business_hours = manager.get("business_hours")
    business_hours["start"] = "not-a-time"

    assert "mutated outside config manager" not in manager.get("prompt.instructions")
    assert manager.get("business_hours.start") == "08:00"


def test_set_save_failure_rolls_back_in_memory_value(tmp_path, monkeypatch):
    manager = Config(config_path=tmp_path / "config.json", auto_create=True)
    original_start = manager.get("business_hours.start")
    monkeypatch.setattr(manager, "save", lambda: False)

    with pytest.raises(ConfigError):
        manager.set("business_hours.start", "09:00", save=True)

    assert manager.get("business_hours.start") == original_start


def test_update_save_failure_rolls_back_in_memory_value(tmp_path, monkeypatch):
    manager = Config(config_path=tmp_path / "config.json", auto_create=True)
    original_start = manager.get("business_hours.start")
    monkeypatch.setattr(manager, "save", lambda: False)

    with pytest.raises(ConfigError):
        manager.update({"business_hours": {"start": "09:00"}}, save=True)

    assert manager.get("business_hours.start") == original_start


def test_atomic_update_save_failure_rolls_back_in_memory_value(tmp_path, monkeypatch):
    manager = Config(config_path=tmp_path / "config.json", auto_create=True)
    original_start = manager.get("business_hours.start")
    monkeypatch.setattr(manager, "save", lambda: False)

    with pytest.raises(ConfigError):
        with manager.atomic_update() as cfg:
            cfg.set("business_hours.start", "09:00", save=False)

    assert manager.get("business_hours.start") == original_start


def test_save_failure_masks_sensitive_error_output(tmp_path, monkeypatch, capsys):
    manager = Config(config_path=tmp_path / "config.json", auto_create=True)

    def fail_replace(self, target):
        raise OSError("disk full api_key=sk-secret token=raw-token")

    def fail_unlink(self):
        raise OSError("cleanup failed password=hunter2")

    monkeypatch.setattr(Path, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    assert manager.save() is False

    output = capsys.readouterr().out
    assert "sk-secret" not in output
    assert "raw-token" not in output
    assert "hunter2" not in output
    assert "api_key=***" in output
    assert "token=***" in output
    assert "password=***" in output
