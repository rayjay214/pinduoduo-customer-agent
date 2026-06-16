from Agent.CustomerAgent.custom import knowledge_action_router
from Agent.CustomerAgent.custom import prompt_rules
from Agent.CustomerAgent.custom.knowledge_action_router import (
    SAFE_TRANSFER_REPLY,
    sanitize_customer_service_text,
    sanitize_final_reply,
)


def test_sanitize_final_reply_preserves_transfer_reply():
    reply = "亲，已转人工为您处理，请稍等。"

    assert sanitize_final_reply(reply) == reply


def test_sanitize_final_reply_maps_direct_transfer_action():
    assert sanitize_final_reply("转人工") == SAFE_TRANSFER_REPLY


def test_sanitize_final_reply_maps_transfer_action_instruction():
    assert sanitize_final_reply("需要转人工处理") == SAFE_TRANSFER_REPLY
    assert sanitize_final_reply("建议联系人工客服") == SAFE_TRANSFER_REPLY


def test_sanitize_final_reply_removes_internal_parenthetical_hint_only():
    reply = "亲，这个情况需要帮您核实（建议转人工处理），请稍等。"

    assert sanitize_final_reply(reply) == "亲，这个情况需要帮您核实，请稍等。"


def test_sanitize_customer_service_text_still_removes_internal_action_terms():
    assert sanitize_customer_service_text("需要转人工处理") == "亲，已转人工为您处理，请稍等。"


def test_sanitize_customer_service_text_keeps_non_transfer_answer():
    assert sanitize_customer_service_text("无需转人工，按页面提示处理即可。") == "无需转人工，按页面提示处理即可。"


def test_sanitize_customer_service_text_keeps_transfer_status_sentence():
    reply = "已经为您转人工了，请稍等。"

    assert sanitize_customer_service_text(reply) == reply


def test_sanitize_final_reply_preserves_real_battery_capacity():
    assert sanitize_final_reply("这款电池容量是10000mAh。") == "这款电池容量是10000mAh。"


def test_sanitize_final_reply_does_not_rewrite_version_name_by_default(monkeypatch):
    monkeypatch.setattr(
        prompt_rules,
        "get_config",
        lambda key, default=None: default,
    )

    assert sanitize_final_reply("VERSION_BASIC是10000毫安。") == "VERSION_BASIC是10000毫安。"


def test_sanitize_final_reply_rewrites_configured_version_name(monkeypatch):
    monkeypatch.setattr(
        prompt_rules,
        "get_config",
        lambda key, default=None: ["VIP版"] if key == "agent.version_name_tokens" else default,
    )

    assert sanitize_final_reply("VIP版是5000毫安。") == "具体容量以页面当前规格标注为准"


def test_sanitize_final_reply_does_not_rewrite_unconfigured_version_name(monkeypatch):
    monkeypatch.setattr(
        prompt_rules,
        "get_config",
        lambda key, default=None: ["VIP版"] if key == "agent.version_name_tokens" else default,
    )

    assert sanitize_final_reply("VERSION_BASIC是10000毫安。") == "VERSION_BASIC是10000毫安。"


def test_sanitize_final_reply_can_disable_version_name_rewrite(monkeypatch):
    monkeypatch.setattr(
        prompt_rules,
        "get_config",
        lambda key, default=None: [] if key == "agent.version_name_tokens" else default,
    )

    assert sanitize_final_reply("VERSION_BASIC是10000毫安。") == "VERSION_BASIC是10000毫安。"


def test_sanitize_final_reply_uses_default_reply_replacements(monkeypatch):
    monkeypatch.setattr(
        knowledge_action_router,
        "get_config",
        lambda key, default=None: default,
    )

    assert sanitize_final_reply("本店没有运费险。") == "本店没有退货包运费服务。"


def test_sanitize_final_reply_can_disable_reply_replacements(monkeypatch):
    monkeypatch.setattr(
        knowledge_action_router,
        "get_config",
        lambda key, default=None: {} if key == "reply_sanitizer.replacements" else default,
    )

    assert sanitize_final_reply("本店没有运费险。") == "本店没有运费险。"


def test_sanitize_final_reply_can_override_reply_replacements(monkeypatch):
    monkeypatch.setattr(
        knowledge_action_router,
        "get_config",
        lambda key, default=None: {"运费险": "运费服务"} if key == "reply_sanitizer.replacements" else default,
    )

    assert sanitize_final_reply("本店没有运费险。") == "本店没有运费服务。"


def test_sanitize_final_reply_removes_default_internal_sentence_terms(monkeypatch):
    monkeypatch.setattr(
        knowledge_action_router,
        "get_config",
        lambda key, default=None: default,
    )

    assert sanitize_final_reply("知识库未提供明确数据。亲，这边帮您确认。") == "亲，这边帮您确认。"


def test_sanitize_final_reply_keeps_safe_clause_after_internal_term(monkeypatch):
    monkeypatch.setattr(
        knowledge_action_router,
        "get_config",
        lambda key, default=None: default,
    )

    assert sanitize_final_reply("知识库未提供明确数据，亲，这边帮您确认。") == "亲，这边帮您确认。"


def test_sanitize_final_reply_preserves_customer_visible_system_judgment(monkeypatch):
    monkeypatch.setattr(
        knowledge_action_router,
        "get_config",
        lambda key, default=None: default,
    )

    reply = "如果您觉得平台系统判断有误，可以按售后页面提示提交申诉。"

    assert sanitize_final_reply(reply) == reply


def test_sanitize_final_reply_can_disable_internal_sentence_terms(monkeypatch):
    monkeypatch.setattr(
        knowledge_action_router,
        "get_config",
        lambda key, default=None: []
        if key == "reply_sanitizer.internal_sentence_terms"
        else default,
    )

    assert sanitize_final_reply("知识库未提供明确数据。亲，这边帮您确认。") == "知识库未提供明确数据。亲，这边帮您确认。"


def test_sanitize_final_reply_can_override_internal_sentence_terms(monkeypatch):
    monkeypatch.setattr(
        knowledge_action_router,
        "get_config",
        lambda key, default=None: ["内部规则"]
        if key == "reply_sanitizer.internal_sentence_terms"
        else default,
    )

    assert sanitize_final_reply("内部规则要求转人工。知识库未提供明确数据。亲，这边帮您确认。") == "知识库未提供明确数据。亲，这边帮您确认。"
