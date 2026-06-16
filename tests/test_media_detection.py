from Agent.CustomerAgent.custom import prompt_rules
from Agent.CustomerAgent.custom.customer_agent import CustomerAgent
from Agent.CustomerAgent.custom.media_detection import infer_media_type_from_url, normalize_media_url
from Agent.CustomerAgent.custom import message_builder as message_builder_module
from Agent.CustomerAgent.custom.message_builder import MessageBuilder
from bridge.context import ChannelType, Context, ContextType


def test_media_type_does_not_treat_arbitrary_video_word_in_url_as_video():
    context = Context(
        type=ContextType.TEXT,
        content="https://example.com/help/product-video-guide",
    )

    assert MessageBuilder._extract_media_type(context) == ""
    assert CustomerAgent._has_media_input(
        {
            "context_type": "text",
            "media_type": "",
            "media_url": "https://example.com/help/product-video-guide",
        }
    ) is False


def test_media_type_detects_explicit_video_file_url():
    assert infer_media_type_from_url("https://cdn.example.com/upload/a.mp4?token=1") == "video"


def test_media_type_ignores_trailing_sentence_punctuation():
    assert infer_media_type_from_url("https://cdn.example.com/upload/a.mp4，") == "video"
    assert infer_media_type_from_url("https://cdn.example.com/upload/a.jpg。") == "image"
    assert infer_media_type_from_url("https://cdn.example.com/upload/a.png)") == "image"


def test_media_url_normalizes_encoded_trailing_quotes():
    bad_url = "https://cdn.example.com/upload/a.jpeg%22"

    assert normalize_media_url(bad_url) == "https://cdn.example.com/upload/a.jpeg"
    assert infer_media_type_from_url(bad_url) == "image"


def test_media_type_detects_chat_image_path_url():
    assert infer_media_type_from_url("https://cdn.example.com/chat-img/a") == "image"


def test_extract_media_url_from_nested_raw_payload():
    context = Context.create_pinduoduo_context(
        content="[图片消息]",
        user_msg_type=ContextType.IMAGE,
        raw_data={
            "message": {
                "info": {
                    "data": {
                        "image_url": {"url": "https://cdn.example.com/chat-img/nested.jpg"}
                    }
                }
            }
        },
        channel_type=ChannelType.PINDUODUO,
    )

    assert MessageBuilder._extract_media_url(context) == "https://cdn.example.com/chat-img/nested.jpg"
    assert MessageBuilder._extract_media_type(context) == "image"


def test_image_grounding_terms_are_shared_and_configurable(monkeypatch):
    def fake_get_config(key, default=None):
        if key == "agent.image_grounding_forbidden_symbols":
            return ["二维码"]
        if key == "agent.image_grounding_forbidden_functions":
            return ["付款"]
        return default

    monkeypatch.setattr(prompt_rules, "get_config", fake_get_config)
    monkeypatch.setattr(message_builder_module, "get_config", lambda key, default=None: [])

    builder = MessageBuilder()
    messages = [{"role": "system", "content": "base"}]

    CustomerAgent()._append_image_grounding_constraint(
        messages,
        {"context_type": "image", "media_type": "image", "media_url": ""},
        "session",
    )

    assert "二维码" in builder.system_prompt
    assert "付款" in builder.system_prompt
    assert "二维码" in messages[0]["content"]
    assert "付款" in messages[0]["content"]


def test_image_grounding_defaults_do_not_inject_product_specific_terms(monkeypatch):
    monkeypatch.setattr(prompt_rules, "get_config", lambda key, default=None: default)
    monkeypatch.setattr(message_builder_module, "get_config", lambda key, default=None: [])

    builder = MessageBuilder()
    messages = [{"role": "system", "content": "base"}]

    CustomerAgent()._append_image_grounding_constraint(
        messages,
        {"context_type": "image", "media_type": "image", "media_url": ""},
        "session",
    )

    assert "图片里的" not in builder.system_prompt
    assert messages[0]["content"] == "base"
    assert "雪花" not in builder.system_prompt
    assert "制冷" not in builder.system_prompt


def test_version_name_tokens_are_configurable_in_system_prompt(monkeypatch):
    def fake_get_config(key, default=None):
        if key == "agent.version_name_tokens":
            return ["VIP版"]
        return []

    monkeypatch.setattr(prompt_rules, "get_config", fake_get_config)
    monkeypatch.setattr(message_builder_module, "get_config", lambda key, default=None: [])

    builder = MessageBuilder()

    assert "VIP版" in builder.system_prompt
    assert "VERSION_PLUS" not in builder.system_prompt


def test_version_name_tokens_can_be_disabled_in_system_prompt(monkeypatch):
    def fake_get_config(key, default=None):
        if key == "agent.version_name_tokens":
            return []
        return []

    monkeypatch.setattr(prompt_rules, "get_config", fake_get_config)
    monkeypatch.setattr(message_builder_module, "get_config", lambda key, default=None: [])

    builder = MessageBuilder()

    assert "版本名约束" not in builder.system_prompt
    assert "VERSION_PLUS" not in builder.system_prompt


def test_image_grounding_terms_can_be_disabled(monkeypatch):
    def fake_get_config(key, default=None):
        if key in {
            "agent.image_grounding_forbidden_symbols",
            "agent.image_grounding_forbidden_functions",
        }:
            return []
        return []

    monkeypatch.setattr(prompt_rules, "get_config", fake_get_config)
    monkeypatch.setattr(message_builder_module, "get_config", lambda key, default=None: [])

    builder = MessageBuilder()

    assert "图片里的" not in builder.system_prompt
    assert prompt_rules.build_image_grounding_constraint() == ""
