from Agent.CustomerAgent.custom import customer_agent
from Agent.CustomerAgent.custom.customer_agent import CustomerAgent


def _build_messages():
    return [{"role": "system", "content": "base prompt"}]


def test_unreceived_query_does_not_inject_received_constraint():
    agent = CustomerAgent()
    messages = _build_messages()

    agent._append_order_hard_constraints(
        messages=messages,
        customer_scene="aftersale",
        dependencies={},
        session_id="session",
        query="我还没收到货，怎么处理",
    )

    assert "当前客户已收到商品" not in messages[0]["content"]


def test_signed_order_injects_received_constraint():
    agent = CustomerAgent()
    messages = _build_messages()

    agent._append_order_hard_constraints(
        messages=messages,
        customer_scene="aftersale",
        dependencies={"order_shipping_status": "signed"},
        session_id="session",
        query="这个有问题",
    )

    assert "当前客户已收到商品" in messages[0]["content"]


def test_order_hard_constraint_fault_examples_can_be_configured(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: ["缩水", "起球"]
        if key == "agent.order_fault_examples"
        else default,
    )

    agent = CustomerAgent()
    messages = _build_messages()

    agent._append_order_hard_constraints(
        messages=messages,
        customer_scene="aftersale",
        dependencies={"order_shipping_status": "signed"},
        session_id="session",
        query="这个有问题",
    )

    assert "例如缩水、起球" in messages[0]["content"]
    assert "噪音、异响" not in messages[0]["content"]


def test_order_constraint_inserts_system_when_first_message_malformed():
    agent = CustomerAgent()
    messages = ["bad-message", {"role": "user", "content": "声音大"}]

    agent._append_order_hard_constraints(
        messages=messages,
        customer_scene="aftersale",
        dependencies={"order_shipping_status": "signed"},
        session_id="session",
        query="声音大",
    )

    assert messages[0]["role"] == "system"
    assert "当前客户已收到商品" in messages[0]["content"]
    assert messages[1] == "bad-message"


def test_signed_order_trace_keywords_keep_default(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: default,
    )

    assert CustomerAgent._is_order_signed({"order_latest_trace": "快件已签收，签收人是本人"})


def test_signed_order_trace_keywords_can_be_configured(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: ["门店自提完成"]
        if key == "pinduoduo.order.signed_trace_keywords"
        else default,
    )

    assert CustomerAgent._is_order_signed({"order_latest_trace": "门店自提完成"})
    assert not CustomerAgent._is_order_signed({"order_latest_trace": "快件已签收"})


def test_signed_order_trace_keywords_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: []
        if key == "pinduoduo.order.signed_trace_keywords"
        else default,
    )

    assert not CustomerAgent._is_order_signed({"order_latest_trace": "快件已签收"})
    assert CustomerAgent._is_order_signed({"order_shipping_status": "signed"})


def test_explicit_usage_feedback_injects_received_constraint():
    agent = CustomerAgent()
    messages = _build_messages()

    agent._append_order_hard_constraints(
        messages=messages,
        customer_scene="aftersale",
        dependencies={},
        session_id="session",
        query="风扇正在用，声音有点大",
    )

    assert "当前客户已收到商品" in messages[0]["content"]


def test_customer_service_dispute_does_not_count_as_usage_feedback():
    agent = CustomerAgent()
    messages = _build_messages()

    agent._append_order_hard_constraints(
        messages=messages,
        customer_scene="aftersale",
        dependencies={},
        session_id="session",
        query="我和客服吵起来了，怎么投诉",
    )

    assert CustomerAgent._has_usage_feedback("我和客服吵起来了，怎么投诉") is False
    assert "当前客户已收到商品" not in messages[0]["content"]


def test_short_feedback_keyword_requires_product_or_fault_context(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: ["吵"] if key == "agent.usage_feedback_keywords" else default,
    )

    assert CustomerAgent._has_usage_feedback("这个风扇很吵，声音大") is True
    assert CustomerAgent._has_usage_feedback("我和客服吵起来了，怎么投诉") is False


def test_usage_feedback_keywords_can_be_configured(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: ["穿了"]
        if key == "agent.usage_feedback_keywords"
        else default,
    )

    assert CustomerAgent._has_usage_feedback("穿了两天就起球")
    assert not CustomerAgent._has_usage_feedback("风扇正在用")


def test_usage_feedback_keywords_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: []
        if key == "agent.usage_feedback_keywords"
        else default,
    )

    assert not CustomerAgent._has_usage_feedback("风扇正在用，声音有点大")


def test_unreceived_patterns_can_be_configured_and_take_priority(monkeypatch):
    def fake_get_config(key, default=None):
        values = {
            "agent.usage_feedback_keywords": ["拆开了"],
            "agent.unreceived_patterns": ["还没拿到"],
        }
        return values.get(key, default)

    monkeypatch.setattr(customer_agent, "get_config", fake_get_config)

    assert CustomerAgent._has_usage_feedback("拆开了，声音大")
    assert not CustomerAgent._has_usage_feedback("还没拿到，但是页面说拆开了")


def test_unreceived_patterns_can_be_disabled(monkeypatch):
    def fake_get_config(key, default=None):
        values = {
            "agent.usage_feedback_keywords": ["拆开了"],
            "agent.unreceived_patterns": [],
        }
        return values.get(key, default)

    monkeypatch.setattr(customer_agent, "get_config", fake_get_config)

    assert CustomerAgent._has_usage_feedback("还没收到，但是页面说拆开了")


def test_mixed_order_context_does_not_force_aftersale_scene():
    agent = CustomerAgent()

    scene = agent._resolve_customer_scene(
        query="我的订单什么时候到",
        history=[],
        dependencies={"order_scene_hint": "mixed_orders"},
    )

    assert scene == "insale"


def test_scene_without_order_stays_presale_even_with_aftersale_words():
    agent = CustomerAgent()

    scene = agent._resolve_customer_scene(
        query="我要退款，这个坏了",
        history=[],
        dependencies={},
    )

    assert scene == "presale"


def test_unsigned_order_stays_insale_even_with_fault_words():
    agent = CustomerAgent()

    scene = agent._resolve_customer_scene(
        query="这个坏了，我要退货",
        history=[],
        dependencies={
            "order_id": "260615-1",
            "order_scene_hint": "insale",
            "order_shipping_status": "shipping",
            "order_business_status": "已发货待收货",
        },
    )

    assert scene == "insale"


def test_signed_order_resolves_aftersale():
    agent = CustomerAgent()

    scene = agent._resolve_customer_scene(
        query="我的订单什么时候到",
        history=[],
        dependencies={
            "order_id": "260615-2",
            "order_shipping_status": "signed",
        },
    )

    assert scene == "aftersale"


def test_order_resolves_aftersale_after_customer_confirms_received():
    agent = CustomerAgent()

    scene = agent._resolve_customer_scene(
        query="已经收到货了，声音很大",
        history=[],
        dependencies={
            "order_id": "260615-3",
            "order_shipping_status": "shipping",
        },
    )

    assert scene == "aftersale"


def test_fault_words_alone_do_not_confirm_received():
    assert CustomerAgent._has_received_confirmation("这个坏了，我要退货") is False
    assert CustomerAgent._has_received_confirmation("已经收到货了，声音很大") is True


def test_unconfirmed_receipt_constraint_injected_for_unsigned_order_fault():
    agent = CustomerAgent()
    messages = _build_messages()

    agent._append_unconfirmed_receipt_constraint(
        messages=messages,
        customer_scene="insale",
        dependencies={
            "order_id": "260615-4",
            "order_shipping_status": "shipping",
        },
        session_id="session",
        query="这个坏了，我要退货",
    )

    assert "当前订单状态和最新物流尚未确认客户已收到商品" in messages[0]["content"]
    assert "先简短确认客户是否已经收到商品" in messages[0]["content"]


def test_unconfirmed_receipt_constraint_skips_signed_or_confirmed_received():
    agent = CustomerAgent()
    signed_messages = _build_messages()
    confirmed_messages = _build_messages()

    agent._append_unconfirmed_receipt_constraint(
        messages=signed_messages,
        customer_scene="insale",
        dependencies={
            "order_id": "260615-5",
            "order_shipping_status": "signed",
        },
        session_id="session",
        query="这个坏了，我要退货",
    )
    agent._append_unconfirmed_receipt_constraint(
        messages=confirmed_messages,
        customer_scene="insale",
        dependencies={
            "order_id": "260615-6",
            "order_shipping_status": "shipping",
        },
        session_id="session",
        query="已经收到货了，这个坏了",
    )

    assert "未确认收货约束" not in signed_messages[0]["content"]
    assert "未确认收货约束" not in confirmed_messages[0]["content"]


def test_night_mode_constraint_uses_configured_time_text(monkeypatch):
    agent = CustomerAgent()
    messages = _build_messages()

    monkeypatch.setattr(
        "Agent.CustomerAgent.custom.customer_agent.is_night_mode",
        lambda: True,
    )
    monkeypatch.setattr(
        "Agent.CustomerAgent.custom.customer_agent.get_night_mode_prompt_values",
        lambda: {"range_text": "22:30-09:30", "resume_text": "09:30"},
    )

    agent._append_night_mode_constraint(messages)

    assert "22:30-09:30" in messages[0]["content"]
    assert "09:30后联系" in messages[0]["content"]
    assert "23:00-08:00" not in messages[0]["content"]
    assert "早上8点后" not in messages[0]["content"]


def test_night_mode_fault_examples_can_be_configured(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "get_night_mode_prompt_values",
        lambda: {"range_text": "22:30-09:30", "resume_text": "09:30"},
    )
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: ["缩水", "起球"]
        if key == "agent.night_mode_fault_examples"
        else default,
    )

    constraint = CustomerAgent._night_mode_fault_constraint()

    assert "缩水、起球" in constraint
    assert "噪音大" not in constraint


def test_daytime_night_mode_leak_markers_keep_default(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "is_night_mode",
        lambda: False,
    )
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: default,
    )
    agent = CustomerAgent()

    reply = agent._sanitize_daytime_night_mode_reply(
        "亲，当前是夜间时段，高级客服已下班。这个问题我先帮您看。",
        [],
    )

    assert reply == "这个问题我先帮您看"


def test_daytime_night_mode_cleanup_does_not_log_reply_content(monkeypatch):
    messages = []

    class FakeLogger:
        def info(self, _message):
            pass

        def warning(self, message):
            messages.append(str(message))

    monkeypatch.setattr(customer_agent, "logger", FakeLogger())
    monkeypatch.setattr(customer_agent, "is_night_mode", lambda: False)
    monkeypatch.setattr(customer_agent, "get_config", lambda key, default=None: default)
    agent = CustomerAgent()

    reply = agent._sanitize_daytime_night_mode_reply(
        "亲，当前是夜间时段 token=secret-token，高级客服已下班。这个问题我先帮您看。",
        [],
    )

    assert reply == "这个问题我先帮您看"
    joined = "\n".join(messages)
    assert "secret-token" not in joined
    assert "token=***" not in joined
    assert "reply_chars=" in joined


def test_turn_context_log_does_not_log_customer_text(monkeypatch):
    messages = []

    class FakeLogger:
        def info(self, message):
            messages.append(str(message))

    monkeypatch.setattr(customer_agent, "logger", FakeLogger())
    tc = customer_agent.parse_turn_context("内容：token=secret-token\n订单卡片：订单号：260511-12345678")

    CustomerAgent._log_turn_context("session-1", tc)

    joined = "\n".join(messages)
    assert "secret-token" not in joined
    assert "260511-12345678" not in joined
    assert "customer_text_chars=" in joined
    assert "has_order_sn=True" in joined


def test_daytime_night_mode_leak_markers_can_be_configured(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "is_night_mode",
        lambda: False,
    )
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: ["值班已结束"]
        if key == "agent.daytime_night_mode_leak_markers"
        else default,
    )
    agent = CustomerAgent()

    custom_reply = agent._sanitize_daytime_night_mode_reply(
        "亲，值班已结束，稍后处理。白天正常回复。",
        [],
    )
    default_marker_reply = agent._sanitize_daytime_night_mode_reply(
        "亲，当前是夜间时段，高级客服已下班。",
        [],
    )

    assert custom_reply == "白天正常回复"
    assert default_marker_reply == "亲，当前是夜间时段，高级客服已下班。"


def test_daytime_night_mode_leak_markers_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "is_night_mode",
        lambda: False,
    )
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: []
        if key == "agent.daytime_night_mode_leak_markers"
        else default,
    )
    agent = CustomerAgent()

    reply = agent._sanitize_daytime_night_mode_reply(
        "亲，当前是夜间时段，高级客服已下班。这个问题我先帮您看。",
        [],
    )

    assert reply == "亲，当前是夜间时段，高级客服已下班。这个问题我先帮您看。"


def test_scene_prompt_files_keep_default_paths(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: default,
    )

    paths = CustomerAgent._scene_prompt_files()

    assert paths["presale"].endswith("presale_prompt.txt")
    assert paths["insale"].endswith("insale_prompt.txt")
    assert paths["aftersale"].endswith("aftersale_prompt.txt")


def test_scene_prompt_files_can_be_configured(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: {
            "售前": "runtime/prompts/shop-a-presale.txt",
            "aftersale": "runtime/prompts/shop-a-aftersale.txt",
            "unknown": "runtime/prompts/ignored.txt",
        }
        if key == "agent.scene_prompt_files"
        else default,
    )

    paths = CustomerAgent._scene_prompt_files()

    assert paths["presale"] == "runtime/prompts/shop-a-presale.txt"
    assert paths["aftersale"] == "runtime/prompts/shop-a-aftersale.txt"
    assert "unknown" not in paths


def test_scene_prompt_cache_reloads_when_file_changes(monkeypatch, tmp_path):
    prompt_path = tmp_path / "presale.txt"
    prompt_path.write_text("old prompt", encoding="utf-8")

    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: {"presale": "runtime/prompts/presale.txt"}
        if key == "agent.scene_prompt_files"
        else default,
    )
    monkeypatch.setattr(
        customer_agent,
        "get_resource_path",
        lambda relative_path: prompt_path,
    )

    agent = CustomerAgent()

    assert agent._load_scene_prompt("presale") == "old prompt"

    prompt_path.write_text("new prompt with different size", encoding="utf-8")

    assert agent._load_scene_prompt("presale") == "new prompt with different size"


def test_scene_prompt_loads_legacy_file_when_default_file_is_missing(monkeypatch, tmp_path):
    legacy_prompt = tmp_path / "legacy.txt"
    legacy_prompt.write_text("legacy prompt", encoding="utf-8")

    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: default,
    )

    def fake_resource_path(relative_path):
        if str(relative_path).endswith("family_a_售前场景prompt_待审.txt"):
            return legacy_prompt
        return tmp_path / "missing.txt"

    monkeypatch.setattr(customer_agent, "get_resource_path", fake_resource_path)

    agent = CustomerAgent()

    assert agent._load_scene_prompt("presale") == "legacy prompt"


def test_scene_prompt_cache_reloads_when_configured_path_changes(monkeypatch, tmp_path):
    first_prompt = tmp_path / "first.txt"
    second_prompt = tmp_path / "second.txt"
    first_prompt.write_text("first prompt", encoding="utf-8")
    second_prompt.write_text("second prompt", encoding="utf-8")

    configured_path = {"value": "runtime/prompts/first.txt"}

    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: {"presale": configured_path["value"]}
        if key == "agent.scene_prompt_files"
        else default,
    )
    monkeypatch.setattr(
        customer_agent,
        "get_resource_path",
        lambda relative_path: second_prompt
        if str(relative_path).endswith("second.txt")
        else first_prompt,
    )

    agent = CustomerAgent()

    assert agent._load_scene_prompt("presale") == "first prompt"

    configured_path["value"] = "runtime/prompts/second.txt"

    assert agent._load_scene_prompt("presale") == "second prompt"


def test_scene_prompt_load_failure_masks_sensitive_log(monkeypatch):
    messages = []

    class FakeLogger:
        def info(self, _message):
            pass

        def warning(self, message):
            messages.append(message)

    monkeypatch.setattr(customer_agent, "logger", FakeLogger())
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: {"presale": "runtime/prompts/presale.txt"}
        if key == "agent.scene_prompt_files"
        else default,
    )

    def fail_resource_path(_relative_path):
        raise RuntimeError("api_key=secret-token")

    monkeypatch.setattr(customer_agent, "get_resource_path", fail_resource_path)

    agent = CustomerAgent()

    assert agent._load_scene_prompt("presale") == ""
    joined = "\n".join(messages)
    assert "secret-token" not in joined
    assert "api_key=***" in joined


def test_high_risk_aftersale_transfer_phrases_are_empty_by_default(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: default,
    )

    assert not CustomerAgent._is_high_risk_aftersale_transfer_issue("这个怎么没有充电口")


def test_high_risk_aftersale_transfer_phrases_can_be_configured(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: ["完全不通电", "按键 没反应"]
        if key == "agent.high_risk_aftersale_transfer_phrases"
        else default,
    )

    assert CustomerAgent._is_high_risk_aftersale_transfer_issue("收到后完全不通电")
    assert CustomerAgent._is_high_risk_aftersale_transfer_issue("这个按键没反应")
    assert not CustomerAgent._is_high_risk_aftersale_transfer_issue("这个怎么没有充电口")


def test_high_risk_aftersale_transfer_phrases_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: []
        if key == "agent.high_risk_aftersale_transfer_phrases"
        else default,
    )

    assert not CustomerAgent._is_high_risk_aftersale_transfer_issue("这个怎么没有充电口")


def test_missing_goods_parameter_keywords_keep_default(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: default,
    )
    agent = CustomerAgent()
    messages = _build_messages()

    agent._append_missing_goods_knowledge_constraint(
        messages,
        query="这款尺寸多大",
        dependencies={},
        session_id="session",
    )

    assert any("当前会话没有识别到 goods_id" in msg["content"] for msg in messages)


def test_missing_goods_parameter_keywords_do_not_block_default_shipping_question(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: default,
    )
    agent = CustomerAgent()
    messages = _build_messages()

    agent._append_missing_goods_knowledge_constraint(
        messages,
        query="什么快递",
        dependencies={},
        session_id="session",
    )

    assert not any("当前会话没有识别到 goods_id" in msg["content"] for msg in messages)


def test_missing_goods_parameter_keywords_can_be_configured(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: ["保质期"]
        if key == "agent.missing_goods_parameter_keywords"
        else default,
    )
    agent = CustomerAgent()
    custom_messages = _build_messages()
    default_messages = _build_messages()

    agent._append_missing_goods_knowledge_constraint(
        custom_messages,
        query="这个保质期多久",
        dependencies={},
        session_id="session",
    )
    agent._append_missing_goods_knowledge_constraint(
        default_messages,
        query="这款尺寸多大",
        dependencies={},
        session_id="session",
    )

    assert any("当前会话没有识别到 goods_id" in msg["content"] for msg in custom_messages)
    assert not any("当前会话没有识别到 goods_id" in msg["content"] for msg in default_messages)


def test_missing_goods_parameter_defaults_do_not_include_category_specific_terms(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: default,
    )
    agent = CustomerAgent()
    messages = _build_messages()

    agent._append_missing_goods_knowledge_constraint(
        messages,
        query="这款续航多久",
        dependencies={},
        session_id="session",
    )

    assert not any("当前会话没有识别到 goods_id" in msg["content"] for msg in messages)


def test_missing_goods_constraint_topics_can_be_configured(monkeypatch):
    def fake_get_config(key, default=None):
        values = {
            "agent.missing_goods_parameter_keywords": ["保质期"],
            "agent.missing_goods_parameter_topics": ["尺码", "材质", "洗涤方式"],
            "agent.missing_goods_unverified_fact_examples": ["尺码", "面料成分", "保质期"],
        }
        return values.get(key, default)

    monkeypatch.setattr(customer_agent, "get_config", fake_get_config)
    agent = CustomerAgent()
    messages = _build_messages()

    agent._append_missing_goods_knowledge_constraint(
        messages,
        query="这个保质期多久",
        dependencies={},
        session_id="session",
    )

    content = messages[-1]["content"]
    assert "尺码、材质、洗涤方式" in content
    assert "尺码、面料成分、保质期" in content
    assert "续航、电池容量、档位" not in content
    assert "小时数、毫安数、档位数" not in content


def test_missing_goods_parameter_keywords_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        customer_agent,
        "get_config",
        lambda key, default=None: []
        if key == "agent.missing_goods_parameter_keywords"
        else default,
    )
    agent = CustomerAgent()
    messages = _build_messages()

    agent._append_missing_goods_knowledge_constraint(
        messages,
        query="这款尺寸多大",
        dependencies={},
        session_id="session",
    )

    assert not any("当前会话没有识别到 goods_id" in msg["content"] for msg in messages)


def test_missing_goods_constraint_log_does_not_include_raw_query(monkeypatch):
    logs = []

    class FakeLogger:
        def info(self, message):
            logs.append(str(message))

    monkeypatch.setattr(customer_agent, "logger", FakeLogger())
    agent = CustomerAgent()
    messages = _build_messages()

    agent._append_missing_goods_knowledge_constraint(
        messages,
        query="这款尺寸多大 token=secret-token",
        dependencies={"shop_id": "shop-1"},
        session_id="session",
    )

    joined = "\n".join(logs)
    assert "secret-token" not in joined
    assert "token=***" not in joined
    assert "query_chars=" in joined


def test_scene_resolution_log_does_not_include_raw_query(monkeypatch):
    logs = []

    class FakeLogger:
        def info(self, message):
            logs.append(str(message))

    monkeypatch.setattr(customer_agent, "logger", FakeLogger())

    CustomerAgent._log_scene_resolution(
        session_id="session",
        customer_scene="presale",
        dependencies={"shop_id": "shop-1", "goods_id": 123, "context_type": "TEXT"},
        query="客户问题 token=secret-token",
    )

    joined = "\n".join(logs)
    assert "secret-token" not in joined
    assert "token=***" not in joined
    assert "query_chars=" in joined


def test_pre_retrieved_knowledge_skips_malformed_results(monkeypatch):
    class FakeKnowledgeService:
        def search_scene_knowledge(self, **_kwargs):
            return [
                "bad",
                {
                    "section_title": "参数",
                    "sub_intent": "续航",
                    "answer": "以页面规格为准",
                    "score": 10,
                    "match_type": "keyword",
                },
            ]

    monkeypatch.setattr(customer_agent, "KnowledgeService", FakeKnowledgeService)
    agent = CustomerAgent()
    messages = _build_messages()

    agent._inject_pre_retrieved_knowledge(
        messages,
        query="续航多久",
        dependencies={"shop_id": "shop-1", "goods_id": 123},
        customer_scene="presale",
    )

    assert "【本轮预检索知识】" in messages[0]["content"]
    assert "以页面规格为准" in messages[0]["content"]


def test_pre_retrieved_knowledge_injects_candidate_selection_context(monkeypatch):
    class FakeKnowledgeService:
        def search_scene_knowledge(self, **_kwargs):
            return [
                {
                    "section_title": "价格优惠",
                    "sub_intent": "价格",
                    "answer": "页面价格以当前优惠为准",
                    "score": 120,
                    "match_type": "keyword",
                },
                {
                    "section_title": "风力参数",
                    "sub_intent": "风力",
                    "answer": "风力以页面参数为准",
                    "score": 90,
                    "match_type": "keyword",
                },
            ]

    monkeypatch.setattr(customer_agent, "KnowledgeService", FakeKnowledgeService)
    agent = CustomerAgent()
    messages = _build_messages()

    agent._inject_pre_retrieved_knowledge(
        messages,
        query="客户消息：风力大吗\n商品卡片：商品：FAMILY_A，价格：999，商品ID：123456",
        dependencies={
            "shop_id": "shop-1",
            "goods_id": 123456,
            "order_shipping_status": "shipping",
        },
        customer_scene="presale",
    )

    content = messages[0]["content"]
    assert "客户当前问题：风力大吗" in content
    assert "当前场景：presale" in content
    assert "先判断客户当前问题主题" in content
    assert "不要只因为某条候选排序靠前" in content
    assert "价格优惠" in content
    assert "风力参数" in content


def test_pre_retrieved_knowledge_uses_recent_customer_context_for_short_followup(monkeypatch):
    captured = {}

    class FakeKnowledgeService:
        def search_scene_knowledge(self, **kwargs):
            captured.update(kwargs)
            return [
                {
                    "section_title": "续航充电参数",
                    "sub_intent": "电池容量",
                    "answer": "电池容量以页面规格为准",
                },
            ]

    monkeypatch.setattr(customer_agent, "KnowledgeService", FakeKnowledgeService)
    agent = CustomerAgent()
    messages = _build_messages()
    history = [
        {"role": "user", "content": "客户消息：这个风扇充一次能用多久"},
        {"role": "assistant", "content": "不同档位续航不同"},
    ]

    agent._inject_pre_retrieved_knowledge(
        messages,
        query="客户消息：多大电",
        dependencies={"shop_id": "shop-1", "goods_id": 123},
        customer_scene="presale",
        history=history,
    )

    assert captured["query"].startswith("多大电")
    assert "充电" in captured["query"] or "续航" in captured["query"]
    assert "【本轮预检索知识】" in messages[0]["content"]
    assert "上下文检索问题：" in messages[0]["content"]


def test_pre_retrieved_knowledge_does_not_pollute_clear_query_with_old_topic(monkeypatch):
    captured = {}

    class FakeKnowledgeService:
        def search_scene_knowledge(self, **kwargs):
            captured.update(kwargs)
            return [
                {
                    "section_title": "续航充电参数",
                    "sub_intent": "电池容量",
                    "answer": "电池容量以页面规格为准",
                },
            ]

    monkeypatch.setattr(customer_agent, "KnowledgeService", FakeKnowledgeService)
    agent = CustomerAgent()
    messages = _build_messages()
    history = [
        {"role": "user", "content": "客户消息：价格还能优惠吗"},
        {"role": "assistant", "content": "页面价格已经是优惠价"},
    ]

    agent._inject_pre_retrieved_knowledge(
        messages,
        query="客户消息：多大电",
        dependencies={"shop_id": "shop-1", "goods_id": 123},
        customer_scene="presale",
        history=history,
    )

    assert captured["query"] == "多大电"
    assert "价格" not in captured["query"]
    assert "优惠" not in captured["query"]
    assert "上下文检索问题：" not in messages[0]["content"]


def test_pre_retrieved_knowledge_injects_candidate_metadata(monkeypatch):
    class FakeKnowledgeService:
        def search_scene_knowledge(self, **_kwargs):
            return [
                {
                    "section_title": "续航参数",
                    "sub_intent": "续航时长",
                    "answer": "最高档约2小时",
                    "tags": "parameter_type:duration,unit:hour",
                    "source_type": "product",
                    "score": 180,
                    "match_type": "keyword",
                },
            ]

    monkeypatch.setattr(customer_agent, "KnowledgeService", FakeKnowledgeService)
    agent = CustomerAgent()
    messages = _build_messages()

    agent._inject_pre_retrieved_knowledge(
        messages,
        query="续航多久",
        dependencies={"shop_id": "shop-1", "goods_id": 123},
        customer_scene="presale",
    )

    content = messages[0]["content"]
    assert "标签：parameter_type:duration,unit:hour" in content
    assert "来源：product" in content
    assert "匹配：keyword" in content


def test_pre_retrieved_knowledge_inserts_system_when_first_message_malformed(monkeypatch):
    class FakeKnowledgeService:
        def search_scene_knowledge(self, **_kwargs):
            return [
                {
                    "section_title": "参数",
                    "sub_intent": "续航",
                    "answer": "以页面规格为准",
                },
            ]

    monkeypatch.setattr(customer_agent, "KnowledgeService", FakeKnowledgeService)
    agent = CustomerAgent()
    messages = ["bad-message", {"role": "user", "content": "续航多久"}]

    agent._inject_pre_retrieved_knowledge(
        messages,
        query="续航多久",
        dependencies={"shop_id": "shop-1", "goods_id": 123},
        customer_scene="presale",
    )

    assert messages[0]["role"] == "system"
    assert "【本轮预检索知识】" in messages[0]["content"]
    assert messages[1] == "bad-message"


def test_pre_retrieved_knowledge_logs_query_length_not_raw_query(monkeypatch):
    logs = []

    class FakeLogger:
        def info(self, message):
            logs.append(str(message))

        def warning(self, message):
            logs.append(str(message))

        def debug(self, message):
            logs.append(str(message))

    class FakeKnowledgeService:
        def search_scene_knowledge(self, **_kwargs):
            return []

    monkeypatch.setattr(customer_agent, "logger", FakeLogger())
    monkeypatch.setattr(customer_agent, "KnowledgeService", FakeKnowledgeService)
    agent = CustomerAgent()
    messages = _build_messages()

    agent._inject_pre_retrieved_knowledge(
        messages,
        query="客户问题 token=secret-token",
        dependencies={"shop_id": "shop-1", "goods_id": 123},
        customer_scene="presale",
    )

    joined = "\n".join(logs)
    assert "secret-token" not in joined
    assert "token=***" not in joined
    assert "query_chars=" in joined
