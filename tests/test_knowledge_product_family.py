from types import SimpleNamespace

from sqlalchemy import select, text

from database import knowledge_service
from database.db_manager import DatabaseManager
from database.knowledge_service import KnowledgeService
from database.models import Channel, CustomerServiceKnowledge, KnowledgeMetaEntry, PresaleKnowledge, Shop


def _taxonomy_config(hints=None):
    hints = hints or {}

    def fake_get_config(key, default=None):
        if key in hints:
            return hints[key]
        return default

    return fake_get_config


def test_product_family_default_rules_do_not_embed_shop_specific_matches(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )

    assert KnowledgeService._infer_product_family("静音示例商品 FAMILY_A") == ""
    assert KnowledgeService._infer_product_family("便携风扇 family_c") == ""
    assert KnowledgeService._infer_product_family("120档示例商品升级款") == ""


def test_product_family_can_be_extended_by_config_without_code_change(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: [
            {"family": "family_b", "contains": ["family_b"]},
            {"family": "FAMILY_D", "regex": [r"(?<!\d)FAMILY_D(?!\d)"]},
        ]
        if key == "knowledge.product_family_rules"
        else default,
    )

    assert KnowledgeService._infer_product_family("FAMILY_B 示例商品B") == "family_b"
    assert KnowledgeService._infer_product_family("FAMILY_D示例商品D") == "FAMILY_D"
    assert KnowledgeService._infer_product_family("1FAMILY_D不是独立型号") == ""


def test_product_family_can_configure_legacy_fan_rules_without_code_change(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: [
            {"family": "family_a", "contains": ["family_a"]},
            {"family": "family_c", "contains": ["family_c"]},
            {"family": "120", "contains": ["120档示例商品"]},
        ]
        if key == "knowledge.product_family_rules"
        else default,
    )

    assert KnowledgeService._infer_product_family("静音示例商品 FAMILY_A") == "family_a"
    assert KnowledgeService._infer_product_family("便携风扇 family_c") == "family_c"
    assert KnowledgeService._infer_product_family("120档示例商品升级款") == "120"


def test_product_family_rules_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: [] if key == "knowledge.product_family_rules" else default,
    )

    assert KnowledgeService._infer_product_family("静音示例商品 FAMILY_A") == ""


def test_tag_score_adjustments_are_configurable(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: {
            "curated_scene_kb": {"meta": 40, "customer_service": 25},
            "simple_tag": 9,
            "bad_weight": "x",
        }
        if key == "knowledge.tag_score_adjustments"
        else default,
    )

    assert KnowledgeService._configured_tag_score("curated_scene_kb presale", scope="meta") == 40
    assert KnowledgeService._configured_tag_score("curated_scene_kb presale", scope="customer_service") == 25
    assert KnowledgeService._configured_tag_score("simple_tag") == 9
    assert KnowledgeService._configured_tag_score("bad_weight") == 0
    assert KnowledgeService._configured_tag_score("family_a_scene_kb") == 0


def test_tag_score_adjustments_match_whole_tags_only(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: {"family_a_scene_kb": {"meta": 35}}
        if key == "knowledge.tag_score_adjustments"
        else default,
    )

    assert KnowledgeService._configured_tag_score("not_family_a_scene_kb", scope="meta") == 0
    assert KnowledgeService._configured_tag_score("family_a_scene_kb, presale", scope="meta") == 35


def test_default_tag_score_has_no_shop_specific_weights(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )

    assert KnowledgeService._configured_tag_score("family_a_scene_kb", default=35, scope="meta") == 0
    assert KnowledgeService._configured_tag_score("family_a_scene_kb", default=25, scope="customer_service") == 0


def test_h_model_source_types_no_longer_drive_scoring(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: ["manual_h_model_source"]
        if key == "knowledge.h_model_source_types"
        else default,
    )
    entry = SimpleNamespace(
        section_title="型号参数",
        sub_intent="型号说明",
        answer="",
        aliases="",
        source_type="manual_h_model_source",
        tags="",
    )

    assert KnowledgeService._intent_score_adjustment({"version_name"}, entry, query="2H是什么") == 400


def test_version_name_match_score_uses_generic_tags(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )
    version_entry = SimpleNamespace(
        section_title="型号参数",
        sub_intent="版本名称",
        answer="2H、5H、10H 是商品版本名称",
        aliases="",
        source_type="",
        tags="parameter_type:version_name",
    )
    duration_entry = SimpleNamespace(
        section_title="续航参数",
        sub_intent="续航时间",
        answer="续航约2小时",
        aliases="",
        source_type="",
        tags="parameter_type:duration",
    )

    assert KnowledgeService._query_intent_hints("5H是什么") == {"version_name"}
    assert KnowledgeService._version_name_match_score("5H是什么", version_entry) > 0
    assert KnowledgeService._version_name_match_score("5H是什么", duration_entry) < 0


def test_score_entries_uses_version_name_tags_for_model_tokens(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )
    service = KnowledgeService.__new__(KnowledgeService)
    service.vector_retriever = None
    version_entry = SimpleNamespace(
        id=1,
        priority=0,
        section_title="型号参数",
        sub_intent="版本名称",
        answer="2H、5H、10H 是商品版本名称，不是续航时长",
        aliases="",
        source_type="product",
        tags="parameter_type:version_name",
        goods_id=123,
    )
    duration_entry = SimpleNamespace(
        id=2,
        priority=0,
        section_title="续航参数",
        sub_intent="续航时间",
        answer="续航约2小时",
        aliases="",
        source_type="product",
        tags="parameter_type:duration",
        goods_id=123,
    )

    ranked = service._score_entries([duration_entry, version_entry], query="5H是什么", goods_id=123, scene_key="presale")

    assert ranked[0][0] is version_entry


def test_version_name_query_tokens_follow_agent_config(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: ["VIP版"] if key == "agent.version_name_tokens" else default,
    )

    assert KnowledgeService._version_name_query_tokens() == ("vip版",)


def test_version_name_score_adjustment_uses_configured_tokens(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: ["VIP版"] if key == "agent.version_name_tokens" else default,
    )
    version_entry = SimpleNamespace(
        section_title="版本参数",
        sub_intent="版本区别",
        answer="VIP版是页面规格名称",
        aliases="VIP版",
        source_type="",
    )
    button_entry = SimpleNamespace(
        section_title="按键",
        sub_intent="开关机教程",
        answer="加减按键说明",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment(set(), version_entry, query="VIP版是什么") > 0
    assert KnowledgeService._intent_score_adjustment(set(), button_entry, query="VIP版是什么") < 0
    assert KnowledgeService._intent_score_adjustment(set(), version_entry, query="40000m是什么") == 0


def test_version_name_score_adjustment_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: [] if key == "agent.version_name_tokens" else default,
    )
    version_entry = SimpleNamespace(
        section_title="版本参数",
        sub_intent="版本区别",
        answer="40000m是页面规格名称",
        aliases="40000m",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment(set(), version_entry, query="40000m是什么") == 0


def test_charger_accessory_weights_are_not_default_text_patches(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )
    entry = SimpleNamespace(
        section_title="配件",
        sub_intent="赠品清单",
        answer="包装内含充电线",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment(set(), entry, query="送充电器吗") == 0


def test_intent_score_adjustment_rules_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.intent_score_adjustment_rules"
        else default,
    )
    entry = SimpleNamespace(
        section_title="配件",
        sub_intent="赠品清单",
        answer="包装内含充电线",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment(set(), entry, query="送充电器吗") == 0


def test_intent_score_adjustment_rules_can_be_configured(monkeypatch):
    def fake_get_config(key, default=None):
        if key == "knowledge.intent_score_adjustment_rules":
            return [
                {
                    "query_any": ["缩水"],
                    "section_any": ["面料"],
                    "score": 77,
                }
            ]
        return default

    monkeypatch.setattr(knowledge_service, "get_config", fake_get_config)
    entry = SimpleNamespace(
        section_title="面料",
        sub_intent="洗护",
        answer="纯棉材质",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment(set(), entry, query="洗了缩水") == 77


def test_switch_tutorial_weights_are_not_default_text_patches(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )
    entry = SimpleNamespace(
        section_title="按键",
        sub_intent="开关机教程",
        answer="长按开机",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment(set(), entry, query="续航多久") == 0


def test_intent_score_adjustment_rules_disable_switch_tutorial_demote(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.intent_score_adjustment_rules"
        else default,
    )
    entry = SimpleNamespace(
        section_title="按键",
        sub_intent="开关机教程",
        answer="长按开机",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment(set(), entry, query="续航多久") == 0


def test_arrival_time_weights_are_not_default_text_patches(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )
    arrival_entry = SimpleNamespace(
        section_title="发货物流",
        sub_intent="到货时效",
        answer="一般几天到，具体以物流为准",
        aliases="",
        source_type="",
    )
    refund_entry = SimpleNamespace(
        section_title="退货运费",
        sub_intent="拒收退款",
        answer="退货退款规则",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment({"arrival_time"}, arrival_entry, query="多久到货") == 0
    assert KnowledgeService._intent_score_adjustment({"arrival_time"}, refund_entry, query="多久到货") == 0


def test_intent_score_adjustment_rules_disable_arrival_time_weights(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.intent_score_adjustment_rules"
        else default,
    )
    arrival_entry = SimpleNamespace(
        section_title="发货物流",
        sub_intent="到货时效",
        answer="一般几天到，具体以物流为准",
        aliases="",
        source_type="",
    )
    refund_entry = SimpleNamespace(
        section_title="退货运费",
        sub_intent="拒收退款",
        answer="退货退款规则",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment({"arrival_time"}, arrival_entry, query="多久到货") == 0
    assert KnowledgeService._intent_score_adjustment({"arrival_time"}, refund_entry, query="多久到货") == 0


def test_accessory_weights_are_not_default_text_patches(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )
    accessory_entry = SimpleNamespace(
        section_title="配件赠品",
        sub_intent="挂绳",
        answer="包装内带挂绳",
        aliases="",
        source_type="",
    )
    other_entry = SimpleNamespace(
        section_title="续航参数",
        sub_intent="电池容量",
        answer="续航以实际使用为准",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment({"accessory"}, accessory_entry, query="送挂绳吗") == 0
    assert KnowledgeService._intent_score_adjustment({"accessory"}, other_entry, query="送挂绳吗") == 0


def test_intent_score_adjustment_rules_disable_accessory_weights(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.intent_score_adjustment_rules"
        else default,
    )
    accessory_entry = SimpleNamespace(
        section_title="配件赠品",
        sub_intent="挂绳",
        answer="包装内带挂绳",
        aliases="",
        source_type="",
    )
    other_entry = SimpleNamespace(
        section_title="续航参数",
        sub_intent="电池容量",
        answer="续航以实际使用为准",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment({"accessory"}, accessory_entry, query="送挂绳吗") == 0
    assert KnowledgeService._intent_score_adjustment({"accessory"}, other_entry, query="送挂绳吗") == 0


def test_color_stock_weights_are_not_default_text_patches(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )
    entry = SimpleNamespace(
        section_title="颜色库存",
        sub_intent="颜色可选",
        answer="有黑色和白色",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment({"color_stock"}, entry, query="有黑色吗") == 0


def test_intent_score_adjustment_rules_disable_color_stock_weight(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.intent_score_adjustment_rules"
        else default,
    )
    entry = SimpleNamespace(
        section_title="颜色库存",
        sub_intent="颜色可选",
        answer="有黑色和白色",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment({"color_stock"}, entry, query="有黑色吗") == 0


def test_price_weights_are_not_default_text_patches(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )
    price_entry = SimpleNamespace(
        section_title="价格优惠",
        sub_intent="多少钱",
        answer="页面价格以当前优惠为准",
        aliases="",
        source_type="",
    )
    other_entry = SimpleNamespace(
        section_title="续航参数",
        sub_intent="电池容量",
        answer="续航以实际使用为准",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment({"price"}, price_entry, query="多少钱") == 0
    assert KnowledgeService._intent_score_adjustment({"price"}, other_entry, query="多少钱") == 0


def test_intent_score_adjustment_rules_disable_price_weights(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.intent_score_adjustment_rules"
        else default,
    )
    price_entry = SimpleNamespace(
        section_title="价格优惠",
        sub_intent="多少钱",
        answer="页面价格以当前优惠为准",
        aliases="",
        source_type="",
    )
    other_entry = SimpleNamespace(
        section_title="续航参数",
        sub_intent="电池容量",
        answer="续航以实际使用为准",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment({"price"}, price_entry, query="多少钱") == 0
    assert KnowledgeService._intent_score_adjustment({"price"}, other_entry, query="多少钱") == 0


def test_wind_power_weights_are_not_default_text_patches(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )
    wind_entry = SimpleNamespace(
        section_title="风力参数",
        sub_intent="风速",
        answer="风力强，转速高",
        aliases="",
        source_type="",
    )
    other_entry = SimpleNamespace(
        section_title="续航参数",
        sub_intent="电池容量",
        answer="续航以实际使用为准",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment({"wind_power"}, wind_entry, query="风大吗") == 0
    assert KnowledgeService._intent_score_adjustment({"wind_power"}, other_entry, query="风大吗") == 0


def test_intent_score_adjustment_rules_disable_wind_power_weights(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.intent_score_adjustment_rules"
        else default,
    )
    wind_entry = SimpleNamespace(
        section_title="风力参数",
        sub_intent="风速",
        answer="风力强，转速高",
        aliases="",
        source_type="",
    )
    other_entry = SimpleNamespace(
        section_title="续航参数",
        sub_intent="电池容量",
        answer="续航以实际使用为准",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment({"wind_power"}, wind_entry, query="风大吗") == 0
    assert KnowledgeService._intent_score_adjustment({"wind_power"}, other_entry, query="风大吗") == 0


def test_logistics_delivery_weights_are_not_default_text_patches(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )
    logistics_entry = SimpleNamespace(
        section_title="发货物流",
        sub_intent="快递公司",
        answer="默认快递配送",
        aliases="",
        source_type="",
    )
    other_entry = SimpleNamespace(
        section_title="续航参数",
        sub_intent="电池容量",
        answer="续航以实际使用为准",
        aliases="",
        source_type="",
    )
    accessory_entry = SimpleNamespace(
        section_title="配件赠品",
        sub_intent="充电线",
        answer="包装内含充电线",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment({"logistics_delivery"}, logistics_entry, query="发什么快递") == 0
    assert KnowledgeService._intent_score_adjustment({"logistics_delivery"}, other_entry, query="发什么快递") == 0
    assert KnowledgeService._intent_score_adjustment({"logistics_delivery"}, accessory_entry, query="发什么快递") == 0


def test_intent_score_adjustment_rules_disable_logistics_delivery_weights(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.intent_score_adjustment_rules"
        else default,
    )
    logistics_entry = SimpleNamespace(
        section_title="发货物流",
        sub_intent="快递公司",
        answer="默认快递配送",
        aliases="",
        source_type="",
    )
    other_entry = SimpleNamespace(
        section_title="续航参数",
        sub_intent="电池容量",
        answer="续航以实际使用为准",
        aliases="",
        source_type="",
    )
    accessory_entry = SimpleNamespace(
        section_title="配件赠品",
        sub_intent="充电线",
        answer="包装内含充电线",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment({"logistics_delivery"}, logistics_entry, query="发什么快递") == 0
    assert KnowledgeService._intent_score_adjustment({"logistics_delivery"}, other_entry, query="发什么快递") == 0
    assert KnowledgeService._intent_score_adjustment({"logistics_delivery"}, accessory_entry, query="发什么快递") == 0


def test_battery_size_weights_are_not_default_text_patches(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )
    battery_entry = SimpleNamespace(
        section_title="电池容量",
        sub_intent="毫安",
        answer="电池容量以页面参数为准",
        aliases="",
        source_type="",
    )
    wind_entry = SimpleNamespace(
        section_title="风力参数",
        sub_intent="风速档位",
        answer="风力强，转速高",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment({"battery_size_query"}, battery_entry, query="电池多大") == 0
    assert KnowledgeService._intent_score_adjustment({"battery_size_query"}, wind_entry, query="电池多大") == 0


def test_intent_score_adjustment_rules_disable_battery_size_weights(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.intent_score_adjustment_rules"
        else default,
    )
    battery_entry = SimpleNamespace(
        section_title="电池容量",
        sub_intent="毫安",
        answer="电池容量以页面参数为准",
        aliases="",
        source_type="",
    )
    wind_entry = SimpleNamespace(
        section_title="风力参数",
        sub_intent="风速档位",
        answer="风力强，转速高",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment({"battery_size_query"}, battery_entry, query="电池多大") == 0
    assert KnowledgeService._intent_score_adjustment({"battery_size_query"}, wind_entry, query="电池多大") == 0


def test_battery_capacity_weights_are_not_default_text_patches(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )
    title_entry = SimpleNamespace(
        section_title="电池容量",
        sub_intent="毫安",
        answer="参数以页面为准",
        aliases="",
        source_type="",
    )
    answer_entry = SimpleNamespace(
        section_title="规格参数",
        sub_intent="基础参数",
        answer="电池容量以页面参数为准",
        aliases="",
        source_type="",
    )
    other_entry = SimpleNamespace(
        section_title="规格参数",
        sub_intent="基础参数",
        answer="风力转速和档位说明",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment({"battery_capacity"}, title_entry, query="电池容量") == 0
    assert KnowledgeService._intent_score_adjustment({"battery_capacity"}, answer_entry, query="电池容量") == 0
    assert KnowledgeService._intent_score_adjustment({"battery_capacity"}, other_entry, query="电池容量") == 0


def test_intent_score_adjustment_rules_disable_battery_capacity_weights(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.intent_score_adjustment_rules"
        else default,
    )
    title_entry = SimpleNamespace(
        section_title="电池容量",
        sub_intent="毫安",
        answer="参数以页面为准",
        aliases="",
        source_type="",
    )
    answer_entry = SimpleNamespace(
        section_title="规格参数",
        sub_intent="基础参数",
        answer="电池容量以页面参数为准",
        aliases="",
        source_type="",
    )
    other_entry = SimpleNamespace(
        section_title="规格参数",
        sub_intent="基础参数",
        answer="风力转速和档位说明",
        aliases="",
        source_type="",
    )

    assert KnowledgeService._intent_score_adjustment({"battery_capacity"}, title_entry, query="电池容量") == 0
    assert KnowledgeService._intent_score_adjustment({"battery_capacity"}, answer_entry, query="电池容量") == 0
    assert KnowledgeService._intent_score_adjustment({"battery_capacity"}, other_entry, query="电池容量") == 0


def test_product_parameter_blocks_keep_default_aliases(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )

    blocks = KnowledgeService._product_knowledge_blocks("## 参数\n- 功率：5W\n- 材质：纯棉")

    assert any("功率多少瓦" in block for block in blocks)
    assert any("是什么材质" in block for block in blocks)
    assert not any("续航多久" in block for block in blocks)


def test_product_parameter_blocks_can_be_configured(monkeypatch):
    def fake_get_config(key, default=None):
        values = {
            "knowledge.product_parameter_keywords": ["面料"],
            "knowledge.product_parameter_alias_rules": [
                {"contains_any": ["面料", "材质"], "alias": "问法：是什么面料/材质是什么"},
            ],
        }
        return values.get(key, default)

    monkeypatch.setattr(knowledge_service, "get_config", fake_get_config)

    blocks = KnowledgeService._product_knowledge_blocks("## 参数\n- 面料：纯棉\n- 功率：5W")

    assert len(blocks) == 1
    assert "是什么面料" in blocks[0]
    assert "功率" not in blocks[0]


def test_product_parameter_keywords_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.product_parameter_keywords"
        else default,
    )

    assert KnowledgeService._product_knowledge_blocks("## 参数\n- 功率：5W\n- 续航：最高档2小时") == []


def test_product_parameter_alias_rules_can_be_disabled(monkeypatch):
    def fake_get_config(key, default=None):
        values = {
            "knowledge.product_parameter_keywords": ["功率"],
            "knowledge.product_parameter_alias_rules": [],
        }
        return values.get(key, default)

    monkeypatch.setattr(knowledge_service, "get_config", fake_get_config)

    blocks = KnowledgeService._product_knowledge_blocks("## 参数\n- 功率：5W")

    assert blocks == ["## 参数\n- 功率：5W"]


def test_qualifier_groups_do_not_keep_default_version_terms(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )

    groups = KnowledgeService._qualifier_groups()

    assert groups == ()


def test_qualifier_groups_can_be_configured(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: [["棉质", "纯棉"]]
        if key == "knowledge.qualifier_groups"
        else default,
    )

    assert KnowledgeService._qualifier_groups() == (("棉质", "纯棉"),)


def test_qualifier_groups_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.qualifier_groups"
        else default,
    )

    assert KnowledgeService._qualifier_groups() == ()


def test_search_phrase_candidates_can_be_configured(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: ["保质期"]
        if key == "knowledge.search_phrase_candidates"
        else {}
        if key == "knowledge.search_synonym_expansions"
        else default,
    )

    assert "保质期" in KnowledgeService._search_terms("这个保质期多久")
    assert "充电口" not in KnowledgeService._search_terms("充电口在哪里")


def test_search_phrase_candidates_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.search_phrase_candidates"
        else {}
        if key == "knowledge.search_synonym_expansions"
        else default,
    )

    assert "退货包运费" not in KnowledgeService._search_terms("退货包运费服务")


def test_search_synonym_expansions_can_be_configured(monkeypatch):
    def fake_get_config(key, default=None):
        values = {
            "knowledge.search_phrase_candidates": [],
            "knowledge.search_synonym_expansions": {"缩水": ["面料", "售后"]},
        }
        return values.get(key, default)

    monkeypatch.setattr(knowledge_service, "get_config", fake_get_config)

    terms = KnowledgeService._search_terms("缩水严重")

    assert "面料" in terms
    assert "售后" in terms
    assert "档位" not in KnowledgeService._search_terms("调档")


def test_intent_keywords_keep_default_noise_fault(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )

    assert "noise_fault" in KnowledgeService._query_intent_hints("这个声音很吵")


def test_intent_keywords_can_be_configured(monkeypatch):
    def fake_get_config(key, default=None):
        values = {
            "knowledge.intent_keywords.noise_fault": ["蜂鸣"],
            "knowledge.intent_keywords.accessory": ["肩带"],
        }
        return values.get(key, default)

    monkeypatch.setattr(knowledge_service, "get_config", fake_get_config)

    assert "noise_fault" not in KnowledgeService._query_intent_hints("这个声音很吵")
    assert "noise_fault" in KnowledgeService._query_intent_hints("这个有蜂鸣")
    assert "accessory" in KnowledgeService._classify_query_intent("送肩带吗")


def test_intent_keywords_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.intent_keywords.noise_fault"
        else default,
    )

    assert "noise_fault" not in KnowledgeService._query_intent_hints("这个声音很吵")


def test_query_intent_debug_log_does_not_include_raw_query(monkeypatch):
    messages = []

    class FakeLogger:
        def debug(self, message):
            messages.append(str(message))

    monkeypatch.setattr(knowledge_service, "logger", FakeLogger())

    hints = KnowledgeService._query_intent_hints("客户说 token=secret-token 会很吵")

    assert "noise_fault" in hints
    joined = "\n".join(messages)
    assert "secret-token" not in joined
    assert "token=***" not in joined
    assert "query_chars=" in joined


def test_customer_service_fallback_log_does_not_include_raw_query(monkeypatch, tmp_path):
    db_path = tmp_path / "knowledge.db"
    manager = DatabaseManager(db_path=str(db_path))
    monkeypatch.setattr(knowledge_service, "db_manager", manager)

    messages = []

    class FakeLogger:
        def info(self, message):
            messages.append(str(message))

        def warning(self, _message):
            pass

    class EmptyVectorRetriever:
        def rank(self, **_kwargs):
            return []

    monkeypatch.setattr(knowledge_service, "logger", FakeLogger())

    service = KnowledgeService()
    service.vector_retriever = EmptyVectorRetriever()
    with service.get_session() as session:
        channel = Channel(channel_name="pdd", description="")
        shop = Shop(channel=channel, shop_id="shop-1", shop_name="测试店铺")
        cs = CustomerServiceKnowledge(
            shop=shop,
            title="售后退款",
            content="退款可以联系人工处理",
            tags="售后",
            enabled=True,
        )
        presale_cs = CustomerServiceKnowledge(
            shop=shop,
            title="售前颜色",
            content="颜色以页面选项为准",
            tags="售前",
            enabled=True,
        )
        session.add_all([channel, shop, cs, presale_cs])
        session.flush()
        session.add_all(
            [
                KnowledgeMetaEntry(
                    shop_id=shop.id,
                    source_type="customer_service",
                    source_id=cs.id,
                    scenario="aftersale",
                    sub_intent="退款",
                    aliases="退款",
                    answer="退款可以联系人工处理",
                    enabled=True,
                ),
                KnowledgeMetaEntry(
                    shop_id=shop.id,
                    source_type="customer_service",
                    source_id=presale_cs.id,
                    scenario="presale",
                    sub_intent="颜色",
                    aliases="有什么颜色",
                    answer="颜色以页面选项为准",
                    enabled=True,
                ),
            ]
        )
        session.commit()

    result = service.search_knowledge(
        shop_id="shop-1",
        query="退款 token=secret-token",
        search_scope="customer_service",
        scene="presale",
    )

    assert result["customer_service_knowledge"]
    joined = "\n".join(messages)
    assert "场景客服知识未命中" in joined
    assert "secret-token" not in joined
    assert "token=***" not in joined
    assert "query_chars=" in joined


def test_structured_scenario_rules_can_be_configured(monkeypatch):
    def fake_get_config(key, default=None):
        if key == "knowledge.structured_scenario_rules":
            return {"fabric": ["面料", "纯棉"]}
        if key == "knowledge.structured_scenario_anchors":
            return {"fabric": ["材质", "面料"]}
        return default

    monkeypatch.setattr(knowledge_service, "get_config", fake_get_config)

    assert KnowledgeService._detect_query_scenario("这个是什么面料") == "fabric"
    assert KnowledgeService._detect_query_scenario("这个怎么充电") == ""
    assert KnowledgeService._scenario_anchor_terms("fabric") == ("材质", "面料")


def test_structured_scenario_rules_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: {}
        if key == "knowledge.structured_scenario_rules"
        else default,
    )

    assert KnowledgeService._detect_query_scenario("这个怎么充电") == ""


def test_structured_intent_rules_can_be_configured(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: {
            "fabric_query": {"all": ["面料"], "any": ["什么", "哪种"]},
            "shrink_fault": {"any": ["缩水", "变形"]},
        }
        if key == "knowledge.structured_intent_rules"
        else default,
    )

    assert KnowledgeService._detect_query_intent("这是什么面料") == "fabric_query"
    assert KnowledgeService._detect_query_intent("洗了缩水") == "shrink_fault"
    assert KnowledgeService._detect_query_intent("送充电器吗") == ""


def test_structured_intent_rules_keep_generic_accessory_default(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )

    assert KnowledgeService._detect_query_intent("送配件吗") == "gift_accessory"
    assert KnowledgeService._detect_query_intent("送充电器吗") == ""


def test_structured_intent_rules_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: {}
        if key == "knowledge.structured_intent_rules"
        else default,
    )

    assert KnowledgeService._detect_query_intent("送配件吗") == ""


def test_intent_specific_score_rules_are_empty_by_default(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )

    score = KnowledgeService._intent_specific_match_score(
        query_clean=KnowledgeService._normalize_match_text("发票"),
        aliases_clean="",
        answer_clean=KnowledgeService._normalize_match_text("支持开普通发票"),
        section_clean=KnowledgeService._normalize_match_text("发票开具咨询"),
        tags_clean="",
    )

    assert score == 0


def test_intent_specific_score_rules_can_be_configured(monkeypatch):
    def fake_get_config(key, default=None):
        if key == "knowledge.intent_specific_score_rules":
            return [
                {
                    "query_any": ["缩水"],
                    "knowledge_any": ["面料", "洗护"],
                    "score": 77,
                },
                {
                    "query_exact": ["发票"],
                    "knowledge_any": ["发票开具咨询", "支持开普通发票"],
                    "score": 65,
                }
            ]
        return default

    monkeypatch.setattr(knowledge_service, "get_config", fake_get_config)

    custom_score = KnowledgeService._intent_specific_match_score(
        query_clean=KnowledgeService._normalize_match_text("洗了缩水"),
        aliases_clean="",
        answer_clean=KnowledgeService._normalize_match_text("面料洗护说明"),
        section_clean="",
        tags_clean="",
    )
    old_score = KnowledgeService._intent_specific_match_score(
        query_clean=KnowledgeService._normalize_match_text("发票"),
        aliases_clean="",
        answer_clean=KnowledgeService._normalize_match_text("支持开普通发票"),
        section_clean=KnowledgeService._normalize_match_text("发票开具咨询"),
        tags_clean="",
    )

    assert custom_score == 77
    assert old_score == 65


def test_intent_specific_score_rules_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.intent_specific_score_rules"
        else default,
    )

    score = KnowledgeService._intent_specific_match_score(
        query_clean=KnowledgeService._normalize_match_text("发票"),
        aliases_clean="",
        answer_clean=KnowledgeService._normalize_match_text("支持开普通发票"),
        section_clean=KnowledgeService._normalize_match_text("发票开具咨询"),
        tags_clean="",
    )

    assert score == 0


def test_customer_scene_does_not_use_aftersale_keywords(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )

    assert KnowledgeService.detect_customer_scene("我要退款", default="presale") == "presale"


def test_customer_scene_ignores_configured_keyword_rules(monkeypatch):
    def fake_get_config(key, default=None):
        values = {
            "knowledge.customer_scene_keywords.direct_aftersale": ["需要售后"],
        }
        return values.get(key, default)

    monkeypatch.setattr(knowledge_service, "get_config", fake_get_config)

    assert KnowledgeService.detect_customer_scene("我要退款", default="presale") == "presale"
    assert KnowledgeService.detect_customer_scene("需要售后", default="presale") == "presale"


def test_customer_scene_keywords_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.customer_scene_keywords.direct_aftersale"
        else default,
    )

    assert KnowledgeService.detect_customer_scene("我要退款", default="presale") == "presale"


def test_customer_scene_quality_words_do_not_force_aftersale(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )

    assert KnowledgeService.detect_customer_scene("声音太大", default="presale") == "presale"
    assert KnowledgeService.detect_customer_scene("风力太小", default="presale") == "presale"


def test_customer_scene_presale_quality_question_is_not_aftersale(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: default,
    )

    assert KnowledgeService.detect_customer_scene("声音大吗", default="presale") == "presale"


def test_customer_scene_quality_config_does_not_force_aftersale(monkeypatch):
    def fake_get_config(key, default=None):
        values = {
            "knowledge.customer_scene_keywords.quality_complaint": ["缩水严重"],
        }
        return values.get(key, default)

    monkeypatch.setattr(knowledge_service, "get_config", fake_get_config)

    assert KnowledgeService.detect_customer_scene("风力太小", default="presale") == "presale"
    assert KnowledgeService.detect_customer_scene("缩水严重", default="presale") == "presale"


def test_customer_scene_received_problem_config_does_not_force_aftersale(monkeypatch):
    def fake_get_config(key, default=None):
        values = {
            "knowledge.customer_scene_keywords.received": ["穿了"],
            "knowledge.customer_scene_keywords.problem": ["起球"],
        }
        return values.get(key, default)

    monkeypatch.setattr(knowledge_service, "get_config", fake_get_config)

    assert KnowledgeService.detect_customer_scene("穿了两天就起球", default="presale") == "presale"


def test_replace_meta_entries_string_false_disables_entry(monkeypatch, tmp_path):
    db_path = tmp_path / "knowledge.db"
    manager = DatabaseManager(db_path=str(db_path))
    monkeypatch.setattr(knowledge_service, "db_manager", manager)
    service = KnowledgeService()

    service.replace_meta_entries(
        1,
        [
            {
                "source_type": "customer_service",
                "source_id": 1,
                "scenario": "presale",
                "aliases": "问法",
                "answer": "答案",
                "enabled": "false",
            }
        ],
    )

    with service.get_session() as session:
        entry = session.scalars(select(KnowledgeMetaEntry)).one()

    assert entry.enabled is False


def test_update_scene_knowledge_string_false_disables_entry(monkeypatch, tmp_path):
    db_path = tmp_path / "knowledge.db"
    manager = DatabaseManager(db_path=str(db_path))
    monkeypatch.setattr(knowledge_service, "db_manager", manager)
    service = KnowledgeService()

    with service.get_session() as session:
        row = PresaleKnowledge(
            shop_id=1,
            goods_id=100,
            aliases="旧问法",
            answer="旧答案",
            enabled=True,
        )
        session.add(row)
        session.commit()
        entry_id = row.id

    assert service.update_scene_knowledge(
        "presale",
        entry_id,
        aliases="新问法",
        answer="新答案",
        enabled="false",
    )

    with service.get_session() as session:
        entry = session.get(PresaleKnowledge, entry_id)

    assert entry.enabled is False


def test_update_scene_knowledge_invalid_priority_keeps_existing_value(monkeypatch, tmp_path):
    db_path = tmp_path / "knowledge.db"
    manager = DatabaseManager(db_path=str(db_path))
    monkeypatch.setattr(knowledge_service, "db_manager", manager)
    service = KnowledgeService()

    with service.get_session() as session:
        row = PresaleKnowledge(
            shop_id=1,
            goods_id=100,
            aliases="旧问法",
            answer="旧答案",
            priority=7,
            enabled=True,
        )
        session.add(row)
        session.commit()
        entry_id = row.id

    assert service.update_scene_knowledge(
        "presale",
        entry_id,
        aliases="新问法",
        answer="新答案",
        priority="bad",
    )

    with service.get_session() as session:
        entry = session.get(PresaleKnowledge, entry_id)

    assert entry.priority == 7
    assert entry.aliases == "新问法"
    assert entry.answer == "新答案"


def test_update_scene_knowledge_corrupt_existing_priority_falls_back_to_zero(monkeypatch, tmp_path):
    db_path = tmp_path / "knowledge.db"
    manager = DatabaseManager(db_path=str(db_path))
    monkeypatch.setattr(knowledge_service, "db_manager", manager)
    service = KnowledgeService()

    with service.get_session() as session:
        row = PresaleKnowledge(
            shop_id=1,
            goods_id=100,
            aliases="旧问法",
            answer="旧答案",
            priority=7,
            enabled=True,
        )
        session.add(row)
        session.commit()
        entry_id = row.id
        session.execute(
            text("UPDATE presale_knowledge SET priority = :priority WHERE id = :id"),
            {"priority": "bad-existing-priority", "id": entry_id},
        )
        session.commit()

    assert service.update_scene_knowledge(
        "presale",
        entry_id,
        aliases="新问法",
        answer="新答案",
        priority="bad-new-priority",
    )

    with service.get_session() as session:
        entry = session.get(PresaleKnowledge, entry_id)

    assert entry.priority == 0
    assert entry.aliases == "新问法"
    assert entry.answer == "新答案"


def test_replace_meta_entries_skips_invalid_rows_without_crashing(monkeypatch, tmp_path):
    db_path = tmp_path / "knowledge.db"
    manager = DatabaseManager(db_path=str(db_path))
    monkeypatch.setattr(knowledge_service, "db_manager", manager)
    service = KnowledgeService()

    created = service.replace_meta_entries(
        1,
        [
            {
                "source_type": "customer_service",
                "source_id": "bad-source",
                "scenario": "presale",
                "aliases": "坏问法",
                "answer": "坏答案",
            },
            {
                "source_type": "customer_service",
                "source_id": "2",
                "scenario": "presale",
                "aliases": "好问法",
                "answer": "好答案",
                "priority": "bad-priority",
            },
        ],
    )

    assert created == 1
    with service.get_session() as session:
        entry = session.scalars(select(KnowledgeMetaEntry)).one()

    assert entry.source_id == 2
    assert entry.aliases == "好问法"
    assert entry.priority == 0


def test_rank_meta_entries_treats_corrupt_priority_as_zero():
    service = KnowledgeService.__new__(KnowledgeService)
    entry = SimpleNamespace(
        id=1,
        scenario="presale",
        sub_intent="续航",
        aliases="续航多久/能用多久",
        answer="续航约 3 小时",
        section_title="参数",
        tags="",
        priority="bad-priority",
    )

    ranked = service._rank_meta_entries([entry], "续航多久", limit=3)

    assert ranked == [entry]


def test_rank_scene_entries_treats_corrupt_priority_as_zero():
    service = KnowledgeService.__new__(KnowledgeService)
    service.vector_retriever = None
    bad_priority = SimpleNamespace(
        id=1,
        priority="bad-priority",
        aliases="续航多久",
        answer="续航约 3 小时",
        section_title="参数",
        sub_intent="续航",
        tags="",
    )
    good_priority = SimpleNamespace(
        id=2,
        priority=3,
        aliases="续航多久",
        answer="续航约 4 小时",
        section_title="参数",
        sub_intent="续航",
        tags="",
    )

    ranked = service._score_entries([bad_priority, good_priority], query="续航多久", goods_id=None)

    assert [item[0] for item in ranked] == [good_priority, bad_priority]


def test_score_entries_uses_parameter_type_tags_for_duration_query(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        _taxonomy_config({
            "knowledge.intent_score_adjustment_rules": [],
            "knowledge.parameter_type_query_hints": {
                "duration": ["续航", "用多久"],
                "charging": ["充电", "充电头"],
                "gear": ["档位", "几档"],
                "price": ["价格", "多少钱"],
            },
        }),
    )
    monkeypatch.setattr(KnowledgeService, "_intent_score_adjustment", classmethod(lambda cls, *args, **kwargs: 0))
    service = KnowledgeService.__new__(KnowledgeService)
    service.vector_retriever = None

    duration_entry = SimpleNamespace(
        id=1,
        priority=0,
        aliases="",
        answer="最高档约2小时",
        section_title="规格参数",
        sub_intent="参数说明",
        tags="parameter_type:duration unit:hour",
        goods_id=123,
        source_type="product",
    )
    charging_entry = SimpleNamespace(
        id=2,
        priority=0,
        aliases="",
        answer="内置充电电池，支持5V普通充电头",
        section_title="规格参数",
        sub_intent="参数说明",
        tags="parameter_type:charging",
        goods_id=123,
        source_type="product",
    )
    gear_entry = SimpleNamespace(
        id=3,
        priority=0,
        aliases="",
        answer="199档调节",
        section_title="规格参数",
        sub_intent="档位数量",
        tags="parameter_type:gear",
        goods_id=123,
        source_type="product",
    )
    price_entry = SimpleNamespace(
        id=4,
        priority=0,
        aliases="",
        answer="价格以页面当前优惠为准",
        section_title="价格优惠",
        sub_intent="页面价格",
        tags="parameter_type:price",
        goods_id=123,
        source_type="product",
    )

    ranked = service._score_entries(
        [charging_entry, gear_entry, price_entry, duration_entry],
        query="续航多久",
        goods_id=123,
        scene_key="presale",
    )

    assert ranked[0][0] is duration_entry
    assert all(item[0] is not price_entry for item in ranked[:2])


def test_battery_duration_query_no_longer_has_hardcoded_score_adjustment(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.intent_score_adjustment_rules"
        else default,
    )
    duration_entry = SimpleNamespace(
        section_title="续航参数",
        sub_intent="使用时间",
        answer="最高档约2小时",
        aliases="",
        source_type="product",
    )
    charging_entry = SimpleNamespace(
        section_title="充电方式",
        sub_intent="充电款确认",
        answer="内置充电电池，支持5V普通充电头",
        aliases="",
        source_type="product",
    )
    gear_entry = SimpleNamespace(
        section_title="风力参数",
        sub_intent="档位数量",
        answer="199档调节",
        aliases="",
        source_type="product",
    )
    price_entry = SimpleNamespace(
        section_title="价格优惠",
        sub_intent="页面价格",
        answer="价格以页面为准",
        aliases="",
        source_type="product",
    )

    for entry in (duration_entry, charging_entry, gear_entry, price_entry):
        assert KnowledgeService._intent_score_adjustment(
            {"battery_duration_query"},
            entry,
            query="续航多久",
        ) == 0


def test_score_entries_uses_parameter_type_tags_for_battery_capacity(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        _taxonomy_config({
            "knowledge.intent_score_adjustment_rules": [],
            "knowledge.parameter_type_query_hints": {
                "battery_capacity": ["电池容量", "容量多大"],
                "duration": ["续航"],
                "wind": ["风力", "风速"],
            },
        }),
    )
    service = KnowledgeService.__new__(KnowledgeService)
    service.vector_retriever = None

    battery_entry = SimpleNamespace(
        id=1,
        priority=0,
        aliases="",
        answer="电池容量以页面参数为准",
        section_title="规格参数",
        sub_intent="电池容量",
        tags="parameter_type:battery_capacity",
        goods_id=123,
        source_type="product",
    )
    wind_entry = SimpleNamespace(
        id=2,
        priority=0,
        aliases="",
        answer="风力强，转速高",
        section_title="风力参数",
        sub_intent="风速档位",
        tags="parameter_type:wind",
        goods_id=123,
        source_type="product",
    )
    duration_entry = SimpleNamespace(
        id=3,
        priority=0,
        aliases="",
        answer="最高档约2小时",
        section_title="续航参数",
        sub_intent="使用时间",
        tags="parameter_type:duration",
        goods_id=123,
        source_type="product",
    )

    ranked = service._score_entries(
        [wind_entry, duration_entry, battery_entry],
        query="电池容量多大",
        goods_id=123,
        scene_key="presale",
    )

    assert ranked[0][0] is battery_entry


def test_score_entries_uses_parameter_type_tags_for_price_wind_and_cooling(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        _taxonomy_config({
            "knowledge.parameter_type_query_hints": {
                "price": ["多少钱", "价格"],
                "wind": ["风力", "风大"],
                "cooling": ["制冷", "半导体"],
                "duration": ["续航"],
            },
        }),
    )
    monkeypatch.setattr(KnowledgeService, "_intent_score_adjustment", classmethod(lambda cls, *args, **kwargs: 0))
    service = KnowledgeService.__new__(KnowledgeService)
    service.vector_retriever = None

    price_entry = SimpleNamespace(
        id=1,
        priority=0,
        aliases="",
        answer="页面价格以当前优惠为准",
        section_title="规格参数",
        sub_intent="参数说明",
        tags="parameter_type:price",
        goods_id=123,
        source_type="product",
    )
    wind_entry = SimpleNamespace(
        id=2,
        priority=0,
        aliases="",
        answer="风力强，转速高",
        section_title="规格参数",
        sub_intent="参数说明",
        tags="parameter_type:wind",
        goods_id=123,
        source_type="product",
    )
    cooling_entry = SimpleNamespace(
        id=3,
        priority=0,
        aliases="",
        answer="支持半导体冰感降温",
        section_title="规格参数",
        sub_intent="参数说明",
        tags="parameter_type:cooling",
        goods_id=123,
        source_type="product",
    )
    duration_entry = SimpleNamespace(
        id=4,
        priority=0,
        aliases="",
        answer="最高档约2小时",
        section_title="规格参数",
        sub_intent="参数说明",
        tags="parameter_type:duration",
        goods_id=123,
        source_type="product",
    )

    entries = [duration_entry, cooling_entry, wind_entry, price_entry]

    assert service._score_entries(entries, query="多少钱", goods_id=123, scene_key="presale")[0][0] is price_entry
    assert service._score_entries(entries, query="风力大吗", goods_id=123, scene_key="presale")[0][0] is wind_entry
    assert service._score_entries(entries, query="有制冷吗", goods_id=123, scene_key="presale")[0][0] is cooling_entry


def test_parameter_query_score_adjustments_are_not_hardcoded(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.intent_score_adjustment_rules"
        else default,
    )
    entries = [
        SimpleNamespace(section_title="价格优惠", sub_intent="页面价格", answer="页面价格以当前优惠为准", aliases="", source_type="product"),
        SimpleNamespace(section_title="风力参数", sub_intent="风力", answer="风力强，转速高", aliases="", source_type="product"),
        SimpleNamespace(section_title="制冷功能", sub_intent="冰感", answer="支持半导体冰感降温", aliases="", source_type="product"),
        SimpleNamespace(section_title="续航参数", sub_intent="使用时间", answer="最高档约2小时", aliases="", source_type="product"),
    ]

    for hints, query in (
        ({"price_query"}, "多少钱"),
        ({"wind_query"}, "风力大吗"),
        ({"cooling_query"}, "有制冷吗"),
    ):
        for entry in entries:
            assert KnowledgeService._intent_score_adjustment(hints, entry, query=query) == 0


def test_score_entries_uses_action_type_tags_for_order_action_queries(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        _taxonomy_config({
            "knowledge.action_type_query_hints": {
                "fulfillment_exception": ["少发", "漏发", "缺件"],
                "order_change": ["改地址", "备注"],
                "product_attribute": ["颜色", "库存"],
                "fault_handling": ["坏了", "不能用"],
            },
        }),
    )
    monkeypatch.setattr(KnowledgeService, "_intent_score_adjustment", classmethod(lambda cls, *args, **kwargs: 0))
    service = KnowledgeService.__new__(KnowledgeService)
    service.vector_retriever = None

    fulfillment_entry = SimpleNamespace(
        id=1,
        priority=0,
        aliases="",
        answer="少件漏发请转人工核实处理",
        section_title="售后处理",
        sub_intent="履约异常",
        tags="action_type:fulfillment_exception",
        goods_id=123,
        source_type="policy",
    )
    order_change_entry = SimpleNamespace(
        id=2,
        priority=0,
        aliases="",
        answer="地址或备注修改需要转人工确认",
        section_title="订单处理",
        sub_intent="订单修改",
        tags="action_type:order_change",
        goods_id=123,
        source_type="policy",
    )
    color_param_entry = SimpleNamespace(
        id=3,
        priority=0,
        aliases="",
        answer="当前有黑色和白色可选",
        section_title="颜色确认",
        sub_intent="颜色选项",
        tags="action_type:product_attribute",
        goods_id=123,
        source_type="product",
    )
    fault_entry = SimpleNamespace(
        id=4,
        priority=0,
        aliases="",
        answer="故障问题请描述具体现象",
        section_title="售后处理",
        sub_intent="故障处理",
        tags="action_type:fault_handling",
        goods_id=123,
        source_type="policy",
    )
    entries = [color_param_entry, fault_entry, order_change_entry, fulfillment_entry]

    assert service._score_entries(entries, query="少发了一个", goods_id=123, scene_key="aftersale")[0][0] is fulfillment_entry
    assert service._score_entries(entries, query="帮我改一下地址", goods_id=123, scene_key="insale")[0][0] is order_change_entry


def test_order_action_score_adjustments_are_not_hardcoded(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.intent_score_adjustment_rules"
        else default,
    )
    entries = [
        SimpleNamespace(section_title="颜色确认", sub_intent="颜色选项", answer="当前有黑色和白色可选", aliases="", source_type="product"),
        SimpleNamespace(section_title="订单处理", sub_intent="订单修改", answer="地址或备注修改需要转人工确认", aliases="", source_type="policy"),
        SimpleNamespace(section_title="售后处理", sub_intent="履约异常", answer="少件漏发请转人工核实处理", aliases="", source_type="policy"),
        SimpleNamespace(section_title="售后处理", sub_intent="故障处理", answer="故障问题请描述具体现象", aliases="", source_type="policy"),
    ]

    for hints, query in (
        ({"wrong_missing"}, "少发了一个"),
        ({"note_change"}, "帮我改一下地址"),
    ):
        for entry in entries:
            assert KnowledgeService._intent_score_adjustment(hints, entry, query=query) == 0


def test_score_entries_uses_complaint_type_tags_for_aftersale_complaints(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        _taxonomy_config({
            "knowledge.complaint_type_query_hints": {
                "battery_runtime": ["续航太短", "续航短"],
                "product_fault": ["坏了", "不转"],
                "noise": ["声音很吵", "噪音", "吵"],
            },
            "knowledge.action_type_query_hints": {
                "fault_handling": ["坏了", "不转", "声音很吵", "续航太短"],
            },
        }),
    )
    monkeypatch.setattr(KnowledgeService, "_intent_score_adjustment", classmethod(lambda cls, *args, **kwargs: 0))
    service = KnowledgeService.__new__(KnowledgeService)
    service.vector_retriever = None

    battery_complaint_entry = SimpleNamespace(
        id=1,
        priority=0,
        aliases="",
        answer="续航短问题请按售后核实处理",
        section_title="售后处理",
        sub_intent="续航投诉",
        tags="complaint_type:battery_runtime action_type:fault_handling",
        goods_id=123,
        source_type="policy",
    )
    fault_entry = SimpleNamespace(
        id=2,
        priority=0,
        aliases="",
        answer="商品故障请转人工核实处理",
        section_title="售后处理",
        sub_intent="故障处理",
        tags="complaint_type:product_fault action_type:fault_handling",
        goods_id=123,
        source_type="policy",
    )
    noise_entry = SimpleNamespace(
        id=3,
        priority=0,
        aliases="",
        answer="噪音异响问题请转人工核实处理",
        section_title="售后处理",
        sub_intent="噪音异响",
        tags="complaint_type:noise action_type:fault_handling",
        goods_id=123,
        source_type="policy",
    )
    parameter_entry = SimpleNamespace(
        id=4,
        priority=0,
        aliases="",
        answer="续航最高档约2小时，风力强",
        section_title="规格参数",
        sub_intent="参数说明",
        tags="parameter_type:duration parameter_type:wind",
        goods_id=123,
        source_type="product",
    )
    price_entry = SimpleNamespace(
        id=5,
        priority=0,
        aliases="",
        answer="页面价格以当前优惠为准",
        section_title="价格优惠",
        sub_intent="页面价格",
        tags="parameter_type:price",
        goods_id=123,
        source_type="product",
    )
    entries = [parameter_entry, price_entry, noise_entry, fault_entry, battery_complaint_entry]

    assert service._score_entries(entries, query="续航太短", goods_id=123, scene_key="aftersale")[0][0] is battery_complaint_entry
    assert service._score_entries(entries, query="坏了不转", goods_id=123, scene_key="aftersale")[0][0] is fault_entry
    assert service._score_entries(entries, query="声音很吵", goods_id=123, scene_key="aftersale")[0][0] is noise_entry


def test_aftersale_complaint_score_adjustments_are_not_hardcoded(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.intent_score_adjustment_rules"
        else default,
    )
    entries = [
        SimpleNamespace(section_title="售后处理", sub_intent="续航投诉", answer="续航短问题请按售后核实处理", aliases="", source_type="policy"),
        SimpleNamespace(section_title="售后处理", sub_intent="故障处理", answer="商品故障请转人工核实处理", aliases="", source_type="policy"),
        SimpleNamespace(section_title="售后处理", sub_intent="噪音异响", answer="噪音异响问题请转人工核实处理", aliases="", source_type="policy"),
        SimpleNamespace(section_title="规格参数", sub_intent="参数说明", answer="续航最高档约2小时，风力强", aliases="", source_type="product"),
        SimpleNamespace(section_title="价格优惠", sub_intent="页面价格", answer="页面价格以当前优惠为准", aliases="", source_type="product"),
    ]

    for hints, query in (
        ({"battery_complaint"}, "续航太短"),
        ({"aftersale_fault"}, "坏了不转"),
        ({"noise_fault"}, "声音很吵"),
    ):
        for entry in entries:
            assert KnowledgeService._intent_score_adjustment(
                hints,
                entry,
                scene_key="aftersale",
                query=query,
            ) == 0


def test_score_entries_uses_logistics_tags_for_logistics_queries(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        _taxonomy_config({
            "knowledge.action_type_query_hints": {"logistics": ["快递", "物流"]},
            "knowledge.case_type_query_hints": {"logistics": ["快递", "物流", "到哪了"]},
        }),
    )
    monkeypatch.setattr(KnowledgeService, "_intent_score_adjustment", classmethod(lambda cls, *args, **kwargs: 0))
    service = KnowledgeService.__new__(KnowledgeService)
    service.vector_retriever = None

    logistics_entry = SimpleNamespace(
        id=1,
        priority=0,
        aliases="",
        answer="物流信息以订单最新轨迹为准",
        section_title="物流处理",
        sub_intent="物流状态",
        tags="action_type:logistics case_type:logistics",
        goods_id=123,
        source_type="policy",
    )
    duration_entry = SimpleNamespace(
        id=2,
        priority=0,
        aliases="",
        answer="续航最高档约2小时",
        section_title="规格参数",
        sub_intent="续航参数",
        tags="parameter_type:duration",
        goods_id=123,
        source_type="product",
    )
    accessory_entry = SimpleNamespace(
        id=3,
        priority=0,
        aliases="",
        answer="配件以页面展示为准",
        section_title="配件说明",
        sub_intent="赠品配件",
        tags="action_type:product_attribute",
        goods_id=123,
        source_type="product",
    )
    entries = [duration_entry, accessory_entry, logistics_entry]

    ranked = service._score_entries(entries, query="快递到哪了", goods_id=123, scene_key="insale")

    assert ranked[0][0] is logistics_entry


def test_logistics_score_adjustment_is_not_hardcoded(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        lambda key, default=None: []
        if key == "knowledge.intent_score_adjustment_rules"
        else default,
    )
    entries = [
        SimpleNamespace(section_title="物流处理", sub_intent="物流状态", answer="物流信息以订单最新轨迹为准", aliases="", source_type="policy"),
        SimpleNamespace(section_title="规格参数", sub_intent="续航参数", answer="续航最高档约2小时", aliases="", source_type="product"),
        SimpleNamespace(section_title="配件说明", sub_intent="赠品配件", answer="配件以页面展示为准", aliases="", source_type="product"),
    ]

    for entry in entries:
        assert KnowledgeService._intent_score_adjustment(
            {"logistics"},
            entry,
            scene_key="insale",
            query="快递到哪了",
        ) == 0


def test_score_entries_uses_case_type_tags_for_logistics_queries(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        _taxonomy_config({
            "knowledge.case_type_query_hints": {"logistics": ["快递", "物流", "到哪了"]},
        }),
    )
    service = KnowledgeService.__new__(KnowledgeService)
    service.vector_retriever = None

    logistics_entry = SimpleNamespace(
        id=1,
        priority=0,
        aliases="",
        answer="物流信息以订单最新轨迹为准",
        section_title="物流处理",
        sub_intent="物流状态",
        tags="case_type:logistics",
        goods_id=123,
        source_type="policy",
    )
    duration_entry = SimpleNamespace(
        id=2,
        priority=0,
        aliases="",
        answer="续航最高档约2小时",
        section_title="规格参数",
        sub_intent="续航参数",
        tags="parameter_type:duration",
        goods_id=123,
        source_type="product",
    )
    entries = [duration_entry, logistics_entry]

    ranked = service._score_entries(entries, query="快递到哪了", goods_id=123, scene_key="insale")

    assert ranked[0][0] is logistics_entry


def test_score_entries_uses_logistics_case_type_for_arrival_time_queries(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        _taxonomy_config({
            "knowledge.intent_score_adjustment_rules": [],
            "knowledge.case_type_query_hints": {
                "logistics": ["多久到货", "到货", "物流"],
                "refund": ["退款", "退货"],
            },
        }),
    )
    service = KnowledgeService.__new__(KnowledgeService)
    service.vector_retriever = None

    logistics_entry = SimpleNamespace(
        id=1,
        priority=0,
        aliases="",
        answer="到货时效以订单物流最新轨迹为准",
        section_title="物流处理",
        sub_intent="到货时效",
        tags="case_type:logistics",
        goods_id=123,
        source_type="policy",
    )
    refund_entry = SimpleNamespace(
        id=2,
        priority=0,
        aliases="",
        answer="退货退款按售后规则处理",
        section_title="售后处理",
        sub_intent="退货退款",
        tags="case_type:refund",
        goods_id=123,
        source_type="policy",
    )
    duration_entry = SimpleNamespace(
        id=3,
        priority=0,
        aliases="",
        answer="续航最高档约2小时",
        section_title="规格参数",
        sub_intent="续航参数",
        tags="parameter_type:duration",
        goods_id=123,
        source_type="product",
    )

    ranked = service._score_entries(
        [refund_entry, duration_entry, logistics_entry],
        query="多久到货",
        goods_id=123,
        scene_key="insale",
    )

    assert ranked[0][0] is logistics_entry


def test_case_type_match_score_prefers_logistics_tags(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        _taxonomy_config({
            "knowledge.intent_score_adjustment_rules": [],
            "knowledge.case_type_query_hints": {"logistics": ["快递", "物流", "到哪了"]},
        }),
    )
    logistics_entry = SimpleNamespace(
        section_title="物流处理",
        sub_intent="物流状态",
        answer="物流信息以订单最新轨迹为准",
        aliases="",
        tags="case_type:logistics",
    )
    duration_entry = SimpleNamespace(
        section_title="规格参数",
        sub_intent="续航参数",
        answer="续航最高档约2小时",
        aliases="",
        tags="parameter_type:duration",
    )

    assert KnowledgeService._case_type_match_score("快递到哪了", logistics_entry) > 0
    assert KnowledgeService._case_type_match_score("快递到哪了", duration_entry) < 0


def test_score_entries_uses_case_type_tags_for_accessory_queries(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        _taxonomy_config({
            "knowledge.intent_score_adjustment_rules": [],
            "knowledge.case_type_query_hints": {"accessory": ["送充电器", "配件", "赠品"]},
        }),
    )
    service = KnowledgeService.__new__(KnowledgeService)
    service.vector_retriever = None

    accessory_entry = SimpleNamespace(
        id=1,
        priority=0,
        aliases="",
        answer="包装内含充电线，是否送充电头以页面展示为准",
        section_title="配件赠品",
        sub_intent="赠品清单",
        tags="case_type:accessory",
        goods_id=123,
        source_type="product",
    )
    charging_entry = SimpleNamespace(
        id=2,
        priority=0,
        aliases="",
        answer="支持5V普通充电头充电",
        section_title="充电方式",
        sub_intent="充电参数",
        tags="parameter_type:charging",
        goods_id=123,
        source_type="product",
    )
    duration_entry = SimpleNamespace(
        id=3,
        priority=0,
        aliases="",
        answer="最高档约2小时",
        section_title="续航参数",
        sub_intent="使用时间",
        tags="parameter_type:duration",
        goods_id=123,
        source_type="product",
    )

    ranked = service._score_entries(
        [charging_entry, duration_entry, accessory_entry],
        query="送充电器吗",
        goods_id=123,
        scene_key="presale",
    )

    assert ranked[0][0] is accessory_entry


def test_score_entries_uses_case_type_tags_for_stock_queries(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        _taxonomy_config({
            "knowledge.intent_score_adjustment_rules": [],
            "knowledge.case_type_query_hints": {"stock": ["黑色", "有货", "库存"]},
        }),
    )
    service = KnowledgeService.__new__(KnowledgeService)
    service.vector_retriever = None

    stock_entry = SimpleNamespace(
        id=1,
        priority=0,
        aliases="",
        answer="颜色和库存以页面可选项为准",
        section_title="颜色库存",
        sub_intent="颜色可选",
        tags="case_type:stock",
        goods_id=123,
        source_type="product",
    )
    duration_entry = SimpleNamespace(
        id=2,
        priority=0,
        aliases="",
        answer="最高档约2小时",
        section_title="续航参数",
        sub_intent="使用时间",
        tags="parameter_type:duration",
        goods_id=123,
        source_type="product",
    )
    price_entry = SimpleNamespace(
        id=3,
        priority=0,
        aliases="",
        answer="价格以页面当前优惠为准",
        section_title="价格优惠",
        sub_intent="页面价格",
        tags="parameter_type:price",
        goods_id=123,
        source_type="product",
    )

    ranked = service._score_entries(
        [duration_entry, price_entry, stock_entry],
        query="有黑色吗",
        goods_id=123,
        scene_key="presale",
    )

    assert ranked[0][0] is stock_entry


def test_score_entries_demotes_tutorial_tags_for_non_tutorial_queries(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        _taxonomy_config({
            "knowledge.intent_score_adjustment_rules": [],
            "knowledge.case_type_query_hints": {"tutorial": ["怎么用", "开机"]},
            "knowledge.parameter_type_query_hints": {"duration": ["续航"]},
        }),
    )
    service = KnowledgeService.__new__(KnowledgeService)
    service.vector_retriever = None

    duration_entry = SimpleNamespace(
        id=1,
        priority=0,
        aliases="",
        answer="最高档约2小时",
        section_title="续航参数",
        sub_intent="使用时间",
        tags="parameter_type:duration",
        goods_id=123,
        source_type="product",
    )
    tutorial_entry = SimpleNamespace(
        id=2,
        priority=0,
        aliases="",
        answer="长按开机",
        section_title="按键",
        sub_intent="开关机教程",
        tags="case_type:tutorial",
        goods_id=123,
        source_type="product",
    )

    ranked = service._score_entries(
        [tutorial_entry, duration_entry],
        query="续航多久",
        goods_id=123,
        scene_key="presale",
    )

    assert ranked[0][0] is duration_entry


def test_score_entries_uses_tutorial_tags_for_tutorial_queries(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        _taxonomy_config({
            "knowledge.intent_score_adjustment_rules": [],
            "knowledge.case_type_query_hints": {"tutorial": ["怎么开机", "怎么用", "教程"]},
        }),
    )
    service = KnowledgeService.__new__(KnowledgeService)
    service.vector_retriever = None

    tutorial_entry = SimpleNamespace(
        id=1,
        priority=0,
        aliases="",
        answer="长按开机",
        section_title="按键",
        sub_intent="开关机教程",
        tags="case_type:tutorial",
        goods_id=123,
        source_type="product",
    )
    duration_entry = SimpleNamespace(
        id=2,
        priority=0,
        aliases="",
        answer="最高档约2小时",
        section_title="续航参数",
        sub_intent="使用时间",
        tags="parameter_type:duration",
        goods_id=123,
        source_type="product",
    )

    ranked = service._score_entries(
        [duration_entry, tutorial_entry],
        query="怎么开机",
        goods_id=123,
        scene_key="presale",
    )

    assert ranked[0][0] is tutorial_entry


def test_type_query_hints_are_empty_by_default(monkeypatch):
    monkeypatch.setattr(knowledge_service, "get_config", lambda key, default=None: default)

    assert KnowledgeService._query_parameter_types("续航多久 风力大吗 有制冷吗") == set()
    assert KnowledgeService._query_complaint_types("续航太短 风小 风扇响") == set()
    assert KnowledgeService._query_case_types("送挂绳吗 快递到哪了 怎么开机") == set()


def test_apply_vector_scores_masks_query_embedding_failure(monkeypatch):
    messages = []

    class FakeLogger:
        def debug(self, message):
            messages.append(str(message))

    class BrokenRetriever:
        def _embed(self, _query):
            raise RuntimeError("token=secret-token")

    monkeypatch.setattr(knowledge_service, "logger", FakeLogger())

    service = KnowledgeService.__new__(KnowledgeService)
    service.vector_retriever = BrokenRetriever()
    entry = SimpleNamespace(id=1)

    result = service._apply_vector_scores([(entry, 12, "keyword")], "续航多久", "presale", None)

    assert result == [(entry, 12, 0, 12, "keyword")]
    joined = "\n".join(messages)
    assert "secret-token" not in joined
    assert "token=***" in joined


def test_score_entries_allows_vector_only_candidates(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        _taxonomy_config({
            "knowledge.case_type_query_hints": {"logistics": ["快递员", "快递", "取件"]},
        }),
    )
    service = KnowledgeService.__new__(KnowledgeService)
    service.vector_retriever = object()

    keyword_entry = SimpleNamespace(
        id=1,
        priority=0,
        aliases="快递员什么时候联系",
        answer="取件时间以物流通知为准",
        section_title="物流处理",
        sub_intent="取件联系",
        tags="case_type:logistics",
        goods_id=123,
        source_type="policy",
    )
    semantic_entry = SimpleNamespace(
        id=2,
        priority=0,
        aliases="",
        answer="上门取件前一般会由快递员联系，请留意电话或短信通知",
        section_title="退货取件",
        sub_intent="上门取件联系时间",
        tags="case_type:logistics",
        goods_id=123,
        source_type="policy",
    )

    def fake_apply_vector_scores(pre_scored, query, scene_key, goods_id):
        result = []
        for entry, rule_score, match_type in pre_scored:
            if entry is semantic_entry:
                result.append((entry, rule_score, 0.7, 350, "vector"))
            else:
                result.append((entry, rule_score, 0.0, rule_score, match_type))
        return result

    monkeypatch.setattr(service, "_apply_vector_scores", fake_apply_vector_scores)

    ranked = service._score_entries(
        [keyword_entry, semantic_entry],
        query="快递员啥时候给我打电话",
        goods_id=123,
        scene_key="insale",
    )

    assert ranked[0][0] is semantic_entry
    assert ranked[0][4] == "vector"


def test_quality_gate_filters_vector_only_topic_conflicts(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        _taxonomy_config({
            "knowledge.case_type_query_hints": {"logistics": ["快递员", "快递", "取件"]},
        }),
    )
    service = KnowledgeService.__new__(KnowledgeService)
    service.vector_retriever = object()

    logistics_entry = SimpleNamespace(
        id=1,
        priority=0,
        aliases="快递员什么时候联系",
        answer="上门取件前一般会由快递员联系，请留意电话或短信通知",
        section_title="退货取件",
        sub_intent="上门取件联系时间",
        tags="case_type:logistics",
        goods_id=123,
        source_type="policy",
    )
    product_entry = SimpleNamespace(
        id=2,
        priority=0,
        aliases="",
        answer="风扇支持静音模式，风力档位可调",
        section_title="风力噪音功能",
        sub_intent="静音特性确认",
        tags="case_type:product_function parameter_type:wind",
        goods_id=123,
        source_type="product",
    )

    def fake_apply_vector_scores(pre_scored, query, scene_key, goods_id):
        result = []
        for entry, rule_score, match_type in pre_scored:
            if entry is product_entry:
                result.append((entry, rule_score, 0.9, 450, "vector"))
            else:
                result.append((entry, rule_score, 0.5, max(rule_score, 250), match_type))
        return result

    monkeypatch.setattr(service, "_apply_vector_scores", fake_apply_vector_scores)

    ranked = service._score_entries(
        [product_entry, logistics_entry],
        query="快递员什么时候联系我",
        goods_id=123,
        scene_key="insale",
    )

    assert ranked[0][0] is logistics_entry
    assert product_entry not in [item[0] for item in ranked]


def test_quality_gate_keeps_parameter_hits(monkeypatch):
    monkeypatch.setattr(
        knowledge_service,
        "get_config",
        _taxonomy_config({
            "knowledge.parameter_type_query_hints": {"gear": ["199档", "档位", "多少档"]},
        }),
    )
    service = KnowledgeService.__new__(KnowledgeService)
    service.vector_retriever = object()

    gear_entry = SimpleNamespace(
        id=1,
        priority=0,
        aliases="199档怎么调/档位怎么调",
        answer="199档可通过加减键调节，具体以页面规格为准",
        section_title="档位参数",
        sub_intent="档位调节",
        tags="parameter_type:gear",
        goods_id=123,
        source_type="product",
    )

    def fake_apply_vector_scores(pre_scored, query, scene_key, goods_id):
        return [(entry, rule_score, 0.6, rule_score + 300, "hybrid") for entry, rule_score, _ in pre_scored]

    monkeypatch.setattr(service, "_apply_vector_scores", fake_apply_vector_scores)

    ranked = service._score_entries(
        [gear_entry],
        query="199档怎么调",
        goods_id=123,
        scene_key="presale",
    )

    assert ranked[0][0] is gear_entry


def test_bm25_like_keyword_score_prefers_alias_and_title_hits(monkeypatch):
    monkeypatch.setattr(knowledge_service, "get_config", lambda key, default=None: default)

    alias_entry = SimpleNamespace(
        aliases="快递员联系时间",
        answer="请留意通知",
        section_title="退货取件",
        sub_intent="上门取件",
        tags="case_type:logistics",
    )
    answer_entry = SimpleNamespace(
        aliases="",
        answer="快递员联系时间以物流通知为准",
        section_title="通用说明",
        sub_intent="说明",
        tags="",
    )

    words = KnowledgeService._search_terms("快递员联系时间")

    assert KnowledgeService._bm25_like_match_score(words, alias_entry) > KnowledgeService._bm25_like_match_score(words, answer_entry)
