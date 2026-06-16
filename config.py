"""
配置文件管理模块。
读取 config.json，提供线程安全的配置访问、校验、保存接口。
"""
import json
import copy
import threading
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator
from core.base_service import _sanitize_for_log
from utils.scene_prompt_paths import DEFAULT_SCENE_PROMPT_FILES


class ModelType(str, Enum):
    """模型类型枚举"""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    KIMI = "kimi"
    CLAUDE = "claude"


class LLMConfig(BaseModel):
    """LLM 配置模型"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_name: str = Field(default="", description="模型名称")
    api_key: str = Field(default="", description="API 密钥")
    api_base: str = Field(default="", description="API 地址")
    max_tokens: int = Field(default=512, description="最大输出 token 数")
    tool_call_max_tokens: int = Field(default=512, description="工具调用最大输出 token 数")
    request_timeout_seconds: float = Field(default=20.0, description="主模型请求超时时间")
    disable_thinking: bool = Field(default=True, description="是否为匹配模型禁用 thinking 参数")
    disable_thinking_api_base_patterns: list[str] = Field(
        default_factory=lambda: ["127.0.0.1", "localhost", "xiaomimimo.com", "siliconflow.cn"],
        description="API 地址包含这些片段时禁用 thinking",
    )
    disable_thinking_model_prefixes: list[str] = Field(
        default_factory=lambda: ["mimo-", "glm-", "qwen/", "nex-agi/"],
        description="模型名称带这些前缀时禁用 thinking",
    )
    fallback: Dict[str, Any] = Field(default_factory=dict, description="兜底模型配置")


class BusinessHoursConfig(BaseModel):
    """营业时间配置模型"""
    start: str = Field(default="08:00", description="开始时间")
    end: str = Field(default="23:00", description="结束时间")

    @field_validator("start", "end")
    @classmethod
    def validate_time_format(cls, value: str) -> str:
        """验证时间格式 HH:MM"""
        try:
            datetime.strptime(value, "%H:%M")
            return value
        except ValueError:
            raise ValueError("时间格式必须为 HH:MM，例如 08:00")


class NightModeConfig(BaseModel):
    """夜间不转人工时间配置模型"""
    start: str = Field(default="23:00", description="夜间模式开始时间")
    end: str = Field(default="08:00", description="夜间模式结束时间")
    reply_templates: list[str] = Field(
        default_factory=lambda: [
            "亲，当前问题需要高级客服为您处理，高级客服上班时间为{work_time_text}，建议您晚点联系这边由高级客服为您处理哦！",
            "亲，专业的高级客服下班了，还没上班，{resume_text}后联系这边，会为您妥善处理的，您耐心等待下。",
            "亲，您的问题这边已经收到啦，目前高级客服不在线，{resume_text}后会有专人继续帮您处理，请您先放心。",
            "亲，您先别着急，夜间无法转接高级客服，{resume_text}后客服上班会继续为您核实处理的。",
            "亲，您反馈的情况我已经了解，目前夜间只能先为您记录，高级客服上班后会优先处理。",
            "亲，现在是夜间值守时段，高级客服暂时不在线，您可以先把情况补充完整，{resume_text}后会继续处理。",
            "亲，已经帮您记录诉求了，当前时段无法转人工，{resume_text}后高级客服会接着为您处理。",
            "亲，您连续发的消息我这边都收到了，请您先耐心等一下，高级客服上班后会为您处理。",
        ],
        description="夜间不转人工回复模板，支持 {range_text}/{resume_text}/{work_time_text}",
    )

    @field_validator("start", "end")
    @classmethod
    def validate_time_format(cls, value: str) -> str:
        """验证时间格式 HH:MM"""
        try:
            datetime.strptime(value, "%H:%M")
            return value
        except ValueError:
            raise ValueError("时间格式必须为 HH:MM，例如 23:00")


class PromptConfig(BaseModel):
    """提示词配置模型"""
    instructions: list[str] = Field(default_factory=list, description="指令")


class AgentRuntimeConfig(BaseModel):
    """Agent 运行配置模型"""
    scene_prompt_files: Dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_SCENE_PROMPT_FILES),
        description="不同客服场景对应的提示词文件路径",
    )
    high_risk_aftersale_transfer_phrases: list[str] = Field(
        default_factory=list,
        description="售后预检直转人工短语列表；默认不内置商品专属短语",
    )
    image_grounding_forbidden_symbols: list[str] = Field(
        default_factory=list,
        description="禁止从图片中直接推断商品功能的视觉符号；默认不内置特定商品符号",
    )
    image_grounding_forbidden_functions: list[str] = Field(
        default_factory=list,
        description="禁止仅凭图片符号推断的商品功能；默认不内置特定商品功能",
    )
    version_name_tokens: list[str] = Field(
        default_factory=list,
        description="商品版本/规格名列表；默认不内置特定店铺版本名",
    )
    usage_feedback_keywords: list[str] = Field(
        default_factory=lambda: [
            "好响",
            "声音大",
            "噪音大",
            "吵",
            "滋滋",
            "异响",
            "没反应",
            "坏了",
            "不转",
            "充不进",
            "充不上",
            "正在用",
            "收到货了",
            "已经收到",
            "已收到",
            "无法使用",
            "功能异常",
        ],
        description="客户已开始使用或明确反馈使用体验时的关键词",
    )
    order_fault_examples: list[str] = Field(
        default_factory=lambda: ["噪音", "异响", "无法使用", "功能异常", "坏了"],
        description="订单已签收硬约束中使用的售后故障示例词",
    )
    unreceived_patterns: list[str] = Field(
        default_factory=lambda: [
            "还没收到",
            "还没有收到",
            "没收到",
            "没有收到",
            "未收到",
            "未收货",
            "没收货",
            "没有收货",
            "未签收",
        ],
        description="客户明确表示尚未收到/签收时的否定模式",
    )
    night_mode_fault_examples: list[str] = Field(
        default_factory=lambda: ["坏了", "没反应", "不转", "充不进电", "噪音大"],
        description="夜间模式提示中展示的故障示例词",
    )
    transfer_escalation_examples: list[str] = Field(
        default_factory=lambda: [
            "已签收异常",
            "坏了不能用",
            "噪音大",
            "功能异常",
            "无法使用",
            "少件漏发",
            "退款/退货/赔付",
        ],
        description="转人工工具说明中的升级处理示例",
    )
    search_knowledge_query_examples: list[str] = Field(
        default_factory=lambda: [
            "商品参数",
            "功能",
            "图片里的按键/图标/部件用途",
            "材质",
            "尺寸",
            "发货",
            "物流",
            "退换货",
            "售后处理",
        ],
        description="知识检索工具说明中的查询示例",
    )
    grounded_knowledge_topics: list[str] = Field(
        default_factory=lambda: [
            "商品参数",
            "功能",
            "按键/图标/部件用途",
            "快递",
            "发货地",
        ],
        description="系统提示中必须以知识库/工具为准的主题列表",
    )
    missing_goods_parameter_keywords: list[str] = Field(
        default_factory=lambda: [
            "商品参数",
            "参数",
            "规格",
            "型号",
            "款式",
            "尺寸",
            "尺码",
            "大小",
            "长度",
            "宽度",
            "高度",
            "重量",
            "容量",
            "功率",
            "电压",
            "材质",
            "面料",
            "成分",
            "保质期",
            "有效期",
            "生产日期",
            "怎么用",
            "使用方法",
            "安装",
            "配套",
            "配件",
            "赠品",
            "发几个",
            "按键",
            "按钮",
            "开关",
            "图标",
            "标识",
            "功能",
            "颜色",
            "库存",
        ],
        description="没有识别到 goods_id 时，触发商品专属知识缺失约束的客户问题关键词",
    )
    missing_goods_parameter_topics: list[str] = Field(
        default_factory=lambda: ["参数", "规格", "尺寸", "材质", "按键/图标/功能", "颜色", "配件"],
        description="未锁定商品时回复中要求客户确认具体商品的主题示例",
    )
    missing_goods_unverified_fact_examples: list[str] = Field(
        default_factory=lambda: ["参数数值", "规格型号", "材质成分", "尺寸重量", "功能", "赠品承诺"],
        description="未锁定商品时禁止编造的未核实事实示例",
    )
    daytime_night_mode_leak_markers: list[str] = Field(
        default_factory=lambda: [
            "夜间时段",
            "夜间不转人工",
            "高级客服已下班",
            "高级客服下班了",
            "高级客服目前下班",
            "当前高级客服不在线",
            "还没上班",
            "上班时间是早上8点",
            "上班时间为早上8点",
            "高级客服上班时间",
            "建议您晚点联系",
            "建议您晚点再联系",
        ],
        description="非夜间回复中需要清理的夜间话术泄漏标记",
    )

class ReplySanitizerConfig(BaseModel):
    """最终回复清洗配置模型"""
    replacements: Dict[str, str] = Field(
        default_factory=lambda: {"运费险": "退货包运费服务"},
        description="客户可见回复中的词语替换表",
    )
    internal_sentence_terms: list[str] = Field(
        default_factory=lambda: [
            "知识库",
            "RAG",
            "rag",
            "预检索",
            "检索结果",
            "未提供明确数据",
            "未提供具体数据",
            "未找到相关",
        ],
        description="最终回复中命中后整句删除的内部术语列表",
    )


class ProductFamilyRule(BaseModel):
    """商品族识别规则"""
    family: str = Field(default="", description="商品族标识")
    contains: list[str] = Field(default_factory=list, description="包含任一文本时命中")
    regex: list[str] = Field(default_factory=list, description="任一正则命中时匹配")


class ProductParameterAliasRule(BaseModel):
    """商品参数问法别名规则"""
    contains_any: list[str] = Field(default_factory=list, description="参数行包含任一文本时命中")
    alias: str = Field(default="", description="追加到检索块的问法别名")


class KnowledgeConfig(BaseModel):
    """知识检索配置模型"""
    product_family_rules: list[ProductFamilyRule] = Field(
        default_factory=list,
        description="从商品标题/规格/知识文本识别商品族的规则；默认不内置特定店铺商品族",
    )
    tag_score_adjustments: Dict[str, Any] = Field(
        default_factory=dict,
        description="知识标签命中后的排序加权；默认不内置特定知识批次标签",
    )
    parameter_type_query_hints: Dict[str, list[str]] = Field(
        default_factory=dict,
        description="参数类型查询提示词；默认空，按店铺/类目显式配置",
    )
    action_type_query_hints: Dict[str, list[str]] = Field(
        default_factory=dict,
        description="动作类型查询提示词；默认空，按店铺/类目显式配置",
    )
    complaint_type_query_hints: Dict[str, list[str]] = Field(
        default_factory=dict,
        description="投诉类型查询提示词；默认空，按店铺/类目显式配置",
    )
    case_type_query_hints: Dict[str, list[str]] = Field(
        default_factory=dict,
        description="案例类型查询提示词；默认空，按店铺/类目显式配置",
    )
    h_model_source_types: list[str] = Field(
        default_factory=list,
        description="明确 H 型号查询时可优先命中的人工确认来源类型，默认关闭，按店铺/知识批次显式配置",
    )
    intent_specific_score_rules: Optional[list[Dict[str, Any]]] = Field(
        default=None,
        description="短问法/相近主题的检索加权规则；null 使用空默认，显式列表才启用",
    )
    intent_score_adjustment_rules: Optional[list[Dict[str, Any]]] = Field(
        default=None,
        description="意图命中后的条目加权规则；null 使用内置默认规则，[] 表示关闭",
    )
    product_parameter_keywords: list[str] = Field(
        default_factory=lambda: [
            "商品参数",
            "参数",
            "规格",
            "型号",
            "款式",
            "尺寸",
            "尺码",
            "重量",
            "容量",
            "功率",
            "电压",
            "材质",
            "面料",
            "成分",
            "功能",
            "使用方法",
            "安装",
            "配件",
            "赠品",
            "颜色",
            "库存",
            "快递",
            "发货",
        ],
        description="从商品知识文本切出参数块的关键词",
    )
    product_parameter_alias_rules: list[ProductParameterAliasRule] = Field(
        default_factory=lambda: [
            ProductParameterAliasRule(contains_any=["功率"], alias="问法：功率多少瓦/几瓦/多少W/功率多大"),
            ProductParameterAliasRule(contains_any=["容量"], alias="问法：容量多大/容量是多少"),
            ProductParameterAliasRule(contains_any=["尺寸", "尺码"], alias="问法：尺寸多大/尺码怎么选/大小是多少"),
            ProductParameterAliasRule(contains_any=["重量"], alias="问法：重量多少/有多重"),
            ProductParameterAliasRule(contains_any=["材质", "面料", "成分"], alias="问法：是什么材质/什么面料/成分是什么"),
        ],
        description="商品参数块追加的问法别名规则",
    )
    qualifier_groups: list[list[str]] = Field(
        default_factory=list,
        description="检索排序时要求 query 与知识条目一致匹配的一组组限定词",
    )
    search_phrase_candidates: list[str] = Field(
        default_factory=lambda: [
            "商品参数", "参数", "规格", "型号", "款式", "尺寸", "尺码",
            "重量", "容量", "功率", "电压", "材质", "面料", "成分",
            "功能", "使用方法", "安装", "配件", "赠品", "颜色",
            "有货", "现货", "库存", "什么快递", "发货地", "质保",
            "保修", "退货包运费", "七天无理由", "7天无理由",
        ],
        description="知识检索时额外保留的短语候选词",
    )
    search_synonym_expansions: Dict[str, list[str]] = Field(
        default_factory=lambda: {
            "邮政": ["快递"],
            "拒收": ["退货", "退款", "拒签"],
            "顿丰": ["快递"],
        },
        description="知识检索 query 词扩展表，key 命中时追加 value 中的检索词",
    )
    structured_scenario_rules: Dict[str, list[str]] = Field(
        default_factory=lambda: {
            "product_attribute": ["参数", "规格", "型号", "尺寸", "尺码", "重量", "容量", "功率", "材质", "面料", "颜色", "库存"],
            "product_usage": ["怎么用", "使用教程", "使用方法", "说明书", "安装"],
            "shipping": ["快递", "发货", "物流", "到货", "发货地", "从哪发", "从哪里发"],
            "aftersale": ["质保", "保修", "退货", "退款", "运费", "运费险", "质量问题", "坏了"],
        },
        description="结构化知识排序的场景识别词表；可按品类替换或置空关闭",
    )
    structured_scenario_anchors: Dict[str, list[str]] = Field(
        default_factory=lambda: {
            "product_attribute": ["参数", "规格", "型号", "尺寸", "材质", "颜色", "库存"],
            "product_usage": ["使用", "教程", "说明书", "安装"],
            "shipping": ["发货", "物流", "快递"],
            "aftersale": ["售后", "退货", "退款", "质保", "质量问题"],
        },
        description="结构化知识排序中各场景对应的知识侧锚点词",
    )
    structured_intent_rules: Dict[str, Any] = Field(
        default_factory=lambda: {
            "gift_accessory": {"all": ["配件"], "any": ["送", "赠", "带", "有", "里面", "包装", "配", "附"]},
            "color_stock": {"all": ["颜色", "色"], "any": ["有货", "现货", "能拍", "拍下", "库存"]},
            "color_query": {"any": ["颜色", "色", "几种颜色", "什么颜色"]},
            "shipping_origin": {"any": ["发货地", "哪里发货", "从哪发货", "从哪里发货"]},
            "shipping_express": {"any": ["什么快递", "发啥快递", "哪家快递", "快递"]},
            "shipping_time": {"any": ["什么时候发货", "多久发货", "几天到", "什么时候到", "多久到", "加急", "还不发货", "不发货", "没发货", "货发了没有", "催发货", "尽快发货", "快点发货"]},
            "warranty": {"any": ["质保", "保修", "坏了怎么办", "质量问题怎么办"]},
            "return_shipping": {"any": ["退货包运费", "运费谁出", "运费险"]},
            "return_policy": {"any": ["可以退货吗", "退货政策", "退款", "7天无理由"]},
            "size_weight": {"any": ["尺寸", "多大", "多重", "重量", "几厘米"]},
        },
        description="结构化知识排序的细分意图规则，支持 any/all 条件；可置空关闭",
    )
    intent_keywords: Dict[str, list[str]] = Field(
        default_factory=lambda: {
            "logistics_query": ["快递", "物流", "包裹", "几小时到", "什么时候到", "到哪了", "到了吗", "发了吗", "寄出了", "还有多久到"],
            "arrival_time": ["几天到货", "多久到货", "什么时候到", "多久能到", "几天能到", "到货", "送达"],
            "battery_complaint": [],
            "wrong_missing": [
                "发错货", "发错颜色", "发错了", "错发", "颜色错", "颜色发错",
                "少了", "少了一个", "少了个", "少发", "少发了", "少发了一个", "漏发", "漏发了",
                "缺件", "缺少", "配件少",
            ],
            "note_change": [
                "备注一下", "备注发", "帮我改一下", "改一下地址", "能改地址", "更改收货地址", "改颜色", "换颜色",
                "别发错", "不要发错", "别弄错", "不要弄错", "混色", "混发", "两个颜色", "发两个颜色",
                "一黑一白", "一白一黑", "一绿一蓝", "一蓝一绿",
            ],
            "mix_color_words": ["颜色", "色"],
            "price_query": ["价格", "多少钱", "几块", "九块", "9块", "9.9", "990元", "太贵", "优惠价", "售价"],
            "wind_query": [],
            "battery_size_query": ["容量多大", "容量是多少"],
            "battery_duration_query": [],
            "logistics_delivery": [
                "发什么快递", "什么快递", "发哪家", "快递公司", "发货地", "哪里发货",
                "今天能发吗", "什么时候发货", "发中通吗", "发极兔吗", "发圆通吗",
                "发顺丰吗", "能指定快递吗", "几天到", "多久到", "什么时候到",
            ],
            "accessory": ["配件", "送什么", "赠品", "带什么", "包装里有什么"],
            "color_stock": [
                "颜色", "有什么颜色", "哪个颜色", "库存", "有货吗", "什么颜色",
            ],
            "aftersale_fault": [
                "不转", "坏了", "异响", "滋滋声", "声音大", "还吵", "风小",
                "充不进电", "没电", "用不了", "开不了", "没反应", "打不开",
                "不出风", "噪音", "响", "松动",
            ],
            "wind_power": [],
            "price": ["多少钱", "价格", "几块", "贵", "优惠", "券", "便宜", "打折", "活动价"],
            "noise_fault": ["噪音", "吵", "滋滋声", "异响", "声音大", "声音不正常"],
            "cooling_query": [],
        },
        description="客户问题轻量意图识别关键词",
    )
    customer_scene_keywords: Dict[str, list[str]] = Field(
        default_factory=lambda: {
            "direct_aftersale": [
                "我要退款", "我要退货", "我要退货退款", "退货退款", "退款退货",
                "退款", "退款不退货", "申请退货", "申请退款", "申请退货退款", "申请退货退款吧",
                "给我退款", "给我退钱", "我希望你退款", "想退款", "退钱", "仅退款",
                "退货", "退的话", "如果退", "能退吗", "可以退吗", "还能退吗",
                "不想要了", "不要了", "退了吧", "周末可以退", "周末再退",
                "现在申请", "退不了", "没法退", "不能退", "补偿", "赔偿", "退差价",
                "给我退", "要求退款", "运费险多少钱", "运费多少钱", "包运费险吗",
                "包运费吗", "退货包运费", "运费谁出", "退货运费",
            ],
            "received": ["收到", "到货", "签收", "刚拿到", "用了", "使用了", "买的"],
            "problem": [
                "打不开", "开不了", "不转", "没反应", "不能用", "用不了",
                "不出风", "没风", "不能吹", "不吹风", "突然不能吹",
                "充不了", "充不了电", "充不进", "充不进电", "不充电", "无法充电", "坏了", "坏的",
                "开关没反应", "开关没有反应", "开关还是没有反应", "一拔充电器没有反应",
                "拔充电器没有反应", "重新充电归零", "又归零", "不保电",
                "声音大", "声音太大", "声音很大", "噪音大", "噪音太大", "噪音很大",
                "异味", "有异味", "有味道",
                "发热", "很烫", "破损", "破了", "裂开", "裂了", "开裂", "松动",
                "少配件", "少件", "少东西", "配件少", "发错",
            ],
            "presale_quality_questions": [
                "有噪音吗", "声音大吗", "声音大不大", "噪音大吗", "噪音大不大",
                "静音吗", "是不是真的静音", "会不会很吵", "吵不吵",
            ],
            "quality_complaint": [
                "声音大跟", "声音大了", "声音比较大", "声音大的", "声音很吵", "声音太吵",
                "声音怎么这么大", "怎么这么大声音", "怎么声音这么大",
                "声特别大", "声音特别大", "声音特别响", "声音好大", "噪音好大",
                "噪音这么大", "噪音怎么这么大", "怎么这么大噪音", "噪音太大", "噪音很大", "有噪音啊", "有噪音了",
                "声音这么大", "声音太大", "声音很大", "声音还很大", "声音挺大",
                "电机声音大", "电机声音也挺大", "电机比较响", "电机大声", "不是静音", "不静音", "还静音呢",
                "吵聋", "吵死", "太吵", "很吵", "耳朵吵",
                "噪音有点大", "噪音比较大", "噪音大了", "声音有点大",
                "为什么声音", "为什么不是静音",
                "贴近脸", "跟没吹一样", "根本就不能用", "一点都不能用",
                "都用了不好使", "都用了，不好使", "用了不好使", "买回来就不好使",
                "不保电",
                "塑料味", "臭味", "有味道", "味道很大", "烧焦味", "发烫", "很烫",
            ],
            "used_bad": ["都用了不好使", "都用了，不好使", "用了不好使", "买回来就不好使", "刚拿到就不好使"],
            "direct_problem": [
                "打不开", "开不了", "不转", "没反应", "不能用", "用不了", "不好使",
                "不出风", "没风", "不能吹", "不吹风", "突然不能吹", "吹不了风",
                "充不了电", "充不进电", "充不去电", "不充电", "无法充电", "坏了", "坏的",
                "开关没反应", "开关没有反应", "开关还是没有反应", "一拔充电器没有反应",
                "拔充电器没有反应", "重新充电归零", "又归零", "不保电",
                "破损", "破了", "裂开", "裂了", "开裂", "碎的", "断了", "少配件", "少件", "发错",
            ],
            "problem_followup": ["怎么办", "咋办", "怎么处理", "怎么解决", "处理一下", "给处理", "补偿", "赔", "退"],
        },
        description="客服大场景识别关键词",
    )


class PinduoduoRequestConfig(BaseModel):
    """拼多多 MMS 请求头配置"""
    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/137.0.0.0 Safari/537.36"
        ),
        description="MMS HTTP 请求使用的 User-Agent",
    )
    sec_ch_ua: str = Field(
        default='"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
        description="MMS HTTP 请求使用的 sec-ch-ua",
    )
    sec_ch_ua_mobile: str = Field(default="?0", description="MMS HTTP 请求使用的 sec-ch-ua-mobile")
    sec_ch_ua_platform: str = Field(default='"Windows"', description="MMS HTTP 请求使用的 sec-ch-ua-platform")


class PinduoduoTransferConfig(BaseModel):
    """拼多多会话转接配置"""
    default_remark: str = Field(default="客服升级处理", description="转接人工时提交给平台的默认备注")


class PinduoduoImmediateAckConfig(BaseModel):
    """拼多多即时消息回执配置"""
    enabled: bool = Field(default=True, description="是否启用即时消息自动回执")
    message: str = Field(default="[玫瑰]", description="即时消息自动回执文案")
    context_types: list[str] = Field(
        default_factory=lambda: ["withdraw", "system_hint", "transfer"],
        description="需要发送即时回执的 ContextType 值",
    )


class PinduoduoOrderConfig(BaseModel):
    """拼多多订单上下文配置"""
    signed_trace_keywords: list[str] = Field(
        default_factory=lambda: [
            "包裹已签收",
            "包裹已签收！",
            "已签收",
            "快件已签收",
            "快件已签收，签收方式",
            "签收人是",
        ],
        description="物流轨迹文本中判定已签收的兜底关键词",
    )


class PinduoduoConfig(BaseModel):
    """拼多多渠道配置"""
    request: PinduoduoRequestConfig = Field(
        default_factory=PinduoduoRequestConfig,
        description="拼多多 MMS 请求配置",
    )
    transfer: PinduoduoTransferConfig = Field(
        default_factory=PinduoduoTransferConfig,
        description="拼多多会话转接配置",
    )
    immediate_ack: PinduoduoImmediateAckConfig = Field(
        default_factory=PinduoduoImmediateAckConfig,
        description="拼多多即时消息回执配置",
    )
    order: PinduoduoOrderConfig = Field(
        default_factory=PinduoduoOrderConfig,
        description="拼多多订单上下文配置",
    )


class ConfigModel(BaseModel):
    """配置模型"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    enable_turn_context: bool = Field(default=True, description="是否启用多轮上下文")
    enable_turn_context_log_only: bool = Field(default=True, description="是否仅记录多轮上下文而不注入回复")
    business_hours: BusinessHoursConfig = Field(
        default_factory=BusinessHoursConfig,
        description="营业时间配置",
    )
    night_mode: NightModeConfig = Field(
        default_factory=NightModeConfig,
        description="夜间不转人工配置",
    )
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM配置")
    prompt: PromptConfig = Field(default_factory=PromptConfig, description="提示词配置")
    agent: AgentRuntimeConfig = Field(default_factory=AgentRuntimeConfig, description="Agent运行配置")
    reply_sanitizer: ReplySanitizerConfig = Field(
        default_factory=ReplySanitizerConfig,
        description="最终回复清洗配置",
    )
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig, description="知识检索配置")
    pinduoduo: PinduoduoConfig = Field(default_factory=PinduoduoConfig, description="拼多多渠道配置")
    db_path: str = Field(default="", description="数据库路径")


config_base = {
    "enable_turn_context": True,
    "enable_turn_context_log_only": True,
    "business_hours": {
        "start": "08:00",
        "end": "23:00",
    },
    "night_mode": {
        "start": "23:00",
        "end": "08:00",
        "reply_templates": [
            "亲，当前问题需要高级客服为您处理，高级客服上班时间为{work_time_text}，建议您晚点联系这边由高级客服为您处理哦！",
            "亲，专业的高级客服下班了，还没上班，{resume_text}后联系这边，会为您妥善处理的，您耐心等待下。",
            "亲，您的问题这边已经收到啦，目前高级客服不在线，{resume_text}后会有专人继续帮您处理，请您先放心。",
            "亲，您先别着急，夜间无法转接高级客服，{resume_text}后客服上班会继续为您核实处理的。",
            "亲，您反馈的情况我已经了解，目前夜间只能先为您记录，高级客服上班后会优先处理。",
            "亲，现在是夜间值守时段，高级客服暂时不在线，您可以先把情况补充完整，{resume_text}后会继续处理。",
            "亲，已经帮您记录诉求了，当前时段无法转人工，{resume_text}后高级客服会接着为您处理。",
            "亲，您连续发的消息我这边都收到了，请您先耐心等一下，高级客服上班后会为您处理。",
        ],
    },
    "llm": {
        "model_name": "",
        "api_key": "",
        "api_base": "",
        "max_tokens": 512,
        "tool_call_max_tokens": 512,
        "request_timeout_seconds": 20,
        "disable_thinking": True,
        "disable_thinking_api_base_patterns": ["127.0.0.1", "localhost", "xiaomimimo.com", "siliconflow.cn"],
        "disable_thinking_model_prefixes": ["mimo-", "glm-", "qwen/", "nex-agi/"],
        "fallback": {
            "enabled": False,
            "model_name": "",
            "api_key": "",
            "api_base": "",
            "timeout_seconds": 20,
        },
    },
    "prompt": {
        "instructions": [
            "回复像真人店铺客服，简短自然，通常1句，最多2句；不要写长段解释，除非客户明确要求详细说明。",
            "不要使用emoji、Markdown标题、表格、加粗符号或列表式大段排版；直接按客服口吻回复客户。",
            "知识库或工具结果有明确答案时，直接短答，不要扩写营销话术，不要反复说抱歉、感谢理解、请放心等套话。",
            "不要每次都用“亲，您好”开头；同一会话里不要逐字重复同一句话。",
            "客户只说转人工、找人工、人工客服时，先尝试正常安抚并处理问题；只有确实需要人工执行动作、升级处理，或同一问题纠结超过3轮时，才转人工。",
            "售前、售中、售后都要按知识库和工具结果回答，不要自行编造功能、参数、赠品、时效、补偿或运费承担方案。",
            "商品未锁定时，不要猜商品；先询问客户要咨询哪一款商品。",
            "商品已锁定后，商品参数、功能、赠品、使用方法按知识库回答；没有明确答案就说暂未查询到，不要反复让客户看详情页。",
            "售后反馈要按当前售后规则和工具边界处理，必要时直接转人工，不要自行编补偿方案。",
            "不能承诺能帮客户改颜色、改备注、按备注发货、特殊包装、补发、退款金额、运费承担等平台或订单外动作。",
            "涉及退货运费时，不要说“运费险”，统一按退货包运费服务表达；是否赠送以当前商品知识和平台页面为准。",
            "不能出现第三方平台、导流、极限词、返现、隐私泄露等违规表达。",
            "客户问价格、质量、靠不靠谱时，可以统一用店铺既有口径，但不要夸大、不要编造机制。",
            "不要输出内部标签、思维过程、XML 标签或工具痕迹；只输出客户可见内容。",
            "不要编造商品机制或参数；没有知识库明确依据时，禁止说具体容量、功率、材质、使用时长、特殊功能或故障原因等内容。",
            "客户投诉已收到后无法使用、功能异常、声音大、破损、少件等问题时，不要继续泛泛解释，优先按售后场景处理。",
            "涉运费退款时，不要说“运费险”，退货包运费服务按当前商品知识和平台页面为准；售后优先按问题处理。",
            "如果同一客户在同一个问题上纠结3轮以上，优先转人工处理。",
            "有多个订单且状态不同，不能猜是哪一单，必须让客户发具体订单号。",
            "不得告诉客户具体到货日期，只能说大概几天能到，具体到货时间以实际物流为准。",
            "禁止提及晒图好评、好评返现、朋友圈、小红书、种草、发帖、返利、返现、红包、补偿换好评；本店没有晒图好评活动，不能引导客户去朋友圈或小红书发布内容。",
        ]
    },
    "agent": {
        "scene_prompt_files": dict(DEFAULT_SCENE_PROMPT_FILES),
        "high_risk_aftersale_transfer_phrases": [],
        "image_grounding_forbidden_symbols": [],
        "image_grounding_forbidden_functions": [],
        "version_name_tokens": [],
        "usage_feedback_keywords": [
            "好响",
            "声音大",
            "噪音大",
            "吵",
            "滋滋",
            "异响",
            "没反应",
            "坏了",
            "不转",
            "充不进",
            "充不上",
            "正在用",
            "收到货了",
            "已经收到",
            "已收到",
            "无法使用",
            "功能异常",
        ],
        "unreceived_patterns": [
            "还没收到",
            "还没有收到",
            "没收到",
            "没有收到",
            "未收到",
            "未收货",
            "没收货",
            "没有收货",
            "未签收",
        ],
        "missing_goods_parameter_keywords": [
            "商品参数",
            "参数",
            "规格",
            "型号",
            "款式",
            "尺寸",
            "尺码",
            "大小",
            "长度",
            "宽度",
            "高度",
            "重量",
            "容量",
            "功率",
            "电压",
            "材质",
            "面料",
            "成分",
            "保质期",
            "有效期",
            "生产日期",
            "怎么用",
            "使用方法",
            "安装",
            "配套",
            "配件",
            "赠品",
            "发几个",
            "按键",
            "按钮",
            "开关",
            "图标",
            "标识",
            "功能",
            "颜色",
            "库存",
        ],
        "daytime_night_mode_leak_markers": [
            "夜间时段",
            "夜间不转人工",
            "高级客服已下班",
            "高级客服下班了",
            "高级客服目前下班",
            "当前高级客服不在线",
            "还没上班",
            "上班时间是早上8点",
            "上班时间为早上8点",
            "高级客服上班时间",
            "建议您晚点联系",
            "建议您晚点再联系",
        ],
    },
    "reply_sanitizer": {
        "replacements": {
            "运费险": "退货包运费服务",
        },
        "internal_sentence_terms": [
            "知识库",
            "RAG",
            "rag",
            "预检索",
            "检索结果",
            "未提供明确数据",
            "未提供具体数据",
            "未找到相关",
        ],
    },
    "knowledge": KnowledgeConfig().model_dump(mode="json"),
    "pinduoduo": {
        "request": {
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
            "sec_ch_ua": '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
            "sec_ch_ua_mobile": "?0",
            "sec_ch_ua_platform": '"Windows"',
        },
        "transfer": {
            "default_remark": "客服升级处理",
        },
        "immediate_ack": {
            "enabled": True,
            "message": "[玫瑰]",
            "context_types": ["withdraw", "system_hint", "transfer"],
        },
        "order": {
            "signed_trace_keywords": [
                "包裹已签收",
                "包裹已签收！",
                "已签收",
                "快件已签收",
                "快件已签收，签收方式",
                "签收人是",
            ],
        }
    },
}


class ConfigError(Exception):
    """配置相关错误基类"""
    pass


class ConfigFileNotFoundError(ConfigError):
    """配置文件未找到错误"""
    pass


class ConfigParseError(ConfigError):
    """配置文件解析错误"""
    pass


class ConfigValidationError(ConfigError):
    """配置校验错误"""
    pass


class Config:
    """线程安全的配置管理器"""

    def __init__(
        self,
        config_path: Union[str, Path] = "config.json",
        auto_create: bool = True,
    ):
        self.config_path = Path(config_path)
        self.auto_create = auto_create
        self._lock = threading.RLock()
        self._config: Optional[Dict[str, Any]] = None
        self._validated_config: Optional[ConfigModel] = None
        self.reload()

    def _load_config(self) -> Dict[str, Any]:
        """从文件加载配置"""
        if not self.config_path.exists():
            raise ConfigFileNotFoundError(f"配置文件不存在: {self.config_path}")

        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                config_data = json.load(file)

            validated_config = ConfigModel(**config_data)
            self._validated_config = validated_config
            return config_data
        except json.JSONDecodeError as exc:
            raise ConfigParseError(f"配置文件格式错误: {exc}")
        except Exception as exc:
            raise ConfigValidationError(f"配置校验失败: {exc}")

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return copy.deepcopy(config_base)

    def _create_default_config_file(self) -> None:
        """创建默认配置文件"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as file:
                json.dump(config_base, file, ensure_ascii=False, indent=4)
            print(f"已创建默认配置文件: {self.config_path}")
        except Exception as exc:
            raise ConfigError(f"创建配置文件失败: {exc}")

    def reload(self) -> Dict[str, Any]:
        """重新加载配置文件"""
        with self._lock:
            try:
                self._config = self._load_config()
                return self._config
            except ConfigFileNotFoundError:
                if not self.auto_create:
                    raise
                self._create_default_config_file()
                self._config = self._default_config()
                self._validated_config = ConfigModel(**config_base)
                return self._config
            except Exception as exc:
                print(f"加载配置文件失败: {_sanitize_for_log(exc)}")
                self._config = self._default_config()
                self._validated_config = ConfigModel(**config_base)
                return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项，支持 llm.api_key 这种点号访问"""
        with self._lock:
            if self._config is None:
                return default

            try:
                value = self._config
                for part in key.split("."):
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return default
                return copy.deepcopy(value) if isinstance(value, (dict, list)) else value
            except Exception:
                return default

    def get_model(self) -> ConfigModel:
        """获取校验后的配置模型"""
        with self._lock:
            return self._validated_config or ConfigModel()

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            if self._config is None:
                return False
            value = self._config
            for part in key.split("."):
                if not isinstance(value, dict) or part not in value:
                    return False
                value = value[part]
            return True

    def set(self, key: str, value: Any, save: bool = True) -> Any:
        """设置配置项"""
        with self._lock:
            if self._config is None:
                self._config = self._default_config()

            original_config = copy.deepcopy(self._config)
            original_validated = copy.deepcopy(self._validated_config)
            candidate = copy.deepcopy(self._config)
            current = candidate
            parts = key.split(".")
            for part in parts[:-1]:
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]

            current[parts[-1]] = value

            try:
                validated_config = ConfigModel(**candidate)
                self._config = candidate
                self._validated_config = validated_config
                if save:
                    if not self.save():
                        self._config = original_config
                        self._validated_config = original_validated
                        raise ConfigError("保存配置文件失败")
            except ConfigError:
                raise
            except Exception as exc:
                raise ConfigValidationError(f"设置配置项失败: {exc}")

            return value

    def update(self, config_dict: Dict[str, Any], save: bool = False) -> Dict[str, Any]:
        """批量更新配置"""
        with self._lock:
            if self._config is None:
                self._config = self._default_config()

            original_config = copy.deepcopy(self._config)
            original_validated = copy.deepcopy(self._validated_config)
            merged_config = self._deep_merge(self._config, config_dict)

            try:
                self._validated_config = ConfigModel(**merged_config)
                self._config = merged_config
                if save:
                    if not self.save():
                        self._config = original_config
                        self._validated_config = original_validated
                        raise ConfigError("保存配置文件失败")
                return self._config
            except ConfigError:
                raise
            except Exception as exc:
                raise ConfigValidationError(f"批量更新配置失败: {exc}")

    def save(self) -> bool:
        """将当前配置原子写入文件"""
        with self._lock:
            if self._config is None:
                raise ConfigError("没有可保存的配置")

            temp_path = self.config_path.with_suffix(".tmp")
            try:
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, "w", encoding="utf-8") as file:
                    json.dump(self._config, file, ensure_ascii=False, indent=4)
                temp_path.replace(self.config_path)
                return True
            except Exception as exc:
                print(f"保存配置文件失败: {_sanitize_for_log(exc)}")
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception as cleanup_exc:
                    print(f"清理临时配置文件失败: {_sanitize_for_log(cleanup_exc)}")
                return False

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并字典"""
        result = base.copy()

        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    @contextmanager
    def atomic_update(self):
        """原子更新配置的上下文管理器"""
        import copy

        had_config = self._config is not None
        original_config = copy.deepcopy(self._config) if self._config else None
        original_validated = copy.deepcopy(self._validated_config)
        try:
            yield self
            if not self.save():
                raise ConfigError("保存配置文件失败")
        except Exception:
            self._config = original_config if had_config else None
            self._validated_config = original_validated
            raise


config = Config()


def get_config(key: str, default: Any = None) -> Any:
    """全局便捷函数：获取配置项"""
    return config.get(key, default)


def set_config(key: str, value: Any, save: bool = False) -> Any:
    """全局便捷函数：设置配置项"""
    return config.set(key, value, save)


def reload_config() -> Dict[str, Any]:
    """全局便捷函数：重新加载配置"""
    return config.reload()


def save_config() -> bool:
    """全局便捷函数：保存配置"""
    return config.save()


def update_config(config_dict: Dict[str, Any], save: bool = False) -> Dict[str, Any]:
    """全局便捷函数：批量更新配置"""
    return config.update(config_dict, save)


def get_validated_config() -> ConfigModel:
    """全局便捷函数：获取验证后的配置模型"""
    return config.get_model()
