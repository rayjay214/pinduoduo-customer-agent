"""Shared prompt rule builders."""
from __future__ import annotations

from config import get_config


DEFAULT_IMAGE_GROUNDING_SYMBOLS = ()
DEFAULT_IMAGE_GROUNDING_FUNCTIONS = ()
DEFAULT_VERSION_NAME_TOKENS = ()


def _configured_terms(key: str, defaults: tuple[str, ...]) -> tuple[str, ...]:
    configured = get_config(key, None)
    if configured is None:
        configured = defaults
    elif not isinstance(configured, (list, tuple)):
        configured = defaults

    terms = []
    for item in configured:
        text = str(item or "").strip()
        if text:
            terms.append(text)
    return tuple(terms)


def image_grounding_symbols() -> tuple[str, ...]:
    return _configured_terms(
        "agent.image_grounding_forbidden_symbols",
        DEFAULT_IMAGE_GROUNDING_SYMBOLS,
    )


def image_grounding_functions() -> tuple[str, ...]:
    return _configured_terms(
        "agent.image_grounding_forbidden_functions",
        DEFAULT_IMAGE_GROUNDING_FUNCTIONS,
    )


def version_name_tokens() -> tuple[str, ...]:
    return _configured_terms(
        "agent.version_name_tokens",
        DEFAULT_VERSION_NAME_TOKENS,
    )


def _join_terms(terms: tuple[str, ...]) -> str:
    return "、".join(terms)


def build_image_grounding_instruction() -> str:
    symbols = _join_terms(image_grounding_symbols())
    functions = _join_terms(image_grounding_functions())
    if not symbols or not functions:
        return ""
    return (
        "收到图片时，图片只能作为可见内容辅助，不能把图片里的 "
        f"{symbols} 等符号自行解释成 {functions} 等商品功能；"
        "看不清或无法确定图片问题时，简短询问客户具体问题，不要编造图片细节。"
    )


def build_image_grounding_constraint() -> str:
    symbols = _join_terms(image_grounding_symbols())
    functions = _join_terms(image_grounding_functions())
    if not symbols or not functions:
        return ""
    return (
        "【图片理解硬约束】\n"
        "图片只能作为客户现场/截图的辅助信息，不是商品功能依据。\n"
        f"禁止根据图片里的 {symbols} 等视觉符号，推断 {functions} 等商品功能。\n"
        "客户问图中按钮、图标、标识、部件用途时，必须以预检索知识或 search_knowledge 的明确答案为准；"
        "没有明确答案时回复：亲，仅凭图片看不出这个位置的具体功能，麻烦您说下是哪个按键/位置，或点一下商品卡片，我按对应款式帮您确认哦。\n"
        "不要说“X 是摇头功能”等没有知识依据的结论。\n"
        "不要向客户提到图片理解硬约束、预检索、search_knowledge、知识库等内部信息。\n"
    )


def build_version_name_instruction() -> str:
    tokens = _join_terms(version_name_tokens())
    if not tokens:
        return ""
    return (
        f"版本名约束：{tokens} 等是商品版本/规格名称，"
        "不等于真实电池毫安容量。禁止把版本名说成对应毫安容量。"
        "如果候选知识或商品信息同时出现版本名和容量/时长，优先把版本名当版本名解释。"
        "如客户问电池容量，按知识库实际数值回答；知识库无明确数据时回复'具体容量以页面当前规格标注为准'。"
    )
