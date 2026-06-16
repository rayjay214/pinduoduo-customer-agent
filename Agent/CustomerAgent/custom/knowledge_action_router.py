"""Customer service knowledge action routing and sanitization."""
from __future__ import annotations

import re
from collections.abc import Mapping

from Agent.CustomerAgent.custom.prompt_rules import version_name_tokens
from config import get_config


INTERNAL_ACTION_TERMS = (
    "转人工",
    "人工客服",
)

DIRECT_TRANSFER_REPLIES = {
    "转人工",
    "转人工处理",
    "联系人工",
    "联系人工客服",
}
TRANSFER_ACTION_PREFIXES = (
    "需要",
    "建议",
    "请",
    "帮忙",
    "帮我",
    "联系",
    "转接",
)

SAFE_TRANSFER_REPLY = "亲，已转人工为您处理，请稍等。"
DEFAULT_REPLY_REPLACEMENTS = {
    "运费险": "退货包运费服务",
}

DEFAULT_INTERNAL_SENTENCE_TERMS = (
    "知识库",
    "RAG",
    "rag",
    "预检索",
    "检索结果",
    "未提供明确数据",
    "未提供具体数据",
    "未找到相关",
)

_SAFE_FALLBACK_REPLY = "亲，这边帮您确认一下，稍后回复您哦。"


def _normalize_action_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip("。？！?!，,")


def _is_transfer_action_answer(compact_text: str) -> bool:
    direct = {_normalize_action_text(item) for item in DIRECT_TRANSFER_REPLIES}
    if compact_text in direct:
        return True
    if not any(term in compact_text for term in INTERNAL_ACTION_TERMS):
        return False
    return compact_text.startswith(TRANSFER_ACTION_PREFIXES)


def _strip_internal_parentheses(text: str) -> str:
    result = str(text or "")
    for open_p, close_p in (("（", "）"), ("(", ")")):
        pattern = rf"\{open_p}[^\{open_p}\{close_p}]*(?:转人工|人工客服)[^\{open_p}\{close_p}]*\{close_p}"
        result = re.sub(pattern, "", result)
    return result


def sanitize_customer_service_text(text: str) -> str:
    """Remove internal action hints from knowledge text."""
    result = _strip_internal_parentheses(text)

    compact = _normalize_action_text(result)
    if _is_transfer_action_answer(compact):
        return SAFE_TRANSFER_REPLY

    result = re.sub(r"\s+", " ", result).strip()
    return result


def sanitize_formatted_knowledge(formatted_knowledge: str) -> str:
    """Clean the rendered knowledge text before it reaches the model."""
    text = str(formatted_knowledge or "")
    if not text.strip():
        return text

    def replace_answer(match: re.Match[str]) -> str:
        answer = match.group(1)
        suffix = match.group(2)
        safe_answer = sanitize_customer_service_text(answer)
        return f"{safe_answer}{suffix}"

    return re.sub(
        r"标准答案[:：]\s*(.+?)(\n\s*\n|\Z)",
        replace_answer,
        text,
        flags=re.S,
    ).strip()


def _remove_sentences_with_internal_terms(text: str) -> str:
    """删除含内部术语的文本片段，保留其余客户可见内容。"""
    internal_terms = _get_internal_sentence_terms()
    if not internal_terms:
        return text

    # 按自然标点分片，避免“内部说明，客户可见回复。”被整句误删。
    sentences = re.split(r"(?<=[，,；;。！？!?])", text)
    kept = []
    for s in sentences:
        s_stripped = s.strip()
        if not s_stripped:
            continue
        if any(term in s_stripped for term in internal_terms):
            continue
        kept.append(s_stripped)
    result = "".join(kept)
    result = re.sub(r"^[，,；;、\s]+", "", result)
    result = re.sub(r"[，,；;、\s]+$", "", result)
    return result


def _get_internal_sentence_terms() -> tuple[str, ...]:
    configured = get_config(
        "reply_sanitizer.internal_sentence_terms",
        list(DEFAULT_INTERNAL_SENTENCE_TERMS),
    )
    if configured is None:
        return ()
    if not isinstance(configured, (list, tuple)):
        configured = DEFAULT_INTERNAL_SENTENCE_TERMS

    terms = []
    for item in configured:
        text = str(item or "").strip()
        if text:
            terms.append(text)
    return tuple(terms)


def _fix_version_name_hallucination(text: str) -> str:
    """修正版本名幻觉：配置里的版本名不等于毫安容量。"""
    pattern = _version_name_hallucination_re()
    if pattern is None or not pattern.search(text):
        return text
    # 按句处理：含版本名幻觉的整句替换为安全表述
    sentences = re.split(r"(?<=[。！？!?])", text)
    kept = []
    for s in sentences:
        s_stripped = s.strip()
        if not s_stripped:
            continue
        if pattern.search(s_stripped):
            kept.append("具体容量以页面当前规格标注为准")
        else:
            kept.append(s_stripped)
    return "".join(kept)


def _version_name_hallucination_re() -> re.Pattern[str] | None:
    tokens = tuple(dict.fromkeys(version_name_tokens()))
    if not tokens:
        return None
    token_pattern = "|".join(re.escape(token) for token in tokens)
    return re.compile(
        rf"(?:{token_pattern})(?=$|\s|[，,。！？!?是为=：:])[^。！？!?]{{0,30}}(?:毫安|mAh|MAH|mah)",
        re.IGNORECASE,
    )


def _get_reply_replacements() -> dict[str, str]:
    configured = get_config("reply_sanitizer.replacements", DEFAULT_REPLY_REPLACEMENTS)
    if configured is None:
        return {}
    if not isinstance(configured, Mapping):
        return dict(DEFAULT_REPLY_REPLACEMENTS)

    replacements: dict[str, str] = {}
    for forbidden, replacement in configured.items():
        forbidden_text = str(forbidden or "")
        if not forbidden_text:
            continue
        replacements[forbidden_text] = "" if replacement is None else str(replacement)
    return replacements


def sanitize_final_reply(reply: str) -> str:
    """Final safety pass before sending text to the customer."""
    raw = str(reply or "")
    if _is_transfer_action_answer(_normalize_action_text(raw)):
        return SAFE_TRANSFER_REPLY

    text = _strip_internal_parentheses(raw)
    for forbidden, replacement in _get_reply_replacements().items():
        text = text.replace(forbidden, replacement)

    # 内部术语句删除
    text = _remove_sentences_with_internal_terms(text)

    # 版本名幻觉修正
    text = _fix_version_name_hallucination(text)

    text = re.sub(r"\s+", " ", text).strip()

    # 清空后兜底
    if not text or len(text) < 2:
        return _SAFE_FALLBACK_REPLY

    return text
