"""
Unified knowledge search tool.

Searches the current product's scene knowledge. If no scene knowledge is
available, escalates the conversation to human support.
"""
from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, Field

from Agent.CustomerAgent.custom.knowledge_action_router import sanitize_formatted_knowledge
from Agent.CustomerAgent.custom.tool_decorator import agent_tool
from config import get_config
from core.di_container import container
from database.knowledge_service import KnowledgeService


SCENE_MAP = {
    "售前": "presale",
    "售中": "insale",
    "售后": "aftersale",
    "presale": "presale",
    "insale": "insale",
    "aftersale": "aftersale",
}

DEFAULT_SEARCH_KNOWLEDGE_QUERY_EXAMPLES = (
    "商品参数",
    "功能",
    "图片里的按键/图标/部件用途",
    "材质",
    "尺寸",
    "发货",
    "物流",
    "退换货",
    "售后处理",
)


def _knowledge_service() -> KnowledgeService:
    try:
        return container.get(KnowledgeService)
    except Exception:
        return KnowledgeService()


def _normalize_scene(scene: Optional[str], query: str) -> str:
    _ = query
    if scene:
        mapped = SCENE_MAP.get(str(scene).strip())
        if mapped:
            return mapped
    return "presale"


def _search_knowledge_query_examples() -> tuple[str, ...]:
    configured = get_config("agent.search_knowledge_query_examples", None)
    if configured is None:
        configured = DEFAULT_SEARCH_KNOWLEDGE_QUERY_EXAMPLES
    elif not isinstance(configured, (list, tuple)):
        configured = DEFAULT_SEARCH_KNOWLEDGE_QUERY_EXAMPLES
    return tuple(str(item or "").strip() for item in configured if str(item or "").strip())


def _search_knowledge_description() -> str:
    examples = _search_knowledge_query_examples()
    example_text = f"客户问{ '、'.join(examples) }等问题时使用。" if examples else "客户问题需要查询商品或店铺知识时使用。"
    return f"查询当前商品或店铺通用知识。{example_text}"


def _transfer_to_human(params: "SearchKnowledgeParams") -> str:
    from Agent.CustomerAgent.tools.move_conversation import TransferConversationParams, transfer_conversation

    return transfer_conversation(
        TransferConversationParams(
            shop_id=params.shop_id,
            user_id=params.user_id,
            recipient_uid=params.recipient_uid,
        )
    )


def _with_candidate_selection_context(
    *,
    query: str,
    scene_key: str,
    goods_id: Optional[int],
    formatted: str,
) -> str:
    context_lines = [
        "【知识候选使用要求】",
        f"客户当前问题：{query}",
        f"当前场景：{scene_key}",
    ]
    if goods_id:
        context_lines.append(f"当前商品ID：{goods_id}")
    context_lines.extend(
        [
            "先判断客户当前问题主题、场景和商品上下文，再选择能直接回答该主题的候选知识。",
            "不要只因为某条候选排序靠前就使用它；候选与当前问题主题冲突时，应忽略该候选并继续检索或转人工。",
            "不要向客户提到知识库、候选、排序、score、RAG、系统检索等内部信息。",
            "",
            formatted,
        ]
    )
    return "\n".join(str(line) for line in context_lines if str(line).strip() or line == "")


class SearchKnowledgeParams(BaseModel):
    """Unified knowledge search params."""

    query: str = Field(..., description="客户原始问题")
    shop_id: Union[str, int] = Field(..., description="店铺ID")
    user_id: Optional[Union[str, int]] = Field(None, description="当前客服账号ID，用于未命中时转人工")
    recipient_uid: Optional[str] = Field(None, description="客户UID，用于未命中时转人工")
    goods_id: Optional[int] = Field(None, description="当前商品ID")
    scene: Optional[str] = Field(None, description="售前/售中/售后")


@agent_tool(
    name="search_knowledge",
    description=_search_knowledge_description,
    param_model=SearchKnowledgeParams,
)
def search_knowledge(params: SearchKnowledgeParams) -> str:
    if not params.shop_id:
        return "[错误：缺少店铺ID，无法查询知识]"
    if not params.query:
        return "[错误：缺少客户问题，无法查询知识]"

    knowledge_service = _knowledge_service()
    scene_key = _normalize_scene(params.scene, params.query)

    if not params.goods_id:
        return _transfer_to_human(params)

    scene_results = knowledge_service.search_scene_knowledge(
        scene=scene_key,
        shop_id=params.shop_id,
        goods_id=params.goods_id,
        query=params.query,
        limit=2,
    )
    if not scene_results:
        return _transfer_to_human(params)

    formatted = sanitize_formatted_knowledge(knowledge_service.format_scene_results(scene_results))
    return _with_candidate_selection_context(
        query=params.query,
        scene_key=scene_key,
        goods_id=params.goods_id,
        formatted=formatted,
    )
