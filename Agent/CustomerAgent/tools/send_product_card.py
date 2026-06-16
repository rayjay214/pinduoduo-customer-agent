"""
Unified product-card tool.

If goods_id is known, send that product card directly. Otherwise fetch product
candidates and return them for the model to ask the customer to choose.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Optional, Union

from pydantic import BaseModel, Field

from Agent.CustomerAgent.custom.tool_decorator import agent_tool
from Channel.pinduoduo.utils.API.product_manager import ProductManager
from Channel.pinduoduo.utils.API.send_message import SendMessage
from core.base_service import _sanitize_for_log
from utils.config_values import as_bool, as_int
from utils.logger_loguru import get_logger

logger = get_logger("SendProductCardTool")
DEFAULT_CANDIDATE_PAGE_SIZE = 10


def _sanitize_external_message(message: str) -> str:
    return _sanitize_for_log(str(message or ""))


class SendProductCardParams(BaseModel):
    """Send product card params."""

    shop_id: Optional[Union[str, int]] = Field(default=None, description="店铺ID")
    user_id: Optional[Union[str, int]] = Field(default=None, description="用户ID（账号ID）")
    recipient_uid: Optional[str] = Field(default=None, description="接收商品卡片的用户UID")
    goods_id: Optional[int] = Field(default=None, description="当前商品ID；已锁定商品时传入")
    candidate_index: Optional[int] = Field(default=None, description="候选商品列表序号，从1开始；只有用户明确选择候选时传入")
    query: Optional[str] = Field(default=None, description="客户原始问题，用于判断是否需要候选商品")


def _format_products_output(products: list, total: int) -> str:
    if not products:
        return "未找到可推荐商品。"

    lines = [f"可推荐商品列表（共{total}个，以下为前{len(products)}个）："]
    for index, product in enumerate(products, 1):
        goods_id = product.get("goods_id", "")
        goods_name = product.get("goods_name", "未命名商品")
        price = product.get("price", "")
        item = f"{index}. 商品名称：{goods_name}\n   商品ID：{goods_id}"
        if price:
            item += f"\n   价格：{price}元"
        lines.append(item)
    lines.append("如果客户没有明确选择哪一款，请先询问客户要哪一款，不要随便发送商品卡片。")
    return "\n\n".join(lines)


def _load_products(shop_id: Union[str, int], user_id: Union[str, int]) -> tuple[list, int, str]:
    product_manager = ProductManager(shop_id=shop_id, user_id=user_id)
    result = product_manager.get_product_list(page=1, size=DEFAULT_CANDIDATE_PAGE_SIZE)
    if not isinstance(result, dict):
        return [], 0, f"商品列表响应格式异常: {type(result).__name__}"
    if not as_bool(result.get("success"), False):
        return [], 0, _sanitize_external_message(result.get("error_msg", "获取商品列表失败"))
    products = result.get("products", [])
    if not isinstance(products, list):
        products = []
    products = [product for product in products if isinstance(product, Mapping)]
    total = as_int(result.get("total"), len(products))
    return products, total, ""


def _send_card(shop_id: Union[str, int], user_id: Union[str, int], recipient_uid: str, goods_id: int) -> str:
    sender = SendMessage(str(shop_id), str(user_id))
    result = sender.send_mallGoodsCard(recipient_uid, goods_id, biz_type=2)
    if isinstance(result, dict) and as_bool(result.get("success"), False):
        logger.info(
            f"商品卡片发送成功: goods_id={goods_id}, recipient_uid={recipient_uid}, shop_id={shop_id}"
        )
        return "商品卡片发送成功"

    error_msg = result.get("error_msg", "发送失败") if isinstance(result, dict) else "发送失败"
    safe_error_msg = _sanitize_external_message(error_msg)
    logger.error(f"商品卡片发送失败: {safe_error_msg}, goods_id={goods_id}, recipient_uid={recipient_uid}")
    return f"商品卡片发送失败: {safe_error_msg}"


def _coerce_goods_id(value) -> Optional[int]:
    try:
        goods_id = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return goods_id if goods_id > 0 else None


def _coerce_candidate_index(value) -> Optional[int]:
    try:
        candidate_index = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return candidate_index if candidate_index > 0 else None


def _send_candidate_index(
    shop_id: Union[str, int],
    user_id: Union[str, int],
    recipient_uid: str,
    candidate_index: int,
) -> str:
    products, total, error_msg = _load_products(shop_id, user_id)
    if error_msg:
        return f"获取商品列表失败：{error_msg}"
    if candidate_index < 1 or candidate_index > len(products):
        return f"发送商品卡片失败：候选序号 {candidate_index} 超出范围，当前可选 1-{len(products)}。"

    product = products[candidate_index - 1]
    selected_goods_id = _coerce_goods_id(product.get("goods_id"))
    if selected_goods_id is None:
        return f"发送商品卡片失败：候选序号 {candidate_index} 的商品ID格式异常。"
    return _send_card(shop_id, user_id, recipient_uid, selected_goods_id)


def _ambiguous_candidate_index_message(
    goods_id: int,
    products: list,
) -> str:
    product = products[goods_id - 1]
    selected_goods_id = product.get("goods_id")
    return (
        f"发送失败：goods_id={goods_id} 像候选列表序号，不是真实商品ID；"
        f"如需发送第 {goods_id} 个候选，请使用 candidate_index={goods_id}，"
        f"或传真实商品ID：{selected_goods_id}。"
    )


def _goods_id_is_ambiguous_candidate_index(
    shop_id: Union[str, int],
    user_id: Union[str, int],
    goods_id: int,
) -> tuple[bool, str]:
    if goods_id < 1 or goods_id > DEFAULT_CANDIDATE_PAGE_SIZE:
        return False, ""

    products, _total, error_msg = _load_products(shop_id, user_id)
    if error_msg:
        return True, f"发送失败：goods_id={goods_id} 处在候选序号范围内，但获取商品列表失败，无法确认真实商品ID：{error_msg}"

    actual_ids = {_coerce_goods_id(product.get("goods_id")) for product in products}
    if goods_id in actual_ids:
        return False, ""
    if goods_id <= len(products):
        return True, _ambiguous_candidate_index_message(goods_id, products)
    return True, f"发送失败：goods_id={goods_id} 处在候选序号范围内，但不在当前候选商品ID中；请传真实商品ID。"


@agent_tool(
    name="send_product_card",
    description="发送当前商品卡片；goods_id 必须是真实商品ID。用户选择候选列表第几项时传 candidate_index。",
    param_model=SendProductCardParams,
)
def send_product_card(params: SendProductCardParams) -> str:
    if not params.shop_id or not params.user_id:
        return "处理商品卡片失败：缺少 shop_id 或 user_id。"

    recipient_uid = str(params.recipient_uid or "").strip()
    goods_id = params.goods_id
    candidate_index = params.candidate_index

    if candidate_index is not None:
        if not recipient_uid:
            return "发送商品卡片失败：缺少 recipient_uid。"
        normalized_candidate_index = _coerce_candidate_index(candidate_index)
        if normalized_candidate_index is None:
            return "发送商品卡片失败：候选序号格式异常。"
        return _send_candidate_index(
            params.shop_id,
            params.user_id,
            recipient_uid,
            normalized_candidate_index,
        )

    if goods_id:
        if not recipient_uid:
            return "发送商品卡片失败：缺少 recipient_uid。"
        normalized_goods_id = int(goods_id)
        is_ambiguous, ambiguity_message = _goods_id_is_ambiguous_candidate_index(
            params.shop_id,
            params.user_id,
            normalized_goods_id,
        )
        if is_ambiguous:
            return ambiguity_message
        return _send_card(params.shop_id, params.user_id, recipient_uid, normalized_goods_id)

    products, total, error_msg = _load_products(params.shop_id, params.user_id)
    if error_msg:
        return f"获取商品列表失败：{error_msg}"

    if len(products) == 1 and recipient_uid:
        only_goods_id = products[0].get("goods_id")
        if only_goods_id:
            try:
                return _send_card(params.shop_id, params.user_id, recipient_uid, int(only_goods_id))
            except (TypeError, ValueError, OverflowError):
                logger.warning(f"候选商品ID格式异常，不能自动发送商品卡片: goods_id={only_goods_id}")

    return _format_products_output(products, total)
