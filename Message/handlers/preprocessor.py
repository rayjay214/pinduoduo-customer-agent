"""
消息预处理器
提取和优化消息预处理逻辑
"""

import json
from typing import Dict, Any, Optional
from utils.logger_loguru import get_logger
from bridge.context import ContextType
from Channel.pinduoduo.utils.base_request import BaseRequest


logger = get_logger(__name__)


def _sanitize_for_log(value: Any) -> Any:
    if isinstance(value, BaseException):
        text = str(value)
        return f"{type(value).__name__}: {BaseRequest()._sanitize_for_log(text)}" if text else type(value).__name__
    return BaseRequest()._sanitize_for_log(value)


class MessagePreprocessor:
    """消息预处理器 - 提取通用逻辑"""

    @staticmethod
    def _is_meaningful_aftersale_status(value: Any) -> bool:
        if value in (None, "", 0, False):
            return False
        text = str(value).strip()
        if not text:
            return False
        return text.casefold() not in {"0", "false", "none", "null", "无", "无售后", "未申请", "未发起"}

    @classmethod
    def _extract_aftersale_status(cls, data: Dict[str, Any]) -> Optional[str]:
        for key in (
            "afterSalesStatusStr",
            "afterSaleStatusStr",
            "after_sales_status",
            "afterSalesStatus",
            "afterSaleStatus",
            "refundStatusStr",
            "refund_status",
        ):
            value = data.get(key)
            if cls._is_meaningful_aftersale_status(value):
                return str(value).strip()

        for key in ("afterSalesInfo", "afterSaleInfo", "refundInfo", "afterSale"):
            info = data.get(key)
            if not isinstance(info, dict):
                continue
            for nested_key in ("statusDesc", "statusStr", "status", "title", "desc"):
                value = info.get(nested_key)
                if cls._is_meaningful_aftersale_status(value):
                    return str(value).strip()
        return None

    @staticmethod
    def _is_refunded_status(status: Optional[str]) -> bool:
        text = str(status or "")
        lowered = text.casefold()
        return any(token in text or token in lowered for token in ("退款成功", "已退款", "refunded"))

    @classmethod
    def _resolve_order_status_label(cls, data: Dict[str, Any]) -> Optional[str]:
        """将订单状态码翻译成稳定的人类可读状态。"""
        aftersale_status = cls._extract_aftersale_status(data)
        if aftersale_status:
            if cls._is_refunded_status(aftersale_status):
                return "已退款"
            return f"售后中（{aftersale_status}）"

        shipping_status = data.get("shipping_status")
        status = data.get("status")

        try:
            shipping_status = int(shipping_status) if shipping_status is not None else None
        except (TypeError, ValueError, OverflowError):
            shipping_status = None

        try:
            status = int(status) if status is not None else None
        except (TypeError, ValueError, OverflowError):
            status = None

        if shipping_status == 0 and status == 2:
            return "待发货"
        if shipping_status == 1 and status == 3:
            return "已发货待收货"
        if shipping_status == 2 and status == 4:
            return "已签收"

        raw_status = data.get("status")
        if isinstance(raw_status, str) and raw_status.strip() and not raw_status.strip().isdigit():
            text = raw_status.strip()
            if cls._is_refunded_status(text):
                return "已退款"
            return text
        return None

    @classmethod
    def _resolve_order_scene_label(cls, data: Dict[str, Any]) -> Optional[str]:
        """将订单状态翻译成更稳定的业务场景标签。"""
        if cls._extract_aftersale_status(data):
            return "售后倾向"

        order_status_label = cls._resolve_order_status_label(data)
        if order_status_label == "待发货":
            return "售中-待发货"
        if order_status_label == "已发货待收货":
            return "售中-物流中"
        if order_status_label in {"已签收", "已退款"}:
            return "售后倾向"
        return None

    @staticmethod
    def _normalize_standard_message_item(item: Dict[str, Any]) -> Dict[str, Any]:
        kind = item.get("type")
        if kind == "text":
            return {"raw_content": item.get("text", "")}
        if kind == "image":
            return {"raw_content": "[图片消息]", "image_url": item.get("url", "")}
        if kind == "video":
            return {"raw_content": "[视频消息]", "video_url": item.get("url", ""), "cover": item.get("cover", "")}
        if kind == "goods_card":
            return {
                "goods_name": item.get("goods_name") or item.get("name"),
                "goods_price": item.get("goods_price") or item.get("price"),
                "goods_id": item.get("goods_id"),
                "thumb_url": item.get("thumb_url"),
                **item,
            }
        if kind == "order_info":
            return {
                "order_id": item.get("order_id") or item.get("order_sn"),
                "goods_name": item.get("goods_name"),
                **item,
            }
        return item

    @classmethod
    def _normalize_standard_message_list(cls, items: list) -> Dict[str, Any]:
        """合并标准消息数组，避免只取第一项导致卡片/订单上下文丢失。"""
        merged: Dict[str, Any] = {}
        raw_parts = []

        for item in items:
            if not isinstance(item, dict):
                if item is not None and str(item).strip():
                    raw_parts.append(str(item))
                continue

            normalized = cls._normalize_standard_message_item(item)
            raw_content = normalized.pop("raw_content", None)
            if raw_content is not None and str(raw_content).strip():
                raw_parts.append(str(raw_content).strip())

            for key, value in normalized.items():
                if value is None:
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                if key not in merged or merged.get(key) in (None, ""):
                    merged[key] = value

        if raw_parts:
            merged["raw_content"] = "\n".join(raw_parts)
        return merged

    @classmethod
    def safe_parse_json(cls, data: Any, default_structure: Dict[str, str] = None) -> Dict[str, str]:
        """安全解析JSON，统一处理各种消息格式"""
        if default_structure is None:
            default_structure = {}

        if not data:
            return default_structure

        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list):
                    normalized = cls._normalize_standard_message_list(parsed)
                    return normalized or {"raw_content": data, **default_structure}
                return {"raw_content": data, **default_structure}
            except json.JSONDecodeError:
                return {"raw_content": data, **default_structure}
        elif isinstance(data, dict):
            return data
        else:
            return {"raw_content": str(data), **default_structure}

    @staticmethod
    def create_text_message(text: str) -> str:
        """创建标准文本消息格式"""
        return json.dumps([{"type": "text", "text": text}], ensure_ascii=False)

    @staticmethod
    def create_image_message(url: str) -> str:
        """创建标准图片消息格式"""
        return json.dumps([{"type": "image", "url": url}], ensure_ascii=False)

    @staticmethod
    def create_video_message(url: str, cover: str = None) -> str:
        """创建标准视频消息格式"""
        data = {"type": "video", "url": url}
        if cover:
            data["cover"] = cover
        return json.dumps([data], ensure_ascii=False)

    @staticmethod
    def create_goods_message(name: str, price: str = None, thumb_url: str = None, goods_id: str = None, **kwargs) -> str:
        """创建标准商品消息格式"""
        data = {"type": "goods_card", "name": name}
        if price: data["price"] = price
        if thumb_url: data["thumb_url"] = thumb_url
        if goods_id: data["goods_id"] = goods_id
        data.update(kwargs)
        return json.dumps([data], ensure_ascii=False)

    @staticmethod
    def create_order_message(order_sn: str, status: str = None, goods_name: str = None, **kwargs) -> str:
        """创建标准订单消息格式"""
        data = {"type": "order_info", "order_sn": order_sn}
        if status: data["status"] = status
        if goods_name: data["goods_name"] = goods_name
        data.update(kwargs)
        return json.dumps([data], ensure_ascii=False)

    def process(self, content: str, msg_type: Optional[ContextType] = None) -> str:
        """统一的消息预处理"""
        try:
            if content is None:
                return ""

            # 根据消息类型进行特定处理
            if msg_type == ContextType.IMAGE:
                return "[图片消息]"
            elif msg_type == ContextType.VIDEO:
                return "[视频消息]"

            # 1. 尝试解析为JSON
            parsed = self.safe_parse_json(content)

            if isinstance(parsed, dict):
                # 2. 如果是字典，提取关键信息，直接返回纯文本
                processed = self._extract_key_info(parsed)
                if processed:
                    return processed

            # 3. 清理文本，直接返回纯文本
            cleaned = self._clean_text(content)
            return cleaned

        except Exception as e:
            logger.error(f"Message preprocessing failed: {_sanitize_for_log(e)}")
            return "消息处理失败"

    def _extract_key_info(self, data: Dict[str, Any]) -> str:
        """提取关键信息"""
        if not isinstance(data, dict):
            return ""

        parts = []

        # 商品信息
        goods_name = data.get('goods_name') or data.get('name')
        if goods_name:
            parts.append(f"商品：{goods_name}")

        goods_price = data.get('goods_price') or data.get('price')
        if goods_price:
            parts.append(f"价格：{goods_price}")

        goods_spec = data.get('goods_spec') or data.get('spec')
        if goods_spec:
            parts.append(f"规格：{goods_spec}")

        # 商品ID - 保留用于知识库查询
        goods_id = data.get('goods_id')
        if goods_id:
            parts.append(f"商品ID：{goods_id}")

        # 订单信息
        order_id = data.get('order_id') or data.get('order_sn') or data.get('orderSn') or data.get('orderSequenceNo') or data.get('order')
        if order_id:
            parts.append(f"订单：{order_id}")

        aftersale_status = self._extract_aftersale_status(data)
        if aftersale_status:
            parts.append(f"售后状态：{aftersale_status}")

        order_status_label = self._resolve_order_status_label(data)
        status = data.get('status')
        if order_status_label:
            parts.append(f"当前订单状态：{order_status_label}")

        order_scene_label = self._resolve_order_scene_label(data)
        if order_scene_label:
            parts.append(f"当前业务场景：{order_scene_label}")

        order_status = data.get('order_status')
        if order_status is not None:
            parts.append(f"订单主状态码：{order_status}")

        shipping_status = data.get('shipping_status')
        if shipping_status is not None:
            parts.append(f"物流状态码：{shipping_status}")

        pay_status = data.get('pay_status')
        if pay_status is not None:
            parts.append(f"支付状态码：{pay_status}")

        if status is not None and (not isinstance(status, str) or status.strip().isdigit()):
            parts.append(f"订单状态码：{status}")

        tracking_number = data.get('tracking_number')
        if tracking_number is not None:
            parts.append(f"快递单号：{tracking_number or '空'}")

        shipping_time = data.get('shipping_time')
        if shipping_time is not None:
            parts.append(f"物流时间：{shipping_time}")

        # 原始内容
        raw_content = data.get('raw_content')
        if raw_content:
            parts.append(f"内容：{raw_content}")

        # 如果没有提取到有效信息，返回原始内容
        if not parts and 'raw_content' in data:
            return str(data['raw_content'] or "")

        return "，".join(parts) if parts else ""

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        if not isinstance(text, str):
            return str(text)

        # 移除多余的空白字符
        cleaned = ' '.join(text.split())

        # 限制长度（避免过长的消息）
        if len(cleaned) > 500:
            cleaned = cleaned[:500] + "..."

        return cleaned
