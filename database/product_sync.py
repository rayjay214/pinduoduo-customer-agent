"""
产品知识自动同步服务
=================

从拼多多API拉取商品列表，将商品基础信息同步到知识库。
"""
import asyncio
import threading
from typing import Optional, Callable, List, Dict, Any, Union
from dataclasses import dataclass
import json

from openai import AsyncOpenAI
from config import get_config
from utils.config_values import as_bool, as_int
from core.base_service import _sanitize_for_log

from Channel.pinduoduo.utils.API.product_manager import ProductManager
from database.knowledge_service import KnowledgeService
from utils.logger_loguru import get_logger

logger = get_logger("ProductSync")


def _error_log_summary(error_msg: Any) -> str:
    return f"error_chars={len(str(error_msg or ''))}"


@dataclass
class SyncProgress:
    """同步进度"""
    total: int
    current: int
    success: int
    failed: int
    current_goods_name: str
    cancelled: bool = False
    phase: str = "fetching"  # "fetching": 抓取商品列表, "extracting": 提取知识


def _normalize_product_list_result(result: Any) -> Dict[str, Any]:
    """Normalize ProductManager.get_product_list output before sync logic uses it."""
    if not isinstance(result, dict):
        return {"success": False, "total": 0, "products": [], "error_msg": "商品列表响应格式异常"}

    products = result.get("products") or []
    if not isinstance(products, list):
        products = []
    products = [product for product in products if isinstance(product, dict)]

    return {
        "success": as_bool(result.get("success"), False),
        "total": max(0, as_int(result.get("total"), len(products))),
        "products": products,
        "error_msg": str(result.get("error_msg") or "获取商品列表失败"),
    }


def _extract_llm_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("LLM response missing choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if content is None:
        raise RuntimeError("LLM response missing message content")
    return str(content).strip()


class ProductSyncService:
    """产品知识自动同步服务"""

    def __init__(
        self,
        knowledge_service: KnowledgeService,
        request_delay: float = 1.0,
    ):
        """
        初始化

        Args:
            knowledge_service: 知识库服务实例
            request_delay: API请求间隔（秒），避免限流
        """
        self.knowledge_service = knowledge_service
        self.request_delay = request_delay
        self._cancellation_event = threading.Event()
        logger.info("ProductSyncService 初始化成功")

    def cancel(self) -> None:
        """取消同步"""
        self._cancellation_event.set()
        logger.info("同步已取消")

    def is_cancelled(self) -> bool:
        """检查是否已取消"""
        return self._cancellation_event.is_set()

    def _reset_cancellation(self) -> None:
        """重置取消事件"""
        self._cancellation_event.clear()

    async def sync_shop(
        self,
        shop_id: Union[str, int],
        shop_db_id: int,
        user_id: str,
        is_full_sync: bool = False,
        progress_callback: Optional[Callable[[SyncProgress], None]] = None,
    ) -> SyncProgress:
        """
        同步店铺商品基础信息。

        这里只抓取商品ID、名称、价格、销量、缩略图等基础字段并保存到数据库。
        结构化客服知识由人工导入或独立流程生成，避免商品同步时隐式调用 LLM。

        Args:
            shop_id: 店铺ID（拼多多的shop_id）
            shop_db_id: 店铺在数据库中的ID
            user_id: 用户ID（用于拼多多API认证）
            is_full_sync: True=全量同步，False=增量同步（仅同步本地不存在的商品）
            progress_callback: 进度回调，每次更新进度调用

        Returns:
            最终同步进度
        """
        self._reset_cancellation()
        pm = ProductManager(shop_id=str(shop_id), user_id=user_id)

        # ================== 第一阶段：快速抓取商品列表 ==================
        logger.info("=== 第一阶段：开始抓取商品列表 ===")

        # 第一页获取总数量
        first_page = _normalize_product_list_result(pm.get_product_list(page=1, size=20))
        if not first_page["success"]:
            logger.error(f"获取商品列表失败: {_error_log_summary(first_page.get('error_msg'))}")
            progress = SyncProgress(
                total=0, current=0, success=0, failed=0, current_goods_name="",
                phase="fetching"
            )
            progress.failed = 1
            return progress

        total = first_page["total"]
        logger.info(f"店铺共有 {total} 个商品，开始抓取商品列表...")

        progress = SyncProgress(
            total=total,
            current=0,
            success=0,
            failed=0,
            current_goods_name="",
            phase="fetching"
        )

        # 分页拉取所有商品
        current_page = 1
        all_products: List[Dict[str, Any]] = []

        while True:
            if self.is_cancelled():
                progress.cancelled = True
                logger.info("同步已被用户取消")
                break

            page_result = _normalize_product_list_result(pm.get_product_list(page=current_page, size=50))
            if not page_result["success"]:
                logger.error(f"获取第 {current_page} 页失败: {_error_log_summary(page_result.get('error_msg'))}")
                break

            products = page_result["products"]
            if not products:
                break

            all_products.extend(products)
            current_page += 1

            # 更新进度
            progress.current = len(all_products)
            if all_products:
                progress.current_goods_name = all_products[-1].get("goods_name", "")
            if progress_callback:
                progress_callback(progress)

            # 延迟避免限流
            await asyncio.sleep(self.request_delay)

        if self.is_cancelled():
            return progress

        logger.info(f"第一阶段完成：共获取 {len(all_products)} 个商品")

        # 增量同步筛选：只处理本地不存在的商品
        products_to_process: List[Dict[str, Any]] = []
        if not is_full_sync:
            original_count = len(all_products)
            filtered_products: List[Dict[str, Any]] = []
            for p in all_products:
                goods_id = p.get("goods_id")
                existing = self.knowledge_service.get_product_by_goods_id(shop_db_id, goods_id)
                if not existing:
                    filtered_products.append(p)
            logger.info(f"增量同步: 总商品 {original_count}，需要同步 {len(filtered_products)} 个（已存在跳过）")
            products_to_process = filtered_products
        else:
            products_to_process = all_products

        # ================== 第一阶段B：快速保存商品基本信息 ==================
        logger.info("=== 开始快速保存商品基本信息 ===")
        progress.phase = "saving_basic"
        progress.total = len(products_to_process)
        progress.current = 0
        progress.success = 0
        progress.failed = 0

        for idx, product in enumerate(products_to_process):
            if self.is_cancelled():
                progress.cancelled = True
                break

            goods_id = product.get("goods_id")
            goods_name = product.get("goods_name", f"goods_{goods_id}")
            progress.current = idx + 1
            progress.current_goods_name = goods_name

            try:
                # 先只保存基本信息，不调用LLM
                self.knowledge_service.add_or_update_product(
                    shop_id=shop_db_id,
                    goods_id=goods_id,
                    goods_name=goods_name,
                    price=product.get("price"),
                    price_min=product.get("price_min"),
                    price_max=product.get("price_max"),
                    sold_quantity=product.get("sold_quantity"),
                    thumb_url=product.get("thumb_url"),
                    specifications=None,
                    extracted_content=None,  # 留空，第二阶段填充
                )
                progress.success += 1
                logger.debug(f"商品基本信息已保存: {goods_name} (ID: {goods_id})")
            except Exception as e:
                logger.error(f"保存商品基本信息失败 {goods_id}: {_sanitize_for_log(e)}")
                progress.failed += 1
                continue

            if progress_callback:
                progress_callback(progress)

        if self.is_cancelled():
            logger.info("同步已取消")
            return progress

        logger.info(f"商品基本信息保存完成: 成功 {progress.success}, 失败 {progress.failed}")

        # 同步商品只负责把商品基础信息/商品ID同步到本地；结构化知识由人工导入。
        # 这里直接结束，避免每次同步商品都调用 LLM，浪费时间和模型资源。
        logger.info(
            "商品基础信息同步完成，已跳过 LLM 结构化知识生成: "
            f"total={progress.total}, success={progress.success}, failed={progress.failed}"
        )
        return progress

    async def _extract_product_knowledge(
        self,
        list_product: Dict[str, Any],
        detail_product: Dict[str, Any],
    ) -> str:
        """
        调用LLM提取产品知识

        Args:
            list_product: 商品列表中的商品信息
            detail_product: 商品详情信息

        Returns:
            LLM提取的产品知识文本
        """
        list_product = list_product if isinstance(list_product, dict) else {}
        detail_product = detail_product if isinstance(detail_product, dict) else {}

        # 读取LLM配置
        model_name = get_config("llm.model_name", "gpt-4o")
        api_key = get_config("llm.api_key", "")
        api_base = get_config("llm.api_base", None)

        if not api_key:
            logger.warning("LLM API key not configured, returning basic info only")
            return self._format_basic_info(list_product, detail_product)

        # 创建客户端
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=120.0,
        )

        # 构建prompt（当前使用纯文本模型，只基于标题、规格、详情文本提取）
        specifications = detail_product.get("specifications", [])

        system_prompt = """你是一个电商产品信息提取助手。请根据提供的商品标题、规格、详情文本，提取详细的产品知识，方便客服回答顾客问题。

请务必先从商品名称和描述中提取以下独立字段，然后再生成其他内容：

请输出 JSON 格式，包含以下字段：
{
  "brand": "品牌（从商品名称或描述中提取，如"葵花"、"同仁堂"等）",
  "origin": "产地（从描述中提取，如"中国广东"、"日本"等）",
  "ingredients": "产品成分/材料/主要原料（从描述中提取，如"草本成分"、"植物精油"等）",
  "spec_quantity": "规格/数量/包装规格（从描述中提取，如"1盒8贴"、"50g/瓶"等）",
  "suitable_age": "适用年龄（从描述中提取，如"儿童成人通用"、"3岁以上"等）",
  "shelf_life": "保质期/有效期（从描述中提取，如"24个月"、"3年"等）",
  "description": "商品整体描述，包含卖点、特点、材质、用途等信息",
  "key_points": ["卖点1", "卖点2", ...],
  "usage": "使用方法或注意事项（如果有）",
  "faq": [{"question": "常见问题", "answer": "答案"}, ...]
}

重要提示：
1. 请优先从商品名称和描述文本中提取品牌、成分、规格、适用年龄等信息
2. 如果某个信息已经在描述中提到，请务必提取到对应的独立字段中
3. 只能基于文本信息提取，不要要求图片，也不要假设图片内容
4. 如果某个信息确实无法提取，对应字段留空字符串"""

        user_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": f"""商品名称: {list_product.get('goods_name')}
商品价格: {list_product.get('price')}
已售数量: {list_product.get('sold_quantity')}
规格: {json.dumps(specifications, ensure_ascii=False)}
"""
            }
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            content = _extract_llm_content(response)
            logger.debug(f"LLM输出长度: {len(str(content or ''))}")

            # 尝试解析JSON
            try:
                data = json.loads(content)
                if not isinstance(data, dict):
                    logger.warning(f"LLM输出JSON不是对象，使用基础商品信息: {type(data).__name__}")
                    return self._format_basic_info(list_product, detail_product)

                def text_field(key: str) -> str:
                    value = data.get(key)
                    return str(value).strip() if value not in (None, "") else ""

                spec_field_names = (
                    "brand",
                    "origin",
                    "ingredients",
                    "spec_quantity",
                    "suitable_age",
                    "shelf_life",
                )
                extracted_fields = [name for name in spec_field_names if text_field(name)]
                logger.debug(f"提取到的规格字段数量: {len(extracted_fields)}, fields={extracted_fields}")

                # 格式化输出
                output_parts = [f"# {list_product.get('goods_name')}"]
                output_parts.append("")
                # 产品规格信息
                spec_info = []
                for key, label in (
                    ("brand", "品牌"),
                    ("origin", "产地"),
                    ("ingredients", "产品成分"),
                    ("spec_quantity", "规格/数量"),
                    ("suitable_age", "适用年龄"),
                    ("shelf_life", "保质期"),
                ):
                    value = text_field(key)
                    if value:
                        spec_info.append(f"- **{label}**: {value}")
                if spec_info:
                    output_parts.append("## 产品规格")
                    output_parts.extend(spec_info)
                    output_parts.append("")
                description = text_field("description")
                if description:
                    output_parts.append("## 产品描述")
                    output_parts.append(description)
                    output_parts.append("")
                if data.get("key_points") and isinstance(data["key_points"], list):
                    output_parts.append("## 产品卖点")
                    for i, point in enumerate(data["key_points"], 1):
                        output_parts.append(f"{i}. {point}")
                    output_parts.append("")
                usage = text_field("usage")
                if usage:
                    output_parts.append("## 使用说明")
                    output_parts.append(usage)
                    output_parts.append("")
                if data.get("faq") and isinstance(data["faq"], list):
                    faq_lines = []
                    for faq in data["faq"]:
                        if not isinstance(faq, dict):
                            continue
                        question = str(faq.get("question") or "").strip()
                        answer = str(faq.get("answer") or "").strip()
                        if not question and not answer:
                            continue
                        faq_lines.append(f"**Q:** {question}")
                        faq_lines.append(f"**A:** {answer}")
                        faq_lines.append("")
                    if faq_lines:
                        output_parts.append("## 常见问题")
                        output_parts.extend(faq_lines)

                result = "\n".join(output_parts).strip()
                return result

            except json.JSONDecodeError:
                logger.warning(f"LLM输出不是合法JSON，使用基础商品信息: LLM输出长度={len(str(content or ''))}")
                return self._format_basic_info(list_product, detail_product)

        except Exception as e:
            logger.error(f"LLM调用失败: {_sanitize_for_log(e)}")
            # 降级返回基本信息
            return self._format_basic_info(list_product, detail_product)

    def _format_basic_info(
        self,
        list_product: Dict[str, Any],
        detail_product: Dict[str, Any],
    ) -> str:
        """LLM调用失败时，格式化基本信息"""
        list_product = list_product if isinstance(list_product, dict) else {}
        detail_product = detail_product if isinstance(detail_product, dict) else {}
        output = [f"# {list_product.get('goods_name')}"]
        output.append("")
        if list_product.get("price"):
            output.append(f"**价格**: {list_product.get('price')}")
        if list_product.get("sold_quantity"):
            output.append(f"**已售**: {list_product.get('sold_quantity')} 件")
        specs = detail_product.get("specifications", [])
        if isinstance(specs, str):
            specs = [specs]
        elif not isinstance(specs, list):
            specs = []
        if specs:
            output.append("")
            output.append("**规格信息**:")
            for spec in specs:
                output.append(f"- {spec}")
        return "\n".join(output).strip()
