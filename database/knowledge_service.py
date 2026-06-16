"""
知识库服务
=============

提供知识库的CRUD操作和检索功能。
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import select, and_, or_
from sqlalchemy.orm import Session
import jieba
import re
from utils.logger_loguru import get_logger
from config import get_config
from utils.config_values import as_bool, as_int
from database.models import (
    Base, ProductKnowledge, CustomerServiceKnowledge, KnowledgeMetaEntry, Shop,
    PresaleKnowledge, InsaleKnowledge, AftersaleKnowledge, SceneKnowledgeEmbedding,
)
from database.db_manager import db_manager
from database.vector_retriever import VectorItem, VectorRetriever
from core.base_service import _sanitize_for_log

logger = get_logger("KnowledgeService")
MIN_PRODUCT_HIT_CHARS = 80


CUSTOMER_SCENE_LABELS = {
    "presale": "售前",
    "insale": "售中",
    "aftersale": "售后",
}
CUSTOMER_SCENE_ALIASES = {
    "presale": (
        "售前", "售前咨询", "购买前", "下单前", "拍前", "买前", "购买咨询",
        "pre_sale", "presale", "pre-sale",
    ),
    "insale": (
        "售中", "售中-待发货", "售中-物流中", "待发货", "已发货待收货",
        "物流中", "催发货", "加急发货", "改地址", "修改地址", "拦截",
        "insale", "in_sale", "in-sale",
    ),
    "aftersale": (
        "售后", "售后倾向", "已签收", "签收后", "收到后", "质量问题",
        "退换货", "退货退款", "退款补偿", "协商", "aftersale", "after_sale",
        "after-sale",
    ),
}
DEFAULT_PRODUCT_FAMILY_RULES = ()
DEFAULT_PRODUCT_PARAMETER_KEYWORDS = (
    "商品参数", "参数", "规格", "型号", "款式", "尺寸", "尺码",
    "重量", "容量", "功率", "电压", "材质", "面料", "成分",
    "功能", "使用方法", "安装", "配件", "赠品", "颜色", "库存",
    "快递", "发货",
)
DEFAULT_PRODUCT_PARAMETER_ALIAS_RULES = (
    {"contains_any": ("功率",), "alias": "问法：功率多少瓦/几瓦/多少W/功率多大"},
    {"contains_any": ("容量",), "alias": "问法：容量多大/容量是多少"},
    {"contains_any": ("尺寸", "尺码"), "alias": "问法：尺寸多大/尺码怎么选/大小是多少"},
    {"contains_any": ("重量",), "alias": "问法：重量多少/有多重"},
    {"contains_any": ("材质", "面料", "成分"), "alias": "问法：是什么材质/什么面料/成分是什么"},
)
DEFAULT_QUALIFIER_GROUPS = ()
DEFAULT_SEARCH_PHRASE_CANDIDATES = (
    "商品参数", "参数", "规格", "型号", "款式", "尺寸", "尺码",
    "重量", "容量", "功率", "电压", "材质", "面料", "成分",
    "功能", "使用方法", "安装", "配件", "赠品", "颜色",
    "有货", "现货", "库存", "什么快递", "发货地", "质保",
    "保修", "退货包运费", "七天无理由", "7天无理由",
)
DEFAULT_SEARCH_SYNONYM_EXPANSIONS = {
    "邮政": ("快递",),
    "拒收": ("退货", "退款", "拒签"),
    "顿丰": ("快递",),
}
DEFAULT_STRUCTURED_SCENARIO_RULES = {
    "product_attribute": ("参数", "规格", "型号", "尺寸", "尺码", "重量", "容量", "功率", "材质", "面料", "颜色", "库存"),
    "product_usage": ("怎么用", "使用教程", "使用方法", "说明书", "安装"),
    "shipping": ("快递", "发货", "物流", "到货", "发货地", "从哪发", "从哪里发"),
    "aftersale": ("质保", "保修", "退货", "退款", "运费", "运费险", "质量问题", "坏了"),
}
DEFAULT_STRUCTURED_SCENARIO_ANCHORS = {
    "product_attribute": ("参数", "规格", "型号", "尺寸", "材质", "颜色", "库存"),
    "product_usage": ("使用", "教程", "说明书", "安装"),
    "shipping": ("发货", "物流", "快递"),
    "aftersale": ("售后", "退货", "退款", "质保", "质量问题"),
}
DEFAULT_STRUCTURED_INTENT_RULES = {
    "gift_accessory": {"all": ("配件",), "any": ("送", "赠", "带", "有", "里面", "包装", "配", "附")},
    "color_stock": {"all": ("颜色", "色"), "any": ("有货", "现货", "能拍", "拍下", "库存")},
    "color_query": {"any": ("颜色", "色", "几种颜色", "什么颜色")},
    "shipping_origin": {"any": ("发货地", "哪里发货", "从哪发货", "从哪里发货")},
    "shipping_express": {"any": ("什么快递", "发啥快递", "哪家快递", "快递")},
    "shipping_time": {"any": ("什么时候发货", "多久发货", "几天到", "什么时候到", "多久到", "加急", "还不发货", "不发货", "没发货", "货发了没有", "催发货", "尽快发货", "快点发货")},
    "warranty": {"any": ("质保", "保修", "坏了怎么办", "质量问题怎么办")},
    "return_shipping": {"any": ("退货包运费", "运费谁出", "运费险")},
    "return_policy": {"any": ("可以退货吗", "退货政策", "退款", "7天无理由")},
    "size_weight": {"any": ("尺寸", "多大", "多重", "重量", "几厘米")},
}
DEFAULT_INTENT_SPECIFIC_SCORE_RULES = ()
DEFAULT_INTENT_SCORE_ADJUSTMENT_RULES = ()
class KnowledgeService:
    """知识库服务，提供产品知识和客服知识的CRUD和检索功能"""

    def __init__(self):
        """初始化知识库服务"""
        # 复用现有的数据库管理器，确保路径一致
        self.session_factory = db_manager.Session
        self.vector_retriever = VectorRetriever()
        # 确保知识库相关的表存在
        Base.metadata.create_all(db_manager.engine)
        logger.info("KnowledgeService 初始化成功，复用全局数据库连接")

    def get_session(self) -> Session:
        """获取数据库会话"""
        return self.session_factory()

    # ========== Meta 知识 ==========

    def replace_meta_entries(
        self,
        shop_id: int,
        entries: List[Dict[str, Any]],
        source_type: Optional[str] = None,
        product_family: Optional[str] = None,
    ) -> int:
        """按来源批量重建 meta 知识。"""
        with self.get_session() as session:
            conditions = [KnowledgeMetaEntry.shop_id == shop_id]
            if source_type:
                conditions.append(KnowledgeMetaEntry.source_type == source_type)
            if product_family:
                conditions.append(KnowledgeMetaEntry.product_family == product_family)

            session.query(KnowledgeMetaEntry).filter(and_(*conditions)).delete()

            now = datetime.now()
            created = 0
            for index, item in enumerate(entries):
                if not isinstance(item, dict):
                    logger.warning(f"跳过无效Meta知识行: index={index}, reason=not_dict")
                    continue
                missing = [
                    key
                    for key in ("source_type", "source_id", "scenario", "aliases", "answer")
                    if item.get(key) in (None, "")
                ]
                if missing:
                    logger.warning(f"跳过无效Meta知识行: index={index}, missing={missing}")
                    continue
                source_id = as_int(item.get("source_id"), -1)
                if source_id < 0:
                    logger.warning(f"跳过无效Meta知识行: index={index}, source_id={item.get('source_id')}")
                    continue
                meta = KnowledgeMetaEntry(
                    shop_id=shop_id,
                    source_type=item["source_type"],
                    source_id=source_id,
                    goods_id=item.get("goods_id"),
                    product_family=item.get("product_family"),
                    scenario=item["scenario"],
                    sub_intent=item.get("sub_intent"),
                    aliases=item["aliases"],
                    answer=item["answer"],
                    section_title=item.get("section_title"),
                    tags=item.get("tags"),
                    enabled=as_bool(item.get("enabled", True), True),
                    priority=as_int(item.get("priority"), 0),
                    created_at=now,
                    updated_at=now,
                )
                session.add(meta)
                created += 1
            session.commit()
            logger.info(
                f"Meta知识重建完成: shop_id={shop_id}, source_type={source_type}, "
                f"product_family={product_family}, created={created}"
            )
            return created

    # ========== 产品知识 ==========

    def get_product_by_goods_id(self, shop_id: int, goods_id: int) -> Optional[ProductKnowledge]:
        """根据商品ID获取产品知识"""
        with self.get_session() as session:
            stmt = select(ProductKnowledge).where(
                and_(
                    ProductKnowledge.shop_id == shop_id,
                    ProductKnowledge.goods_id == goods_id
                )
            )
            return session.scalar(stmt)

    def list_products_by_shop(self, shop_id: int) -> List[ProductKnowledge]:
        """获取店铺所有产品知识"""
        with self.get_session() as session:
            stmt = select(ProductKnowledge).where(
                ProductKnowledge.shop_id == shop_id
            ).order_by(
                ProductKnowledge.last_extracted_at.desc(),
                ProductKnowledge.updated_at.desc(),
                ProductKnowledge.created_at.desc(),
            )
            return list(session.scalars(stmt))

    def list_product_families_by_shop(self, shop_id: int) -> List[Dict[str, Any]]:
        """List product families available in scene knowledge for one shop."""
        families: Dict[str, Dict[str, Any]] = {}
        table_models = (
            ("presale", PresaleKnowledge),
            ("insale", InsaleKnowledge),
            ("aftersale", AftersaleKnowledge),
        )
        with self.get_session() as session:
            for scene_key, model in table_models:
                rows = session.execute(
                    select(model.product_family, model.goods_id).where(and_(
                        model.shop_id == shop_id,
                        model.product_family.is_not(None),
                        model.product_family != "",
                    ))
                ).all()
                for family, goods_id in rows:
                    key = str(family or "").strip()
                    if not key:
                        continue
                    item = families.setdefault(
                        key,
                        {
                            "product_family": key,
                            "goods_count": set(),
                            "presale_count": 0,
                            "insale_count": 0,
                            "aftersale_count": 0,
                        },
                    )
                    if goods_id is not None:
                        item["goods_count"].add(int(goods_id))
                    item[f"{scene_key}_count"] += 1

        result = []
        for item in families.values():
            result.append({
                "product_family": item["product_family"],
                "goods_count": len(item["goods_count"]),
                "presale_count": item["presale_count"],
                "insale_count": item["insale_count"],
                "aftersale_count": item["aftersale_count"],
            })
        result.sort(key=lambda row: row["product_family"].lower())
        return result

    def list_scene_knowledge_by_family(
        self,
        scene: str,
        shop_id: int,
        product_family: str,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        """List editable scene knowledge rows for one product family."""
        scene_key = str(scene or "").lower().strip()
        model = self._SCENE_MODEL_MAP.get(scene_key)
        if not model:
            return []
        family = str(product_family or "").strip()
        if not family:
            return []

        db_shop_id = self._resolve_shop_id(shop_id)
        with self.get_session() as session:
            rows = list(session.scalars(
                select(model).where(and_(
                    model.shop_id == db_shop_id,
                    model.product_family == family,
                )).order_by(
                    model.goods_id.asc(),
                    model.priority.desc(),
                    model.section_title.asc(),
                    model.id.asc(),
                ).limit(limit)
            ))

        return [
            {
                "id": row.id,
                "scene": scene_key,
                "goods_id": row.goods_id,
                "product_family": row.product_family or "",
                "sub_intent": row.sub_intent or "",
                "aliases": row.aliases or "",
                "answer": row.answer or "",
                "section_title": row.section_title or "",
                "tags": row.tags or "",
                "priority": self._priority_value(row),
                "enabled": as_bool(row.enabled, True),
                "source_type": row.source_type or "",
                "source_id": row.source_id,
                "source_meta_id": row.source_meta_id,
            }
            for row in rows
        ]

    def list_product_family_links(self, shop_id: int, product_family: str) -> List[Dict[str, Any]]:
        """List goods linked to one product family."""
        family = str(product_family or "").strip()
        if not family:
            return []
        table_models = (PresaleKnowledge, InsaleKnowledge, AftersaleKnowledge)
        goods_ids = set()
        with self.get_session() as session:
            for model in table_models:
                rows = session.execute(
                    select(model.goods_id).where(and_(
                        model.shop_id == shop_id,
                        model.product_family == family,
                        model.goods_id.is_not(None),
                    ))
                ).all()
                goods_ids.update(int(row[0]) for row in rows if row[0] is not None)

            products_by_goods = {
                int(product.goods_id): product
                for product in session.scalars(
                    select(ProductKnowledge).where(and_(
                        ProductKnowledge.shop_id == shop_id,
                        ProductKnowledge.goods_id.in_(goods_ids) if goods_ids else False,
                    ))
                )
            }

        links = []
        for goods_id in sorted(goods_ids):
            product = products_by_goods.get(goods_id)
            links.append({
                "goods_id": goods_id,
                "goods_name": product.goods_name if product else "",
                "price": product.price if product else "",
            })
        return links

    def list_unbound_products_for_family(self, shop_id: int, product_family: str) -> List[Dict[str, Any]]:
        """List products in the shop that are not linked to the given product family."""
        bound_goods = {
            int(item["goods_id"])
            for item in self.list_product_family_links(shop_id, product_family)
            if item.get("goods_id") is not None
        }
        with self.get_session() as session:
            products = list(session.scalars(
                select(ProductKnowledge).where(ProductKnowledge.shop_id == shop_id).order_by(
                    ProductKnowledge.updated_at.desc(),
                    ProductKnowledge.created_at.desc(),
                )
            ))
        return [
            {
                "goods_id": int(product.goods_id),
                "goods_name": product.goods_name or "",
                "price": product.price or "",
            }
            for product in products
            if int(product.goods_id) not in bound_goods
        ]

    def bind_product_to_family(self, shop_id: int, product_family: str, goods_id: int) -> Dict[str, int]:
        """Bind one product to a family by copying the family's scene knowledge templates."""
        family = str(product_family or "").strip()
        target_goods_id = int(goods_id)
        if not family or target_goods_id <= 0:
            return {"inserted": 0, "deleted": 0, "embedding_deleted": 0}

        scene_models = (
            ("presale", PresaleKnowledge),
            ("insale", InsaleKnowledge),
            ("aftersale", AftersaleKnowledge),
        )
        now = datetime.now()
        summary = {"inserted": 0, "deleted": 0, "embedding_deleted": 0}
        with self.get_session() as session:
            product = session.scalar(select(ProductKnowledge).where(and_(
                ProductKnowledge.shop_id == shop_id,
                ProductKnowledge.goods_id == target_goods_id,
            )))
            if not product:
                return summary

            for scene_key, model in scene_models:
                existing = list(session.scalars(select(model).where(and_(
                    model.shop_id == shop_id,
                    model.goods_id == target_goods_id,
                    model.product_family == family,
                ))))
                existing_ids = [row.id for row in existing]
                if existing_ids:
                    embeddings = list(session.scalars(select(SceneKnowledgeEmbedding).where(and_(
                        SceneKnowledgeEmbedding.scene == scene_key,
                        SceneKnowledgeEmbedding.knowledge_table == model.__tablename__,
                        SceneKnowledgeEmbedding.knowledge_id.in_(existing_ids),
                    ))))
                    for row in embeddings:
                        session.delete(row)
                    summary["embedding_deleted"] += len(embeddings)
                    for row in existing:
                        session.delete(row)
                    summary["deleted"] += len(existing)
                    session.flush()

                template_goods_id = session.scalar(select(model.goods_id).where(and_(
                    model.shop_id == shop_id,
                    model.product_family == family,
                    model.goods_id.is_not(None),
                    model.goods_id != target_goods_id,
                )).order_by(model.goods_id.asc()))
                if template_goods_id is None:
                    continue

                templates = list(session.scalars(select(model).where(and_(
                    model.shop_id == shop_id,
                    model.product_family == family,
                    model.goods_id == template_goods_id,
                )).order_by(model.id.asc())))

                for template in templates:
                    row = model(
                        shop_id=shop_id,
                        goods_id=target_goods_id,
                        product_family=family,
                        sub_intent=template.sub_intent,
                        aliases=template.aliases,
                        answer=template.answer,
                        section_title=template.section_title,
                        tags=template.tags,
                        priority=self._priority_value(template),
                        enabled=as_bool(template.enabled, True),
                        source_type=f"family_bind_{family}"[:50],
                        source_id=target_goods_id,
                        source_meta_id=template.id,
                        migrated_at=now,
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(row)
                    summary["inserted"] += 1
            session.commit()
        return summary

    def rebuild_knowledge_index(self, shop_id: Optional[int] = None, batch_size: int = 100) -> Dict[str, Any]:
        """Rebuild scene knowledge embeddings for all scenes."""
        db_shop_id = self._resolve_shop_id(shop_id) if shop_id is not None else None
        with self.get_session() as session:
            stmt = select(SceneKnowledgeEmbedding)
            if db_shop_id is not None:
                stmt = stmt.where(SceneKnowledgeEmbedding.shop_id == db_shop_id)
            rows = list(session.scalars(stmt))
            deleted = len(rows)
            for row in rows:
                session.delete(row)
            session.commit()

        stats = {}
        for scene_key in ("presale", "insale", "aftersale"):
            stats[scene_key] = self.build_scene_embeddings(
                scene=scene_key,
                shop_id=db_shop_id,
                batch_size=batch_size,
            )
        stats["_deleted_embeddings"] = deleted
        return stats

    def count_products_by_shop(self, shop_id: int) -> int:
        """统计店铺产品知识数量"""
        with self.get_session() as session:
            return session.query(ProductKnowledge).filter(
                ProductKnowledge.shop_id == shop_id
            ).count()

    def add_or_update_product(
        self,
        shop_id: int,
        goods_id: int,
        goods_name: str,
        price: Optional[str] = None,
        price_min: Optional[int] = None,
        price_max: Optional[int] = None,
        sold_quantity: Optional[int] = None,
        thumb_url: Optional[str] = None,
        specifications: Optional[str] = None,
        extracted_content: Optional[str] = None,
    ) -> ProductKnowledge:
        """添加或更新产品知识"""
        with self.get_session() as session:
            # 在同一个 session 中查询
            stmt = select(ProductKnowledge).where(
                and_(
                    ProductKnowledge.shop_id == shop_id,
                    ProductKnowledge.goods_id == goods_id
                )
            )
            existing = session.scalar(stmt)

            if existing:
                # 更新现有记录
                if goods_name is not None:
                    existing.goods_name = goods_name
                if price is not None:
                    existing.price = price
                if price_min is not None:
                    existing.price_min = price_min
                if price_max is not None:
                    existing.price_max = price_max
                if sold_quantity is not None:
                    existing.sold_quantity = sold_quantity
                if thumb_url is not None:
                    existing.thumb_url = thumb_url
                if specifications is not None:
                    existing.specifications = specifications
                if extracted_content is not None:
                    existing.extracted_content = extracted_content
                existing.last_extracted_at = datetime.now()
                product = existing
                session.flush()
            else:
                # 创建新记录
                product = ProductKnowledge(
                    shop_id=shop_id,
                    goods_id=goods_id,
                    goods_name=goods_name,
                    price=price,
                    price_min=price_min,
                    price_max=price_max,
                    sold_quantity=sold_quantity,
                    thumb_url=thumb_url,
                    specifications=specifications,
                    extracted_content=extracted_content,
                )
                session.add(product)
                session.flush()

            session.commit()
            # 重新查询以确保返回的是附加到 session 的对象
            stmt = select(ProductKnowledge).where(
                and_(
                    ProductKnowledge.shop_id == shop_id,
                    ProductKnowledge.goods_id == goods_id
                )
            )
            result = session.scalar(stmt)
            logger.info(f"产品知识保存成功: shop_id={shop_id}, goods_id={goods_id}")
            return result

    def update_product_extracted_content(
        self,
        shop_id: int,
        goods_id: int,
        specifications: Optional[str] = None,
        extracted_content: Optional[str] = None,
    ) -> bool:
        """仅更新产品的提取内容（用于第二阶段更新）"""
        with self.get_session() as session:
            stmt = select(ProductKnowledge).where(
                and_(
                    ProductKnowledge.shop_id == shop_id,
                    ProductKnowledge.goods_id == goods_id
                )
            )
            product = session.scalar(stmt)
            if not product:
                logger.warning(f"产品不存在，无法更新提取内容: shop_id={shop_id}, goods_id={goods_id}")
                return False

            if specifications is not None:
                product.specifications = specifications
            if extracted_content is not None:
                product.extracted_content = extracted_content
            product.last_extracted_at = datetime.now()

            session.commit()
            logger.info(f"产品提取内容更新成功: shop_id={shop_id}, goods_id={goods_id}")
            return True

    def delete_product(self, product_id: int) -> bool:
        """删除产品知识"""
        with self.get_session() as session:
            product = session.get(ProductKnowledge, product_id)
            if not product:
                return False
            session.delete(product)
            session.commit()
            logger.info(f"产品知识删除成功: id={product_id}")
            return True

    def clear_products_by_shop(self, shop_id: int) -> int:
        """清空店铺所有产品知识，返回删除数量"""
        with self.get_session() as session:
            count = session.query(ProductKnowledge).filter(
                ProductKnowledge.shop_id == shop_id
            ).delete()
            session.commit()
            logger.info(f"清空店铺产品知识: shop_id={shop_id}, deleted={count}")
            return count

    # ========== 客服知识 ==========

    def get_customer_service_by_id(self, cs_id: int) -> Optional[CustomerServiceKnowledge]:
        """根据ID获取客服知识"""
        with self.get_session() as session:
            return session.get(CustomerServiceKnowledge, cs_id)

    def list_customer_service_by_shop(self, shop_id: int) -> List[CustomerServiceKnowledge]:
        """获取店铺所有启用的客服知识"""
        with self.get_session() as session:
            stmt = select(CustomerServiceKnowledge).where(
                and_(
                    CustomerServiceKnowledge.shop_id == shop_id,
                    CustomerServiceKnowledge.enabled == True
                )
            ).order_by(
                CustomerServiceKnowledge.updated_at.desc(),
                CustomerServiceKnowledge.created_at.desc(),
            )
            return list(session.scalars(stmt))

    def list_customer_service_with_disabled(self, shop_id: int) -> List[CustomerServiceKnowledge]:
        """获取店铺所有客服知识（包括禁用的）"""
        with self.get_session() as session:
            stmt = select(CustomerServiceKnowledge).where(
                CustomerServiceKnowledge.shop_id == shop_id
            ).order_by(
                CustomerServiceKnowledge.updated_at.desc(),
                CustomerServiceKnowledge.created_at.desc(),
            )
            return list(session.scalars(stmt))

    def count_customer_service_by_shop(self, shop_id: int) -> int:
        """统计店铺客服知识数量"""
        with self.get_session() as session:
            return session.query(CustomerServiceKnowledge).filter(
                CustomerServiceKnowledge.shop_id == shop_id
            ).count()

    def add_customer_service(
        self,
        shop_id: int,
        title: str,
        content: str,
        tags: Optional[str] = None,
        enabled: bool = True,
    ) -> CustomerServiceKnowledge:
        """添加客服知识"""
        with self.get_session() as session:
            cs = CustomerServiceKnowledge(
                shop_id=shop_id,
                title=title,
                content=content,
                tags=tags,
                enabled=as_bool(enabled, True),
            )
            session.add(cs)
            session.commit()
            logger.info(f"客服知识添加成功: shop_id={shop_id}, title={title}")
            return cs

    def update_customer_service(
        self,
        cs_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> Optional[CustomerServiceKnowledge]:
        """更新客服知识"""
        with self.get_session() as session:
            cs = session.get(CustomerServiceKnowledge, cs_id)
            if not cs:
                return None
            if title is not None:
                cs.title = title
            if content is not None:
                cs.content = content
            if tags is not None:
                cs.tags = tags
            if enabled is not None:
                cs.enabled = as_bool(enabled, as_bool(cs.enabled, True))
            session.commit()
            logger.info(f"客服知识更新成功: id={cs_id}")
            return cs

    def delete_customer_service(self, cs_id: int) -> bool:
        """删除客服知识"""
        with self.get_session() as session:
            cs = session.get(CustomerServiceKnowledge, cs_id)
            if not cs:
                return False
            session.delete(cs)
            session.commit()
            logger.info(f"客服知识删除成功: id={cs_id}")
            return True

    def batch_import_customer_service(
        self,
        shop_id: int,
        rows: List[Dict[str, Any]],
    ) -> tuple[int, int]:
        """批量导入客服知识，跳过重复项（同店铺内标题+内容完全相同）

        Args:
            shop_id: 店铺数据库ID
            rows: 待导入行列表，每项含 title, content, tags

        Returns:
            (success_count, skipped_count)
        """
        success = 0
        skipped = 0
        with self.get_session() as session:
            for index, row in enumerate(rows or []):
                if not isinstance(row, dict):
                    logger.warning(f"跳过无效客服知识行: index={index}, reason=not_dict")
                    skipped += 1
                    continue

                title = str(row.get("title") or "").strip()
                content = str(row.get("content") or "").strip()
                if not title or not content:
                    logger.warning(f"跳过无效客服知识行: index={index}, reason=missing_title_or_content")
                    skipped += 1
                    continue

                tags = row.get("tags")

                # 重复检测：同店铺下标题+内容完全相同
                stmt = select(CustomerServiceKnowledge).where(
                    and_(
                        CustomerServiceKnowledge.shop_id == shop_id,
                        CustomerServiceKnowledge.title == title,
                        CustomerServiceKnowledge.content == content,
                    )
                )
                if session.scalar(stmt) is not None:
                    skipped += 1
                    continue

                cs = CustomerServiceKnowledge(
                    shop_id=shop_id,
                    title=title,
                    content=content,
                    tags=tags,
                    enabled=True,
                )
                session.add(cs)
                success += 1

            session.commit()
        logger.info(f"批量导入客服知识: shop_id={shop_id}, success={success}, skipped={skipped}")
        return success, skipped

    def filter_customer_service_by_tag(self, shop_id: int, tag: str) -> List[CustomerServiceKnowledge]:
        """按标签筛选客服知识"""
        with self.get_session() as session:
            # LIKE 查询匹配标签
            stmt = select(CustomerServiceKnowledge).where(
                and_(
                    CustomerServiceKnowledge.shop_id == shop_id,
                    CustomerServiceKnowledge.enabled == True,
                    CustomerServiceKnowledge.tags.like(f"%{tag}%"),
                )
            ).order_by(
                CustomerServiceKnowledge.updated_at.desc(),
                CustomerServiceKnowledge.created_at.desc(),
            )
            return list(session.scalars(stmt))

    def get_all_tags(self, shop_id: int) -> List[str]:
        """获取店铺所有标签（去重）"""
        with self.get_session() as session:
            stmt = select(CustomerServiceKnowledge.tags).where(
                CustomerServiceKnowledge.shop_id == shop_id
            )
            tags_list = []
            for row in session.execute(stmt):
                if isinstance(row, (str, bytes, bytearray)):
                    continue
                try:
                    raw_tags = row[0]
                except (TypeError, IndexError, KeyError):
                    continue
                if raw_tags in (None, ""):
                    continue
                tags_text = str(raw_tags)
                tags_list.extend([t.strip() for t in tags_text.split(',') if t.strip()])
            # 去重
            return sorted(list(set(tags_list)))

    # ========== 检索 ==========

    @staticmethod
    def _coerce_internal_shop_id(shop_id: Any) -> Optional[int]:
        """Return a DB primary-key candidate only for unambiguous integer IDs."""
        if isinstance(shop_id, bool):
            return None
        if isinstance(shop_id, int):
            return shop_id
        if isinstance(shop_id, str):
            text = shop_id.strip()
            if text.isdigit():
                return int(text)
        return None

    def _resolve_shop_id(self, shop_id: int | str) -> int | str:
        """
        将店铺原始ID转换为数据库中的Shop.id

        Args:
            shop_id: 店铺原始ID（如591119888）

        Returns:
            数据库中的Shop.id（如1），如果找不到返回原值
        """
        with self.get_session() as session:
            stmt = select(Shop).where(Shop.shop_id == str(shop_id))
            shop = session.scalar(stmt)
            if shop:
                return shop.id
            # 如果没找到，尝试直接用内部主键查询（兼容已有数据）
            internal_shop_id = self._coerce_internal_shop_id(shop_id)
            if internal_shop_id is not None:
                shop2 = session.get(Shop, internal_shop_id)
                if shop2:
                    return shop2.id
            # 找不到时返回原值，让后续查询返回空结果
            logger.warning(f"未找到店铺: shop_id={shop_id}")
            return shop_id

    @staticmethod
    def _priority_value(entry: Any, default: int = 0) -> int:
        return as_int(getattr(entry, "priority", default), default)

    def _product_vector_items(self, products: List[ProductKnowledge]) -> List[VectorItem]:
        items = []
        for product in products:
            if not (product.extracted_content or "").strip():
                continue
            parts = [
                product.goods_name or "",
                product.price or "",
                product.specifications or "",
                product.extracted_content or "",
            ]
            text = "\n".join(part for part in parts if part)
            if text.strip():
                items.append(VectorItem(f"product:{product.id}", text, product))
        return items

    def _customer_service_vector_items(self, cs_list: List[CustomerServiceKnowledge]) -> List[VectorItem]:
        items = []
        for cs in cs_list:
            parts = [
                cs.title or "",
                cs.tags or "",
                cs.content or "",
            ]
            text = "\n".join(part for part in parts if part)
            if text.strip():
                items.append(VectorItem(f"customer_service:{cs.id}", text, cs))
        return items

    def _product_content_vector_items(self, product: ProductKnowledge) -> List[VectorItem]:
        content = product.extracted_content or ""
        chunks = self._product_knowledge_blocks(content) or self._chunk_text(content, max_chars=420, overlap_chars=80)
        return [
            VectorItem(
                f"product:{product.id}:chunk:{index}",
                "\n".join(part for part in [product.goods_name or "", chunk] if part),
                self._strip_embedding_aliases(chunk),
            )
            for index, chunk in enumerate(chunks)
            if chunk.strip()
        ]

    def _rank_product_content(self, product: ProductKnowledge, query: Optional[str], limit: int = 5) -> str:
        if not query or not query.strip() or not (product.extracted_content or "").strip():
            return ""

        structured_content = self._rank_product_faq_content(product.extracted_content or "", query)
        if structured_content:
            return structured_content

        vector_chunks = self.vector_retriever.rank(
            namespace=f"product_knowledge_{product.id}_chunks",
            shop_id=product.shop_id,
            query=query,
            items=self._product_content_vector_items(product),
            limit=limit,
        )
        keyword_content = self._keyword_product_content(product.extracted_content or "", query, limit)
        if vector_chunks:
            return self._merge_text_blocks(
                keyword_content.split("\n\n") if keyword_content else [],
                [str(chunk) for chunk in vector_chunks],
                limit,
            )

        return keyword_content

    @classmethod
    def _rank_product_faq_content(cls, content: str, query: str) -> str:
        records = cls._product_faq_records(content)
        if not records:
            return ""

        scored = []
        for index, record in enumerate(records):
            score = cls._structured_match_score(
                query=query,
                aliases=record["aliases"],
                answer=record["answer"],
                section=record["section"],
            )
            if score > 0:
                scored.append((score, index, record))

        if not scored:
            return ""

        scored.sort(key=lambda item: (-item[0], item[1]))
        top_score, _, top_record = scored[0]
        if top_score < 8:
            return ""

        return (
            f"{top_record['section']}\n"
            f"{top_record['answer']}"
        ).strip()

    @classmethod
    def _product_faq_records(cls, content: str) -> List[Dict[str, str]]:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        records: List[Dict[str, str]] = []
        section = ""
        index = 0

        while index < len(lines):
            line = lines[index]
            if line.startswith("###"):
                section = line
                index += 1
                continue
            if line.startswith("##"):
                section = line
                index += 1
                continue

            aliases = cls._label_value(line, ("问法",))
            if not aliases:
                index += 1
                continue

            answer = ""
            next_index = index + 1
            while next_index < len(lines):
                next_line = lines[next_index]
                if next_line.startswith(("##", "###")) or cls._label_value(next_line, ("问法",)):
                    break
                answer = cls._label_value(next_line, ("要点", "答案"))
                if answer:
                    break
                next_index += 1

            if answer:
                records.append({
                    "section": section,
                    "aliases": aliases,
                    "answer": answer,
                })
                index = next_index + 1
                continue

            index += 1

        return records

    @staticmethod
    def _label_value(line: str, labels: tuple[str, ...]) -> str:
        clean = str(line or "").strip()
        if clean.startswith("-"):
            clean = clean[1:].strip()
        for label in labels:
            for separator in ("：", ":"):
                prefix = f"{label}{separator}"
                if clean.startswith(prefix):
                    return clean[len(prefix):].strip()
        return ""

    @classmethod
    def _structured_match_score(
        cls,
        query: str,
        aliases: str,
        answer: str,
        section: str = "",
        tags: str = "",
    ) -> int:
        query_clean = cls._normalize_query_match_text(query)
        aliases_clean = cls._normalize_match_text(aliases)
        answer_clean = cls._normalize_match_text(answer)
        section_clean = cls._normalize_match_text(section)
        tags_clean = cls._normalize_match_text(tags)

        score = 0
        best_direct_alias_score = 0
        weak_alias_score = 0
        for alias in re.split(r"[/|;；\n\r]+", aliases or ""):
            alias_clean = cls._normalize_match_text(alias)
            if len(alias_clean) < 2:
                continue
            alias_score = 0
            if alias_clean == query_clean:
                alias_score = 240 + min(len(alias_clean), 30)
            elif len(alias_clean) >= 4 and alias_clean in query_clean:
                alias_score = 115 + min(len(alias_clean), 12)
            elif len(query_clean) >= 6 and query_clean in alias_clean:
                alias_score = 80 + min(len(query_clean), 12)

            if alias_score:
                best_direct_alias_score = max(best_direct_alias_score, alias_score)
            elif alias_clean in query_clean or query_clean in alias_clean:
                weak_alias_score = max(weak_alias_score, 8 + min(len(alias_clean), 6))
        score += best_direct_alias_score + weak_alias_score

        for number in re.findall(r"\d+", query_clean):
            if len(number) < 2:
                continue
            if number in aliases_clean:
                score += 45
            elif number in answer_clean or number in section_clean or number in tags_clean:
                score += 18

        for term in cls._search_terms(query):
            term_clean = cls._normalize_match_text(term)
            if not term_clean:
                continue
            if term_clean in aliases_clean:
                score += 8
            if term_clean in section_clean:
                score += 4
            if term_clean in answer_clean:
                score += 2
            if term_clean in tags_clean:
                score += 2

        direct_alias_matched = best_direct_alias_score > 0
        query_scenario = cls._detect_query_scenario(query)
        scenario_anchors = cls._scenario_anchor_terms(query_scenario)
        if scenario_anchors:
            if any(term in section_clean or term in tags_clean for term in scenario_anchors):
                score += 30
            elif any(term in aliases_clean or term in answer_clean for term in scenario_anchors):
                score += 8

        query_intent = cls._detect_query_intent(query)
        score += cls._intent_specific_match_score(
            query_clean=query_clean,
            aliases_clean=aliases_clean,
            answer_clean=answer_clean,
            section_clean=section_clean,
            tags_clean=tags_clean,
        )
        if query_intent == "charge_abnormal":
            abnormal_terms = tuple(
                cls._normalize_match_text(term)
                for term in (
                    "充电异常", "充不了", "充不进", "充不上", "不能充电",
                    "无法充电", "没法充电", "充电没反应", "充电不亮",
                )
            )
            non_fault_charge_terms = tuple(
                cls._normalize_match_text(term)
                for term in (
                    "电量显示", "不插电", "怎么看电量", "充电时间", "充满",
                    "充电线", "充电器", "充电头", "边充边用", "充电口", "指示灯",
                )
            )
            structured_text = f"{section_clean}{tags_clean}{aliases_clean}"
            if any(term in structured_text for term in abnormal_terms):
                score += 60
            elif any(term in structured_text for term in non_fault_charge_terms):
                score -= 40

        for qualifier_group in cls._qualifier_groups():
            query_has = any(term in query_clean for term in qualifier_group)
            knowledge_has = any(term in aliases_clean or term in answer_clean for term in qualifier_group)
            if query_has and knowledge_has:
                score += 14
            elif not direct_alias_matched:
                if query_has and not knowledge_has:
                    score -= 18
                elif knowledge_has and not query_has:
                    score -= 18

        knowledge_scenario = cls._detect_query_scenario(f"{section} {tags} {aliases} {answer}")
        if query_scenario and knowledge_scenario:
            if query_scenario == knowledge_scenario:
                score += 36
            elif not direct_alias_matched:
                score -= 42

        knowledge_intent = cls._detect_query_intent(f"{aliases} {answer}")
        if query_intent and knowledge_intent:
            if query_intent == knowledge_intent:
                score += 24
            elif not direct_alias_matched:
                score -= 24
                hard_conflict_intents = {
                    "shipping_time", "shipping_express", "shipping_origin",
                    "return_policy", "return_shipping", "warranty",
                }
                if query_intent in hard_conflict_intents or knowledge_intent in hard_conflict_intents:
                    score -= 36

        return score

    @staticmethod
    def _normalize_match_text(text: str) -> str:
        return re.sub(r"[\s\?？!！,，。.;；:：、~～\[\]【】()（）]+", "", str(text or "").lower())

    @classmethod
    def _normalize_query_match_text(cls, text: str) -> str:
        clean = cls._normalize_match_text(text)
        for prefix in ("内容客户消息", "客户消息"):
            if clean.startswith(prefix):
                return clean[len(prefix):]
        return clean

    @classmethod
    def _has_exact_alias_match(cls, query: str, aliases: str) -> bool:
        query_clean = cls._normalize_query_match_text(query)
        if not query_clean:
            return False
        return any(
            cls._normalize_match_text(alias) == query_clean
            for alias in re.split(r"[/|;；\n\r]+", aliases or "")
        )

    @classmethod
    def _scenario_anchor_terms(cls, scenario: str) -> tuple[str, ...]:
        anchors = cls._configured_term_map(
            "knowledge.structured_scenario_anchors",
            DEFAULT_STRUCTURED_SCENARIO_ANCHORS,
        )
        return tuple(cls._normalize_match_text(term) for term in anchors.get(scenario or "", ()))

    @classmethod
    def _intent_specific_match_score(
        cls,
        query_clean: str,
        aliases_clean: str,
        answer_clean: str,
        section_clean: str,
        tags_clean: str,
    ) -> int:
        """处理短问法和高风险相近主题，避免旧泛答案抢过精细知识。"""
        score = 0
        structured_text = f"{section_clean}{tags_clean}{aliases_clean}{answer_clean}"

        for rule in cls._configured_intent_specific_score_rules():
            if not cls._intent_specific_score_rule_matches(rule, query_clean, structured_text):
                continue
            score += int(rule["score"])

        return score

    @classmethod
    def _qualifier_groups(cls) -> tuple[tuple[str, ...], ...]:
        configured = get_config("knowledge.qualifier_groups", None)
        if configured is None:
            configured = DEFAULT_QUALIFIER_GROUPS
        elif not isinstance(configured, (list, tuple)):
            configured = DEFAULT_QUALIFIER_GROUPS

        groups = []
        for group in configured:
            if not isinstance(group, (list, tuple, set)):
                continue
            normalized = tuple(
                cls._normalize_match_text(term)
                for term in group
                if str(term or "").strip()
            )
            if normalized:
                groups.append(normalized)
        return tuple(groups)

    @classmethod
    def _detect_query_scenario(cls, text: str) -> str:
        clean = cls._normalize_match_text(text)
        if not clean:
            return ""

        def has_any(words: tuple[str, ...]) -> bool:
            return any(cls._normalize_match_text(word) in clean for word in words)

        scenarios = cls._configured_term_map(
            "knowledge.structured_scenario_rules",
            DEFAULT_STRUCTURED_SCENARIO_RULES,
        )
        for name, words in scenarios.items():
            if has_any(words):
                return name
        return ""

    @classmethod
    def _detect_query_intent(cls, text: str) -> str:
        clean = cls._normalize_match_text(text)
        if not clean:
            return ""

        for name, rule in cls._configured_structured_intent_rules().items():
            if cls._structured_intent_rule_matches(clean, rule):
                return name
        return ""

    @classmethod
    def _configured_term_map(cls, config_key: str, defaults: Dict[str, Any]) -> Dict[str, tuple[str, ...]]:
        configured = get_config(config_key, None)
        if configured is None:
            configured = defaults
        elif not isinstance(configured, dict):
            configured = defaults

        result: Dict[str, tuple[str, ...]] = {}
        for name, terms in configured.items():
            if isinstance(terms, str):
                terms = (terms,)
            if not isinstance(terms, (list, tuple, set)):
                continue
            cleaned = tuple(str(term or "").strip() for term in terms if str(term or "").strip())
            if cleaned:
                result[str(name)] = cleaned
        return result

    @classmethod
    def _configured_structured_intent_rules(cls) -> Dict[str, Dict[str, tuple[str, ...]]]:
        configured = get_config("knowledge.structured_intent_rules", None)
        if configured is None:
            configured = DEFAULT_STRUCTURED_INTENT_RULES
        elif not isinstance(configured, dict):
            configured = DEFAULT_STRUCTURED_INTENT_RULES

        rules: Dict[str, Dict[str, tuple[str, ...]]] = {}
        for name, rule in configured.items():
            if isinstance(rule, (list, tuple, set, str)):
                rule = {"any": rule}
            if not isinstance(rule, dict):
                continue
            normalized_rule: Dict[str, tuple[str, ...]] = {}
            for mode in ("any", "all"):
                terms = rule.get(mode, ())
                if isinstance(terms, str):
                    terms = (terms,)
                if not isinstance(terms, (list, tuple, set)):
                    continue
                cleaned = tuple(str(term or "").strip() for term in terms if str(term or "").strip())
                if cleaned:
                    normalized_rule[mode] = cleaned
            if normalized_rule:
                rules[str(name)] = normalized_rule
        return rules

    @classmethod
    def _configured_intent_specific_score_rules(cls) -> List[Dict[str, Any]]:
        configured = get_config("knowledge.intent_specific_score_rules", None)
        if configured is None:
            configured = DEFAULT_INTENT_SPECIFIC_SCORE_RULES
        elif not isinstance(configured, (list, tuple)):
            configured = DEFAULT_INTENT_SPECIFIC_SCORE_RULES

        rules: List[Dict[str, Any]] = []
        for rule in configured:
            if not isinstance(rule, dict):
                continue
            try:
                score = int(rule.get("score", 0))
            except (TypeError, ValueError):
                continue
            if score == 0:
                continue
            normalized_rule: Dict[str, Any] = {"score": score}
            for field in (
                "query_any",
                "query_all",
                "query_not_any",
                "query_exact",
                "knowledge_any",
                "knowledge_all",
                "knowledge_not_any",
            ):
                terms = cls._normalize_rule_terms(rule.get(field, ()))
                if terms:
                    normalized_rule[field] = terms
            if any(field.startswith("query_") for field in normalized_rule):
                rules.append(normalized_rule)
        return rules

    @classmethod
    def _normalize_rule_terms(cls, terms: Any) -> tuple[str, ...]:
        if isinstance(terms, str):
            terms = (terms,)
        if not isinstance(terms, (list, tuple, set)):
            return ()
        return tuple(
            cls._normalize_match_text(term)
            for term in terms
            if str(term or "").strip()
        )

    @classmethod
    def _intent_specific_score_rule_matches(
        cls,
        rule: Dict[str, Any],
        query_clean: str,
        structured_text: str,
    ) -> bool:
        query_exact = rule.get("query_exact", ())
        if query_exact and query_clean not in query_exact:
            return False

        query_any = rule.get("query_any", ())
        if query_any and not any(term in query_clean for term in query_any):
            return False

        query_all = rule.get("query_all", ())
        if query_all and not all(term in query_clean for term in query_all):
            return False

        query_not_any = rule.get("query_not_any", ())
        if query_not_any and any(term in query_clean for term in query_not_any):
            return False

        knowledge_any = rule.get("knowledge_any", ())
        if knowledge_any and not any(term in structured_text for term in knowledge_any):
            return False

        knowledge_all = rule.get("knowledge_all", ())
        if knowledge_all and not all(term in structured_text for term in knowledge_all):
            return False

        knowledge_not_any = rule.get("knowledge_not_any", ())
        if knowledge_not_any and any(term in structured_text for term in knowledge_not_any):
            return False

        return True

    @classmethod
    def _configured_intent_score_adjustment_rules(cls) -> List[Dict[str, Any]]:
        configured = get_config("knowledge.intent_score_adjustment_rules", None)
        if configured is None:
            configured = DEFAULT_INTENT_SCORE_ADJUSTMENT_RULES
        elif not isinstance(configured, (list, tuple)):
            configured = DEFAULT_INTENT_SCORE_ADJUSTMENT_RULES

        rules: List[Dict[str, Any]] = []
        term_fields = (
            "hints_any",
            "hints_all",
            "hints_not_any",
            "query_any",
            "query_all",
            "query_not_any",
            "section_any",
            "section_all",
            "section_not_any",
            "sub_intent_any",
            "sub_intent_all",
            "sub_intent_not_any",
            "answer_any",
            "answer_all",
            "answer_not_any",
            "aliases_any",
            "aliases_all",
            "aliases_not_any",
            "combined_any",
            "combined_all",
            "combined_not_any",
        )
        for rule in configured:
            if not isinstance(rule, dict):
                continue
            try:
                score = int(rule.get("score", 0))
            except (TypeError, ValueError):
                continue
            if score == 0:
                continue
            normalized_rule: Dict[str, Any] = {"score": score}
            group = str(rule.get("group") or "").strip()
            if group:
                normalized_rule["group"] = group
            scene = str(rule.get("scene") or "").strip()
            if scene:
                normalized_rule["scene"] = scene
            for field in term_fields:
                terms = cls._normalize_rule_terms(rule.get(field, ()))
                if terms:
                    normalized_rule[field] = terms
            if len(normalized_rule) > (1 + int(bool(group)) + int(bool(scene))):
                rules.append(normalized_rule)
        return rules

    @classmethod
    def _intent_score_adjustment_from_rules(
        cls,
        hints: set,
        query: str,
        scene_key: str,
        section: str,
        sub_intent: str,
        answer: str,
        aliases: str,
        combined: str,
    ) -> int:
        score = 0
        matched_groups: set[str] = set()
        texts = {
            "query": cls._normalize_match_text(query),
            "section": cls._normalize_match_text(section),
            "sub_intent": cls._normalize_match_text(sub_intent),
            "answer": cls._normalize_match_text(answer),
            "aliases": cls._normalize_match_text(aliases),
            "combined": cls._normalize_match_text(combined),
        }
        normalized_hints = {cls._normalize_match_text(hint) for hint in hints if str(hint or "").strip()}

        for rule in cls._configured_intent_score_adjustment_rules():
            if not cls._intent_score_adjustment_rule_matches(rule, normalized_hints, texts, scene_key):
                continue
            group = str(rule.get("group") or "").strip()
            if group:
                if group in matched_groups:
                    continue
                matched_groups.add(group)
            score += int(rule["score"])
        return score

    @classmethod
    def _intent_score_adjustment_rule_matches(
        cls,
        rule: Dict[str, Any],
        hints: set[str],
        texts: Dict[str, str],
        scene_key: str,
    ) -> bool:
        scene = str(rule.get("scene") or "").strip()
        if scene and scene != str(scene_key or "").strip():
            return False

        for prefix, value in (("hints", "".join(sorted(hints))), *texts.items()):
            any_terms = rule.get(f"{prefix}_any", ())
            if any_terms and not any(term in value for term in any_terms):
                return False

            all_terms = rule.get(f"{prefix}_all", ())
            if all_terms and not all(term in value for term in all_terms):
                return False

            not_any_terms = rule.get(f"{prefix}_not_any", ())
            if not_any_terms and any(term in value for term in not_any_terms):
                return False

        return True

    @classmethod
    def _structured_intent_rule_matches(cls, clean: str, rule: Dict[str, tuple[str, ...]]) -> bool:
        any_terms = rule.get("any", ())
        all_terms = rule.get("all", ())
        has_any = not any_terms or any(cls._normalize_match_text(term) in clean for term in any_terms)
        has_all = all(cls._normalize_match_text(term) in clean for term in all_terms)
        return has_any and has_all

    @staticmethod
    def _normalize_scenario_name(name: str) -> str:
        mapping = {
            "charge_power": "充电用电",
            "color_purchase": "购买相关",
            "product_usage": "产品使用",
            "cooling": "制冷功能",
            "noise": "静音噪音",
            "shipping": "发货物流",
            "aftersale": "退换货售后",
            "battery_endurance": "续航电池",
            "wind_power": "风力风速",
            "size_weight": "尺寸重量",
        }
        return mapping.get(name or "", name or "")

    def _rank_meta_entries(
        self,
        entries: List[KnowledgeMetaEntry],
        query: str,
        limit: int,
        scene_key: str = "",
    ) -> List[KnowledgeMetaEntry]:
        scored = []
        for index, entry in enumerate(entries):
            scenario_label = self._normalize_scenario_name(entry.scenario)
            sub_intent_label = entry.sub_intent or ""
            match_score = self._structured_match_score(
                query=query,
                aliases=entry.aliases or "",
                answer=entry.answer or "",
                section=scenario_label,
                tags=f"{entry.tags or ''} {sub_intent_label}",
            )
            if match_score <= 0:
                continue
            score = match_score + min(self._priority_value(entry), 20)
            scene_score = self._customer_scene_match_score(
                scene_key,
                scenario_label,
                sub_intent_label,
                entry.section_title or "",
                entry.tags or "",
            )
            if scene_score > 0:
                score += scene_score
            elif scene_score < 0:
                score += scene_score
            if scene_key:
                tags_text = entry.tags or ""
                scene_text = " ".join(
                    str(part or "")
                    for part in (scenario_label, entry.section_title, entry.sub_intent, tags_text)
                )
                tag_score = self._configured_tag_score(tags_text, default=35, scope="meta")
                if tag_score:
                    score += tag_score
                elif not self._primary_customer_scene(scene_text):
                    score -= 45
            if (
                scene_key
                and "售前同步" in (entry.tags or "")
                and "补充" in (entry.section_title or "")
                and not self._has_exact_alias_match(query, entry.aliases or "")
            ):
                score -= 12
            if score > 0:
                scored.append((score, getattr(entry, "id", 0) or 0, index, entry))

        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [entry for score, _, _, entry in scored[:limit] if score >= 8]

    @staticmethod
    def _search_terms(query: str) -> List[str]:
        text = str(query or "").strip()
        if not text:
            return []

        phrase_candidates = KnowledgeService._search_phrase_candidates()
        synonym_expansions = KnowledgeService._search_synonym_expansions()
        raw_terms = []
        raw_terms.extend(word.strip() for word in jieba.cut_for_search(text))
        raw_terms.extend(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text))
        raw_terms.extend(phrase for phrase in phrase_candidates if phrase.lower() in text.lower())
        if re.search(r"充(?:不|不了|不上|不进|不进去)|(?:不能|无法|没法)充电|不充电", text):
            raw_terms.extend(["充电", "充电异常"])

        # Apply synonym expansion
        _text_lower = text.lower()
        for _src, _expansions in synonym_expansions.items():
            if _src in _text_lower:
                raw_terms.extend(_expansions)

        stop_words = {
            "多久", "多少", "什么", "怎么", "可以", "有没有", "是不是",
            "这个", "那个", "一下", "大概", "请问", "亲", "需要", "帮我",
        }
        terms: List[str] = []
        for term in raw_terms:
            clean = term.strip()
            if len(clean) < 2 or clean in stop_words or clean in terms:
                continue
            terms.append(clean)
        return terms

    @staticmethod
    def _product_knowledge_blocks(content: str) -> List[str]:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        blocks = []
        section = ""
        index = 0
        parameter_keywords = KnowledgeService._product_parameter_keywords()

        while index < len(lines):
            line = lines[index]
            if line.startswith("##"):
                section = line
                index += 1
                continue

            if line.startswith("- 问法") or line.startswith("问法"):
                block_lines = [section] if section else []
                block_lines.append(line)
                next_index = index + 1
                if next_index < len(lines) and "要点" in lines[next_index]:
                    block_lines.append(lines[next_index])
                    index += 2
                else:
                    index += 1
                blocks.append("\n".join(block_lines))
                continue

            if line.startswith("- ") and any(keyword in line for keyword in parameter_keywords):
                block_lines = [section, line] if section else [line]
                aliases = KnowledgeService._parameter_aliases(line)
                if aliases:
                    block_lines.append(aliases)
                blocks.append("\n".join(block_lines))

            index += 1

        return blocks

    @staticmethod
    def _parameter_aliases(line: str) -> str:
        aliases = []
        for rule in KnowledgeService._product_parameter_alias_rules():
            keywords = rule.get("contains_any") or ()
            if isinstance(keywords, str):
                keywords = (keywords,)
            alias = str(rule.get("alias") or "").strip()
            if alias and any(str(keyword or "") and str(keyword or "") in line for keyword in keywords):
                aliases.append(alias)
        return "；".join(aliases)

    @staticmethod
    def _strip_embedding_aliases(block: str) -> str:
        lines = [
            line
            for line in block.splitlines()
            if not re.match(r"^-?\s*问法[:：]", line.strip())
        ]
        return "\n".join(lines).strip()

    @staticmethod
    def _merge_text_blocks(primary: List[str], fallback: List[str], limit: int) -> str:
        merged = []
        seen = set()
        for block in [*primary, *fallback]:
            clean_block = block.strip()
            if not clean_block or clean_block in seen:
                continue
            seen.add(clean_block)
            merged.append(clean_block)
            if len(merged) >= limit:
                break
        return "\n\n".join(merged)

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 420, overlap_chars: int = 80) -> List[str]:
        paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
        chunks = []
        current = ""

        for paragraph in paragraphs:
            if len(paragraph) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                start = 0
                step = max_chars - overlap_chars
                while start < len(paragraph):
                    chunks.append(paragraph[start:start + max_chars])
                    start += step
                continue

            candidate = f"{current}\n{paragraph}".strip() if current else paragraph
            if len(candidate) > max_chars and current:
                chunks.append(current)
                current = paragraph
            else:
                current = candidate

        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _keyword_product_content(content: str, query: str, limit: int) -> str:
        words = KnowledgeService._search_terms(query)
        if not words:
            return ""
        stop_words = {
            "多久", "多少", "什么", "怎么", "可以", "有没有", "是不是",
            "这个", "那个", "一下", "大概", "能用", "请问",
        }
        match_words = [word for word in words if word not in stop_words] or words
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        matched_blocks = []
        seen = set()
        for index, line in enumerate(lines):
            if any(word in line for word in match_words):
                block_lines = []
                if index > 0 and lines[index - 1].startswith("###"):
                    block_lines.append(lines[index - 1])
                block_lines.append(line)
                for offset in range(1, 3):
                    next_index = index + offset
                    if next_index < len(lines):
                        next_line = lines[next_index]
                        if next_line.startswith("要点") or next_line.startswith("- 要点") or "要点：" in next_line:
                            block_lines.append(next_line)
                            break
                block = "\n".join(block_lines)
                if block not in seen:
                    seen.add(block)
                    matched_blocks.append(block)
            if len(matched_blocks) >= limit:
                break
        return "\n\n".join(matched_blocks)

    @staticmethod
    def _merge_ranked_results(primary: List[Any], fallback: List[Any], limit: int) -> List[Any]:
        merged = []
        seen = set()
        for item in [*primary, *fallback]:
            item_id = getattr(item, "id", id(item))
            if item_id in seen:
                continue
            seen.add(item_id)
            merged.append(item)
            if len(merged) >= limit:
                break
        return merged

    @classmethod
    def _product_family_rules(cls) -> List[Dict[str, Any]]:
        rules = get_config("knowledge.product_family_rules", None)
        if rules is None:
            rules = list(DEFAULT_PRODUCT_FAMILY_RULES)
        elif not isinstance(rules, list):
            rules = list(DEFAULT_PRODUCT_FAMILY_RULES)
        return [rule for rule in rules if isinstance(rule, dict)]

    @staticmethod
    def _product_parameter_keywords() -> tuple[str, ...]:
        configured = get_config("knowledge.product_parameter_keywords", None)
        if configured is None:
            configured = DEFAULT_PRODUCT_PARAMETER_KEYWORDS
        elif not isinstance(configured, (list, tuple, set)):
            configured = DEFAULT_PRODUCT_PARAMETER_KEYWORDS
        keywords = tuple(str(item or "").strip() for item in configured if str(item or "").strip())
        return keywords

    @staticmethod
    def _product_parameter_alias_rules() -> List[Dict[str, Any]]:
        rules = get_config("knowledge.product_parameter_alias_rules", None)
        if rules is None:
            rules = list(DEFAULT_PRODUCT_PARAMETER_ALIAS_RULES)
        elif not isinstance(rules, list):
            rules = list(DEFAULT_PRODUCT_PARAMETER_ALIAS_RULES)
        clean_rules = []
        for rule in rules:
            if isinstance(rule, dict) and str(rule.get("alias") or "").strip():
                clean_rules.append(rule)
        return clean_rules

    @staticmethod
    def _search_phrase_candidates() -> tuple[str, ...]:
        configured = get_config("knowledge.search_phrase_candidates", None)
        if configured is None:
            configured = DEFAULT_SEARCH_PHRASE_CANDIDATES
        elif not isinstance(configured, (list, tuple, set)):
            configured = DEFAULT_SEARCH_PHRASE_CANDIDATES
        return tuple(str(item or "").strip() for item in configured if str(item or "").strip())

    @staticmethod
    def _search_synonym_expansions() -> Dict[str, tuple[str, ...]]:
        configured = get_config("knowledge.search_synonym_expansions", None)
        if configured is None:
            configured = DEFAULT_SEARCH_SYNONYM_EXPANSIONS
        elif not isinstance(configured, dict):
            configured = DEFAULT_SEARCH_SYNONYM_EXPANSIONS

        expansions: Dict[str, tuple[str, ...]] = {}
        for source, values in configured.items():
            source_text = str(source or "").strip().lower()
            if not source_text or not isinstance(values, (list, tuple, set)):
                continue
            clean_values = tuple(str(item or "").strip() for item in values if str(item or "").strip())
            if clean_values:
                expansions[source_text] = clean_values
        return expansions

    @staticmethod
    def _configured_tag_score(tags_text: str, default: int = 0, scope: str = "") -> int:
        adjustments = get_config("knowledge.tag_score_adjustments", {})
        if not isinstance(adjustments, dict):
            adjustments = {}

        score = 0
        tags = KnowledgeService._split_tags(tags_text)
        for tag, value in adjustments.items():
            tag_text = str(tag or "").strip()
            if not tag_text or tag_text not in tags:
                continue
            if isinstance(value, dict):
                value = value.get(scope, value.get("default", 0))
            try:
                score += int(value)
            except (TypeError, ValueError):
                continue
        return score

    @staticmethod
    def _split_tags(tags_text: str) -> set[str]:
        return {
            part.strip()
            for part in re.split(r"[,，;；、\s]+", str(tags_text or ""))
            if part.strip()
        }

    @classmethod
    def _infer_product_family(cls, goods_name: str) -> str:
        text = str(goods_name or "")
        lowered = text.lower()
        for rule in cls._product_family_rules():
            family = str(rule.get("family") or "").strip()
            if not family:
                continue
            contains = rule.get("contains") or ()
            if isinstance(contains, str):
                contains = (contains,)
            for keyword in contains:
                keyword_text = str(keyword or "")
                if keyword_text and keyword_text.lower() in lowered:
                    return family

            regexes = rule.get("regex") or ()
            if isinstance(regexes, str):
                regexes = (regexes,)
            for pattern in regexes:
                pattern_text = str(pattern or "")
                if not pattern_text:
                    continue
                try:
                    if re.search(pattern_text, text, flags=re.IGNORECASE):
                        return family
                except re.error:
                    logger.warning(f"商品族识别正则无效，已跳过: {pattern_text}")
        return ""

    @classmethod
    def normalize_customer_scene(cls, scene: Optional[str]) -> str:
        """归一化客服大场景，返回 presale/insale/aftersale。"""
        clean = cls._normalize_match_text(scene or "")
        if not clean:
            return ""
        direct = {
            "售前": "presale",
            "presale": "presale",
            "pre_sale": "presale",
            "pre-sale": "presale",
            "售中": "insale",
            "insale": "insale",
            "in_sale": "insale",
            "in-sale": "insale",
            "售后": "aftersale",
            "aftersale": "aftersale",
            "after_sale": "aftersale",
            "after-sale": "aftersale",
        }
        if clean in {cls._normalize_match_text(key) for key in direct}:
            for key, value in direct.items():
                if clean == cls._normalize_match_text(key):
                    return value
        for scene_key, aliases in CUSTOMER_SCENE_ALIASES.items():
            if any(cls._normalize_match_text(alias) in clean for alias in aliases):
                return scene_key
        return ""

    @classmethod
    def customer_scene_label(cls, scene: Optional[str]) -> str:
        scene_key = cls.normalize_customer_scene(scene) or str(scene or "")
        return CUSTOMER_SCENE_LABELS.get(scene_key, scene_key or "售前")

    @classmethod
    def detect_customer_scene(cls, text: str, default: str = "presale") -> str:
        """仅从显式场景/订单状态文本识别大场景；不根据客户问题关键词猜测。"""
        clean = cls._normalize_match_text(text)
        if not clean:
            return cls.normalize_customer_scene(default)

        def has_any(words: tuple[str, ...]) -> bool:
            return any(cls._normalize_match_text(word) in clean for word in words)

        if has_any(("当前业务场景：售后倾向", "当前业务场景售后倾向", "当前订单状态：已签收", "当前订单状态已签收", "已签收")):
            return "aftersale"
        if has_any((
            "当前业务场景：售中-待发货", "当前业务场景：售中-物流中",
            "当前业务场景售中待发货", "当前业务场景售中物流中",
            "当前订单状态：待发货", "当前订单状态：已发货待收货",
            "当前订单状态待发货", "当前订单状态已发货待收货",
        )):
            return "insale"
        if has_any(("售前咨询", "购买前", "下单前", "拍前", "买前")):
            return "presale"

        return cls.normalize_customer_scene(default)

    @classmethod
    def _customer_scene_match_score(cls, scene: Optional[str], *texts: str) -> int:
        scene_key = cls.normalize_customer_scene(scene)
        if not scene_key:
            return 0
        raw_text = " ".join(str(text or "") for text in texts)
        primary_scene = cls._primary_customer_scene(raw_text)
        if primary_scene:
            return 48 if primary_scene == scene_key else -48

        clean = cls._normalize_match_text(raw_text)
        if not clean:
            return 0

        hit_keys = []
        for key, aliases in CUSTOMER_SCENE_ALIASES.items():
            if any(cls._normalize_match_text(alias) in clean for alias in aliases):
                hit_keys.append(key)
        if scene_key not in hit_keys:
            return -12 if hit_keys else 0
        # 只有一个明确场景时强加权；同时出现多个场景说明是兼容规则，不强排除。
        return 36 if len(hit_keys) == 1 else 8

    @classmethod
    def _primary_customer_scene(cls, text: str) -> str:
        """识别知识条目的主场景，避免“售前同步”把跨场景副本当成原生售前。"""
        clean_text = str(text or "")
        if not clean_text.strip():
            return ""

        for token in re.split(r"[,，/|;；\s]+", clean_text):
            clean_token = cls._normalize_match_text(token)
            if clean_token in ("售前", "presale"):
                return "presale"
            if clean_token in ("售中", "insale"):
                return "insale"
            if clean_token in ("售后", "aftersale"):
                return "aftersale"

        compact = cls._normalize_match_text(clean_text)
        if compact.startswith("售前补充"):
            return "presale"
        if compact.startswith("售中补充"):
            return "insale"
        if compact.startswith("售后补充"):
            return "aftersale"
        return ""

    def _filter_customer_service_by_scene(
        self,
        cs_list: List[CustomerServiceKnowledge],
        scene: Optional[str],
        fallback_to_all: bool = True,
    ) -> List[CustomerServiceKnowledge]:
        scene_key = self.normalize_customer_scene(scene)
        if not scene_key:
            return cs_list
        scored = []
        for index, cs in enumerate(cs_list):
            score = self._customer_scene_match_score(scene_key, cs.title or "", cs.tags or "")
            if score >= 0:
                scored.append((score, index, cs))
        if not scored:
            return cs_list if fallback_to_all else []
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [cs for _, _, cs in scored]

    def _filter_meta_entries_by_customer_scene(
        self,
        entries: List[KnowledgeMetaEntry],
        scene: Optional[str],
        fallback_to_all: bool = True,
    ) -> List[KnowledgeMetaEntry]:
        scene_key = self.normalize_customer_scene(scene)
        if not scene_key:
            return entries
        scored = []
        for index, entry in enumerate(entries):
            score = self._customer_scene_match_score(
                scene_key,
                self._normalize_scenario_name(entry.scenario),
                entry.sub_intent or "",
                entry.section_title or "",
                entry.tags or "",
            )
            if score >= 0:
                scored.append((score, index, entry))
        if not scored:
            return entries if fallback_to_all else []
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [entry for _, _, entry in scored]

    def get_full_scene_customer_service_knowledge(
        self,
        shop_id: int,
        goods_id: int,
        scene: Optional[str],
    ) -> str:
        """按商品ID和当前场景读取完整客服知识，用于RAG未命中后的兜底注入。"""
        scene_key = self.normalize_customer_scene(scene) or "presale"
        db_shop_id = self._resolve_shop_id(shop_id)

        with self.get_session() as session:
            meta_entries = list(session.scalars(
                select(KnowledgeMetaEntry).where(
                    and_(
                        KnowledgeMetaEntry.shop_id == db_shop_id,
                        KnowledgeMetaEntry.source_type == "customer_service",
                        KnowledgeMetaEntry.goods_id == goods_id,
                        KnowledgeMetaEntry.enabled == True,
                    )
                )
            ))
            scene_entries = self._filter_meta_entries_by_customer_scene(
                meta_entries,
                scene_key,
                fallback_to_all=False,
            )
            if not scene_entries:
                return ""

            ordered_source_ids = []
            seen_source_ids = set()
            for entry in sorted(scene_entries, key=lambda item: (-self._priority_value(item), item.id or 0)):
                if entry.source_id in seen_source_ids:
                    continue
                seen_source_ids.add(entry.source_id)
                ordered_source_ids.append(entry.source_id)

            if not ordered_source_ids:
                return ""

            cs_rows = list(session.scalars(
                select(CustomerServiceKnowledge).where(
                    and_(
                        CustomerServiceKnowledge.shop_id == db_shop_id,
                        CustomerServiceKnowledge.enabled == True,
                        CustomerServiceKnowledge.id.in_(ordered_source_ids),
                    )
                )
            ))
            cs_by_id = {item.id: item for item in cs_rows}
            ordered_cs = [cs_by_id[source_id] for source_id in ordered_source_ids if source_id in cs_by_id]
            if not ordered_cs:
                return ""

            output_parts = ["【客服知识】"]
            for index, cs in enumerate(ordered_cs, 1):
                title = (cs.title or "").split("/")[0].strip() or "命中客服知识"
                output_parts.append(f"{index}. {title}\n  {cs.content or ''}")
            return "\n\n".join(output_parts).strip()

    @staticmethod
    def _keyword_customer_service_entries(
        cs_list: List[CustomerServiceKnowledge],
        words: List[str],
        limit: int,
    ) -> List[CustomerServiceKnowledge]:
        if not words:
            return []
        matched = []
        lowered_words = [word.lower() for word in words if word]
        for index, cs in enumerate(cs_list):
            text = "\n".join([cs.title or "", cs.tags or "", cs.content or ""]).lower()
            if all(word in text for word in lowered_words):
                created_at = getattr(cs, "created_at", None) or datetime.min
                matched.append((created_at, index, cs))
        matched.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        return [cs for _, _, cs in matched[:limit]]

    def _search_customer_service_candidates(
        self,
        all_cs: List[CustomerServiceKnowledge],
        candidate_cs: List[CustomerServiceKnowledge],
        candidate_meta_entries: List[KnowledgeMetaEntry],
        query: str,
        words: List[str],
        db_shop_id: int,
        limit: int,
        scene: Optional[str] = None,
    ) -> List[CustomerServiceKnowledge]:
        scene_key = self.normalize_customer_scene(scene)
        ranked_meta_cs = self._rank_meta_entries(candidate_meta_entries, query, limit, scene_key=scene_key)
        if ranked_meta_cs:
            meta_source_ids = [entry.source_id for entry in ranked_meta_cs]
            structured_cs = []
            for source_id in meta_source_ids:
                match = next((item for item in all_cs if item.id == source_id), None)
                if match:
                    structured_cs.append(match)
        else:
            structured_cs = self._rank_customer_service_entries(
                candidate_cs,
                query,
                limit,
                scene_key=scene_key,
            )

        if structured_cs:
            return self._filter_relevant_customer_service_entries(
                structured_cs[:1],
                query,
                1,
                scene_key=scene_key,
            )

        vector_cs = self.vector_retriever.rank(
            namespace=f"customer_service_knowledge_{scene_key or 'all'}",
            shop_id=db_shop_id,
            query=query,
            items=self._customer_service_vector_items(candidate_cs),
            limit=limit,
        )
        keyword_cs = self._keyword_customer_service_entries(candidate_cs, words, limit)

        merged_cs = self._merge_ranked_results(vector_cs, keyword_cs, limit)
        return self._filter_relevant_customer_service_entries(
            merged_cs,
            query,
            limit,
            scene_key=scene_key,
        )

    def _rank_customer_service_entries(
        self,
        cs_list: List[CustomerServiceKnowledge],
        query: str,
        limit: int,
        scene_key: str = "",
    ) -> List[CustomerServiceKnowledge]:
        scored = []
        for index, cs in enumerate(cs_list):
            tags = cs.tags or ""
            match_score = self._structured_match_score(
                query=query,
                aliases=cs.title or "",
                answer=cs.content or "",
                tags=tags,
            )
            if match_score <= 0:
                continue
            score = match_score
            if "faq_split" in tags:
                score += 4
            scene_score = self._customer_scene_match_score(scene_key, cs.title or "", tags)
            if scene_score > 0:
                score += scene_score
            elif scene_score < 0:
                score += scene_score
            if scene_key:
                tags_text = cs.tags or ""
                scene_text = " ".join(str(part or "") for part in (cs.title, tags_text))
                tag_score = self._configured_tag_score(tags_text, default=25, scope="customer_service")
                if tag_score:
                    score += tag_score
                elif not self._primary_customer_scene(scene_text):
                    score -= 35
            if score > 0:
                scored.append((score, getattr(cs, "id", 0) or 0, index, cs))

        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [cs for score, _, _, cs in scored[:limit] if score >= 8]

    def _filter_relevant_customer_service_entries(
        self,
        cs_list: List[CustomerServiceKnowledge],
        query: str,
        limit: int,
        scene_key: str = "",
    ) -> List[CustomerServiceKnowledge]:
        if not cs_list or not str(query or "").strip():
            return cs_list[:limit]

        scored = []
        for index, cs in enumerate(cs_list):
            match_score = self._structured_match_score(
                query=query,
                aliases=cs.title or "",
                answer=cs.content or "",
                tags=cs.tags or "",
            )
            if match_score <= 0:
                continue
            score = match_score
            scene_score = self._customer_scene_match_score(scene_key, cs.title or "", cs.tags or "")
            if scene_score > 0:
                score += scene_score
            elif scene_score < 0:
                score += scene_score
            if scene_key:
                tags_text = cs.tags or ""
                scene_text = " ".join(str(part or "") for part in (cs.title, tags_text))
                tag_score = self._configured_tag_score(tags_text, default=25, scope="customer_service")
                if tag_score:
                    score += tag_score
                elif not self._primary_customer_scene(scene_text):
                    score -= 35
            if score >= 18:
                scored.append((score, getattr(cs, "id", 0) or 0, index, cs))

        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [cs for _, _, _, cs in scored[:limit]]

    def search_knowledge(
        self,
        shop_id: int,
        query: Optional[str] = None,
        goods_id: Optional[int] = None,
        limit: int = 10,
        search_scope: str = "all",
        scene: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = {
            "product_knowledge": [],
            "customer_service_knowledge": [],
        }
        db_shop_id = self._resolve_shop_id(shop_id)

        with self.get_session() as session:
            product: Optional[ProductKnowledge] = None
            if goods_id is not None:
                product_stmt = select(ProductKnowledge).where(
                    and_(
                        ProductKnowledge.shop_id == db_shop_id,
                        ProductKnowledge.goods_id == goods_id,
                    )
                )
                product = session.scalar(product_stmt)

            if goods_id is not None and search_scope in ("all", "product") and product:
                result["product_knowledge"] = [product]
                force_full_content = False
                meta_entries = list(session.scalars(
                    select(KnowledgeMetaEntry).where(
                        and_(
                            KnowledgeMetaEntry.shop_id == db_shop_id,
                            KnowledgeMetaEntry.source_type == "product",
                            KnowledgeMetaEntry.goods_id == goods_id,
                            KnowledgeMetaEntry.enabled == True,
                        )
                    )
                ))
                ranked_meta = self._rank_meta_entries(meta_entries, query or "", limit=1)
                if ranked_meta:
                    meta = ranked_meta[0]
                    matched_content = (
                        f"### {self._normalize_scenario_name(meta.scenario)}\n"
                        f"{meta.answer}"
                    )
                    if len(matched_content.strip()) < MIN_PRODUCT_HIT_CHARS:
                        fallback_content = self._rank_product_content(product, query, limit=5)
                        if fallback_content and len(fallback_content.strip()) > len(matched_content.strip()):
                            matched_content = fallback_content
                            force_full_content = False
                        else:
                            force_full_content = True
                else:
                    matched_content = self._rank_product_content(product, query, limit=5)
                    if matched_content and len(matched_content.strip()) < MIN_PRODUCT_HIT_CHARS:
                        force_full_content = True
                if matched_content:
                    result["product_knowledge_hits"] = {
                        product.id: matched_content,
                    }
                    result["product_force_full_content"] = {
                        product.id: force_full_content,
                    }
                else:
                    result["product_force_full_content"] = {
                        product.id: True,
                    }

            if goods_id is not None and search_scope == "product":
                return result

            if query and query.strip():
                words = [word.strip() for word in jieba.cut_for_search(query.strip()) if len(word.strip()) >= 2]

                if search_scope in ("all", "product"):
                    if goods_id is None or not product:
                        all_products_stmt = select(ProductKnowledge).where(ProductKnowledge.shop_id == db_shop_id)
                        all_products = list(session.scalars(all_products_stmt))
                        vector_products = self.vector_retriever.rank(
                            namespace="product_knowledge",
                            shop_id=db_shop_id,
                            query=query,
                            items=self._product_vector_items(all_products),
                            limit=limit,
                        )
                        keyword_products = []
                        if words:
                            product_conditions = [ProductKnowledge.shop_id == db_shop_id]
                            for word in words:
                                product_conditions.append(
                                    or_(
                                        ProductKnowledge.goods_name.contains(word),
                                        ProductKnowledge.extracted_content.contains(word),
                                    )
                                )
                            stmt_p = select(ProductKnowledge).where(and_(*product_conditions))\
                                .order_by(ProductKnowledge.created_at.desc())\
                                .limit(limit)
                            keyword_products = list(session.scalars(stmt_p))
                        result["product_knowledge"] = self._merge_ranked_results(
                            vector_products,
                            keyword_products,
                            limit,
                        )

                if search_scope in ("all", "customer_service"):
                    scene_key = self.normalize_customer_scene(scene) or "presale"
                    result["customer_service_scene"] = scene_key
                    customer_scope = "shop"
                    allow_shop_fallback = True
                    all_cs: List[CustomerServiceKnowledge] = []
                    meta_cs_entries: List[KnowledgeMetaEntry] = []
                    candidate_cs: List[CustomerServiceKnowledge] = []
                    candidate_meta_entries: List[KnowledgeMetaEntry] = []

                    if goods_id is not None and product:
                        product_family = self._infer_product_family(
                            " ".join(
                                part for part in [
                                    product.goods_name or "",
                                    product.specifications or "",
                                    product.extracted_content or "",
                                ]
                                if part
                            )
                        )
                        exact_meta_entries = list(session.scalars(
                            select(KnowledgeMetaEntry).where(
                                and_(
                                    KnowledgeMetaEntry.shop_id == db_shop_id,
                                    KnowledgeMetaEntry.source_type == "customer_service",
                                    KnowledgeMetaEntry.goods_id == goods_id,
                                    KnowledgeMetaEntry.enabled == True,
                                )
                            )
                        ))
                        if exact_meta_entries:
                            candidate_meta_entries = self._filter_meta_entries_by_customer_scene(
                                exact_meta_entries,
                                scene_key,
                                fallback_to_all=False,
                            )
                            source_ids = sorted({entry.source_id for entry in candidate_meta_entries})
                            if source_ids:
                                cs_stmt = select(CustomerServiceKnowledge).where(
                                    and_(
                                        CustomerServiceKnowledge.shop_id == db_shop_id,
                                        CustomerServiceKnowledge.enabled == True,
                                        CustomerServiceKnowledge.id.in_(source_ids),
                                    )
                                )
                                candidate_cs = list(session.scalars(cs_stmt))
                            customer_scope = f"goods_id:{goods_id}"
                            allow_shop_fallback = False
                        elif product_family:
                            family_meta_entries = list(session.scalars(
                                select(KnowledgeMetaEntry).where(
                                    and_(
                                        KnowledgeMetaEntry.shop_id == db_shop_id,
                                        KnowledgeMetaEntry.source_type == "customer_service",
                                        KnowledgeMetaEntry.product_family == product_family,
                                        KnowledgeMetaEntry.goods_id.is_(None),
                                        KnowledgeMetaEntry.enabled == True,
                                    )
                                )
                            ))
                            if family_meta_entries:
                                candidate_meta_entries = self._filter_meta_entries_by_customer_scene(
                                    family_meta_entries,
                                    scene_key,
                                    fallback_to_all=False,
                                )
                                source_ids = sorted({entry.source_id for entry in candidate_meta_entries})
                                if source_ids:
                                    cs_stmt = select(CustomerServiceKnowledge).where(
                                        and_(
                                            CustomerServiceKnowledge.shop_id == db_shop_id,
                                            CustomerServiceKnowledge.enabled == True,
                                            CustomerServiceKnowledge.id.in_(source_ids),
                                        )
                                    )
                                    candidate_cs = list(session.scalars(cs_stmt))
                                customer_scope = f"product_family:{product_family}"
                                allow_shop_fallback = False

                    if not candidate_meta_entries and not candidate_cs and allow_shop_fallback:
                        all_cs_stmt = select(CustomerServiceKnowledge).where(
                            and_(
                                CustomerServiceKnowledge.shop_id == db_shop_id,
                                CustomerServiceKnowledge.enabled == True,
                                or_(
                                    CustomerServiceKnowledge.tags.is_(None),
                                    ~CustomerServiceKnowledge.tags.contains("goods_id:"),
                                ),
                            )
                        )
                        all_cs = list(session.scalars(all_cs_stmt))
                        meta_cs_stmt = select(KnowledgeMetaEntry).where(
                            and_(
                                KnowledgeMetaEntry.shop_id == db_shop_id,
                                KnowledgeMetaEntry.source_type == "customer_service",
                                KnowledgeMetaEntry.goods_id.is_(None),
                                KnowledgeMetaEntry.enabled == True,
                            )
                        )
                        meta_cs_entries = list(session.scalars(meta_cs_stmt))
                        candidate_cs = self._filter_customer_service_by_scene(
                            all_cs,
                            scene_key,
                            fallback_to_all=True,
                        )
                        candidate_meta_entries = self._filter_meta_entries_by_customer_scene(
                            meta_cs_entries,
                            scene_key,
                            fallback_to_all=True,
                        )
                        customer_scope = "shop"
                        allow_shop_fallback = True

                    result["customer_service_scope"] = customer_scope
                    result["customer_service_knowledge"] = self._search_customer_service_candidates(
                        all_cs=candidate_cs or all_cs,
                        candidate_cs=candidate_cs,
                        candidate_meta_entries=candidate_meta_entries,
                        query=query,
                        words=words,
                        db_shop_id=db_shop_id,
                        limit=limit,
                        scene=scene_key,
                    )
                    if (
                        not result["customer_service_knowledge"]
                        and allow_shop_fallback
                        and scene_key
                        and (len(candidate_cs) < len(all_cs) or len(candidate_meta_entries) < len(meta_cs_entries))
                    ):
                        result["customer_service_knowledge"] = self._search_customer_service_candidates(
                            all_cs=all_cs,
                            candidate_cs=all_cs,
                            candidate_meta_entries=meta_cs_entries,
                            query=query,
                            words=words,
                            db_shop_id=db_shop_id,
                            limit=limit,
                            scene=None,
                        )
                        if result["customer_service_knowledge"]:
                            logger.info(
                                f"场景客服知识未命中，已回退全量检索: shop_id={db_shop_id}, "
                                f"scene={scene_key}, query_chars={len(str(query or ''))}"
                            )
                return result

            if search_scope in ("all", "product"):
                stmt_p = select(ProductKnowledge).where(ProductKnowledge.shop_id == db_shop_id)\
                    .order_by(ProductKnowledge.created_at.desc())\
                    .limit(limit)
                result["product_knowledge"] = list(session.scalars(stmt_p))

            if search_scope in ("all", "customer_service"):
                stmt_cs = select(CustomerServiceKnowledge).where(
                    and_(
                        CustomerServiceKnowledge.shop_id == db_shop_id,
                        CustomerServiceKnowledge.enabled == True,
                    )
                ).order_by(CustomerServiceKnowledge.created_at.desc())\
                    .limit(limit)
                result["customer_service_knowledge"] = list(session.scalars(stmt_cs))

        return result

    def format_search_result(
        self,
        result: Dict[str, Any],
    ) -> str:
        """
        将检索结果格式化为Agent可读的字符串

        Args:
            result: search_knowledge 返回的结果

        Returns:
            格式化后的字符串
        """
        if not isinstance(result, dict):
            return "未找到相关知识。"

        output_parts = []

        products = result.get("product_knowledge", [])
        product_hits = result.get("product_knowledge_hits", {})
        product_force_full_content = result.get("product_force_full_content", {})
        product_hits = product_hits if isinstance(product_hits, dict) else {}
        product_force_full_content = (
            product_force_full_content
            if isinstance(product_force_full_content, dict)
            else {}
        )
        if products:
            product_parts = []
            for i, p in enumerate(products, 1):
                if not hasattr(p, "goods_name") or not hasattr(p, "goods_id"):
                    continue
                info = []
                product_id = getattr(p, "id", None)
                price = getattr(p, "price", None)
                extracted_content = getattr(p, "extracted_content", None)
                info.append(
                    f"{i}. {str(getattr(p, 'goods_name', '') or '')} "
                    f"(ID: {str(getattr(p, 'goods_id', '') or '')})"
                )
                if price:
                    info.append(f"  价格: {str(price)}")
                matched_content = product_hits.get(product_id)
                force_full_content = bool(product_force_full_content.get(product_id))
                if matched_content and not force_full_content:
                    info.append(f"  【与客户问题最相关的商品知识】\n  {str(matched_content)}")
                elif extracted_content:
                    # 截断避免太长
                    content = str(extracted_content)
                    max_content_length = 3200 if force_full_content else 1800
                    if len(content) > max_content_length:
                        content = content[:max_content_length] + "..."
                    info.append(f"  {content}")
                product_parts.append("\n".join(info))
            if product_parts:
                output_parts.append("【产品知识】")
                output_parts.extend(product_parts)
                output_parts.append("")

        cs_list = result.get("customer_service_knowledge", [])
        if cs_list:
            cs_parts = []
            for i, cs in enumerate(cs_list, 1):
                if not hasattr(cs, "content"):
                    continue
                info = []
                title = str(getattr(cs, "title", "") or "").split("/")[0].strip() or "命中客服知识"
                info.append(f"{i}. {title}")
                content = str(getattr(cs, "content", "") or "")
                if len(content) > 800:
                    content = content[:800] + "..."
                info.append(f"  {content}")
                cs_parts.append("\n".join(info))
            if cs_parts:
                output_parts.append("【客服知识】")
                output_parts.extend(cs_parts)
                output_parts.append("")

        if not output_parts:
            return "未找到相关知识。"

        return "\n".join(output_parts).strip()

    def get_all_shops(self) -> List[Shop]:
        """获取所有店铺列表（用于UI选择器）"""
        with self.get_session() as session:
            stmt = select(Shop).order_by(Shop.shop_name.asc())
            return list(session.scalars(stmt))

    # ========== 新场景知识检索 ==========

    _SCENE_MODEL_MAP = {
        "presale": PresaleKnowledge,
        "insale": InsaleKnowledge,
        "aftersale": AftersaleKnowledge,
    }
    _SCENE_TABLE_MAP = {
        "presale": "presale_knowledge",
        "insale": "insale_knowledge",
        "aftersale": "aftersale_knowledge",
    }
    _EMBED_TEXT_FIELDS = ("section_title", "sub_intent", "aliases", "answer")

    # ── Embedding 构建 ──

    @staticmethod
    def _build_embedding_text(entry) -> str:
        """拼接用于生成 embedding 的文本。"""
        parts = []
        for field in ("section_title", "sub_intent", "aliases", "answer"):
            val = getattr(entry, field, None)
            if val:
                parts.append(str(val))
        return "\n".join(parts)

    @staticmethod
    def _content_hash(text: str) -> str:
        import hashlib
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _cosine_similarity(left, right) -> float:
        from math import sqrt
        left_values = list(left)
        right_values = list(right)
        if not left_values or not right_values or len(left_values) != len(right_values):
            return 0.0
        numerator = sum(a * b for a, b in zip(left_values, right_values))
        left_norm = sqrt(sum(v * v for v in left_values))
        right_norm = sqrt(sum(v * v for v in right_values))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def build_scene_embeddings(self, scene: str = None, shop_id: int = None, batch_size: int = 100) -> dict:
        """为场景知识生成 embedding 并写入 scene_knowledge_embeddings 表。

        按 content_hash 去重：相同内容只调用一次 embedding，复用到所有 knowledge_id。

        Args:
            scene: 指定场景，None=全部
            shop_id: 指定店铺，None=全部
            batch_size: 每批调用 embedding 的条数

        Returns:
            {"total": N, "skipped": N, "created": N, "failed": N, "embed_calls": N}
        """
        import struct
        import requests as _requests

        scenes = [scene] if scene else ["presale", "insale", "aftersale"]
        stats = {"total": 0, "skipped": 0, "created": 0, "failed": 0, "embed_calls": 0}

        for sc in scenes:
            model = self._SCENE_MODEL_MAP.get(sc)
            table_name = self._SCENE_TABLE_MAP.get(sc)
            if not model:
                continue

            # 1. 加载所有启用条目，计算 embed_text + content_hash
            with self.get_session() as session:
                stmt = select(model).where(model.enabled == True)
                if shop_id is not None:
                    db_sid = self._resolve_shop_id(shop_id)
                    stmt = stmt.where(model.shop_id == db_sid)
                entries = list(session.scalars(stmt))

            # hash -> {embed_text, entries: [(entry, table_name)]}
            hash_groups = {}
            for entry in entries:
                stats["total"] += 1
                embed_text = self._build_embedding_text(entry)
                if not embed_text.strip():
                    stats["skipped"] += 1
                    continue
                c_hash = self._content_hash(embed_text)
                if c_hash not in hash_groups:
                    hash_groups[c_hash] = {"embed_text": embed_text, "entries": []}
                hash_groups[c_hash]["entries"].append((entry, table_name))

            # 2. 查询该 scene 已有的 content_hash
            with self.get_session() as session:
                existing_hashes = set(
                    row[0] for row in session.execute(
                        select(SceneKnowledgeEmbedding.content_hash).where(
                            SceneKnowledgeEmbedding.scene == sc
                        )
                    ).all()
                )

            # 3. 分离：已有 hash（直接复用）vs 需要新生成
            to_generate = {}  # hash -> embed_text
            for c_hash, info in hash_groups.items():
                if c_hash in existing_hashes:
                    # 已有 embedding，只写映射行
                    self._write_mapping_rows(sc, c_hash, info["entries"])
                    stats["skipped"] += len(info["entries"])
                else:
                    to_generate[c_hash] = info

            if not to_generate:
                logger.info(f"[embedding构建] scene={sc} total={stats['total']} "
                            f"reused={stats['skipped']} created=0 embed_calls=0")
                continue

            # 4. 批量生成 embedding（按 unique hash，不是按知识行）
            embedding_model = self.vector_retriever.embedding_model
            embed_url = self.vector_retriever.embedding_url
            timeout = self.vector_retriever.timeout_seconds

            hash_list = list(to_generate.keys())
            for batch_start in range(0, len(hash_list), batch_size):
                batch_hashes = hash_list[batch_start:batch_start + batch_size]
                texts = [to_generate[h]["embed_text"] for h in batch_hashes]

                try:
                    resp = _requests.post(
                        embed_url,
                        json={"input": texts, "model": embedding_model},
                        timeout=max(timeout, 60),
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                    if not isinstance(payload, dict):
                        raise RuntimeError(f"embedding response payload is not a dict: {type(payload).__name__}")
                    vectors = payload.get("data")
                    if not isinstance(vectors, list):
                        raise RuntimeError(f"embedding response missing data list: {payload!r}")
                    vectors.sort(key=lambda x: x["index"])
                    if len(vectors) < len(batch_hashes):
                        raise RuntimeError(
                            f"embedding response data too short: expected={len(batch_hashes)}, got={len(vectors)}"
                        )
                    stats["embed_calls"] += len(batch_hashes)
                except Exception as exc:
                    logger.warning(
                        f"[embedding批次失败] scene={sc} batch={batch_start//batch_size} error={_sanitize_for_log(exc)}"
                    )
                    stats["failed"] += sum(len(to_generate[h]["entries"]) for h in batch_hashes)
                    continue

                # 5. 写入：每个 hash 一条主记录 + 所有 knowledge_id 映射行
                for i, c_hash in enumerate(batch_hashes):
                    vec = vectors[i]["embedding"]
                    blob = struct.pack(f"{len(vec)}f", *vec)
                    info = to_generate[c_hash]
                    entries_list = info["entries"]

                    with self.get_session() as session:
                        for entry, tbl in entries_list:
                            row = SceneKnowledgeEmbedding(
                                scene=sc,
                                knowledge_table=tbl,
                                knowledge_id=entry.id,
                                shop_id=entry.shop_id,
                                goods_id=entry.goods_id,
                                embedding_text=info["embed_text"],
                                embedding=blob,
                                embedding_model=embedding_model,
                                embedding_dim=len(vec),
                                content_hash=c_hash,
                            )
                            session.add(row)
                        session.commit()
                        stats["created"] += len(entries_list)

            logger.info(f"[embedding构建] scene={sc} total={stats['total']} "
                        f"reused={stats['skipped']} created={stats['created']} "
                        f"embed_calls={stats['embed_calls']} failed={stats['failed']}")

        return stats

    def _write_mapping_rows(self, scene: str, content_hash: str, entries: list):
        """为已有 embedding 的 hash 写入 knowledge_id 映射行（如果不存在）。"""
        with self.get_session() as session:
            existing_ids = set(
                row[0] for row in session.execute(
                    select(SceneKnowledgeEmbedding.knowledge_id).where(and_(
                        SceneKnowledgeEmbedding.scene == scene,
                        SceneKnowledgeEmbedding.content_hash == content_hash,
                    ))
                ).all()
            )
            # 获取已有 embedding 的 blob
            ref_row = session.scalar(
                select(SceneKnowledgeEmbedding).where(and_(
                    SceneKnowledgeEmbedding.scene == scene,
                    SceneKnowledgeEmbedding.content_hash == content_hash,
                )).limit(1)
            )
            if not ref_row:
                return
            for entry, tbl in entries:
                if entry.id in existing_ids:
                    continue
                row = SceneKnowledgeEmbedding(
                    scene=scene,
                    knowledge_table=tbl,
                    knowledge_id=entry.id,
                    shop_id=entry.shop_id,
                    goods_id=entry.goods_id,
                    embedding_text=ref_row.embedding_text,
                    embedding=ref_row.embedding,
                    embedding_model=ref_row.embedding_model,
                    embedding_dim=ref_row.embedding_dim,
                    content_hash=content_hash,
                )
                session.add(row)
            session.commit()

    def search_scene_knowledge(
        self,
        scene: str,
        shop_id: int,
        goods_id: Optional[int] = None,
        query: Optional[str] = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        按场景检索知识（新表）。

        Args:
            scene: presale / insale / aftersale
            shop_id: 店铺 ID（原始值，会自动解析为 DB 内部 ID）
            goods_id: 商品 ID，可选
            query: 客户问题，可选
            limit: 返回条数，默认 3

        Returns:
            结果列表，每条包含 id/scene/goods_id/sub_intent/aliases/answer/
            section_title/tags/score/match_type/source_type/source_id/source_meta_id
        """
        scene_key = str(scene or "").lower().strip()
        model = self._SCENE_MODEL_MAP.get(scene_key)
        if not model:
            logger.warning(f"search_scene_knowledge: 未知场景 '{scene}'，允许值: presale/insale/aftersale")
            return []

        db_shop_id = self._resolve_shop_id(shop_id)
        results: List[Dict[str, Any]] = []

        with self.get_session() as session:
            # ── 第一步：查商品专属知识 ──
            specific_entries: List = []
            if goods_id is not None:
                stmt = select(model).where(and_(
                    model.shop_id == db_shop_id,
                    model.goods_id == goods_id,
                    model.enabled == True,
                ))
                specific_entries = list(session.scalars(stmt))

            # ── 第二步：查店铺通用知识 ──
            generic_stmt = select(model).where(and_(
                model.shop_id == db_shop_id,
                model.goods_id.is_(None),
                model.enabled == True,
            ))
            generic_entries = list(session.scalars(generic_stmt))

            # ── 合并：专属优先 ──
            all_entries = specific_entries + generic_entries
            if not all_entries:
                return []

            # ── 排序 ──
            ranked = self._rank_scene_entries(
                all_entries,
                self._knowledge_match_query(query),
                scene_key,
                goods_id,
            )

            # ── 截取 top N ──
            for entry, rule_score, vector_score, final_score, match_type in ranked[:limit]:
                results.append({
                    "id": entry.id,
                    "scene": scene_key,
                    "goods_id": entry.goods_id,
                    "sub_intent": entry.sub_intent or "",
                    "aliases": entry.aliases or "",
                    "answer": entry.answer or "",
                    "section_title": entry.section_title or "",
                    "tags": entry.tags or "",
                    "rule_score": rule_score,
                    "vector_score": round(vector_score, 4) if vector_score else 0,
                    "score": final_score,
                    "match_type": match_type,
                    "source_type": entry.source_type or "",
                    "source_id": entry.source_id,
                    "source_meta_id": entry.source_meta_id,
                })

        return results

    def list_scene_knowledge_by_goods(
        self,
        scene: str,
        shop_id: int,
        goods_id: int,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """List editable scene knowledge rows for one product."""
        scene_key = str(scene or "").lower().strip()
        model = self._SCENE_MODEL_MAP.get(scene_key)
        if not model:
            return []

        db_shop_id = self._resolve_shop_id(shop_id)
        with self.get_session() as session:
            rows = list(session.scalars(
                select(model).where(and_(
                    model.shop_id == db_shop_id,
                    model.goods_id == goods_id,
                )).order_by(
                    model.priority.desc(),
                    model.section_title.asc(),
                    model.id.asc(),
                ).limit(limit)
            ))

        return [
            {
                "id": row.id,
                "scene": scene_key,
                "goods_id": row.goods_id,
                "sub_intent": row.sub_intent or "",
                "aliases": row.aliases or "",
                "answer": row.answer or "",
                "section_title": row.section_title or "",
                "tags": row.tags or "",
                "priority": self._priority_value(row),
                "enabled": as_bool(row.enabled, True),
                "source_type": row.source_type or "",
                "source_id": row.source_id,
                "source_meta_id": row.source_meta_id,
            }
            for row in rows
        ]

    def update_scene_knowledge(
        self,
        scene: str,
        entry_id: int,
        aliases: str,
        answer: str,
        sub_intent: Optional[str] = None,
        section_title: Optional[str] = None,
        priority: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> bool:
        """Update one scene knowledge row from the management UI."""
        scene_key = str(scene or "").lower().strip()
        model = self._SCENE_MODEL_MAP.get(scene_key)
        if not model:
            return False

        with self.get_session() as session:
            row = session.get(model, entry_id)
            if not row:
                return False
            row.aliases = aliases
            row.answer = answer
            row.sub_intent = sub_intent or ""
            row.section_title = section_title or ""
            if priority is not None:
                row.priority = as_int(priority, self._priority_value(row))
            if enabled is not None:
                row.enabled = as_bool(enabled, as_bool(row.enabled, True))
            row.updated_at = datetime.now()
            session.commit()
            return True

    def update_scene_knowledge_entries(
        self,
        scene: str,
        entry_ids: List[int],
        aliases: str,
        answer: str,
        sub_intent: Optional[str] = None,
        section_title: Optional[str] = None,
        priority: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> int:
        """Update multiple scene knowledge rows, returning updated count."""
        scene_key = str(scene or "").lower().strip()
        model = self._SCENE_MODEL_MAP.get(scene_key)
        if not model or not entry_ids:
            return 0

        normalized_ids = [int(entry_id) for entry_id in entry_ids if entry_id]
        if not normalized_ids:
            return 0

        with self.get_session() as session:
            rows = list(session.scalars(
                select(model).where(model.id.in_(normalized_ids))
            ))
            for row in rows:
                row.aliases = aliases
                row.answer = answer
                row.sub_intent = sub_intent or ""
                row.section_title = section_title or ""
                if priority is not None:
                    row.priority = as_int(priority, self._priority_value(row))
                if enabled is not None:
                    row.enabled = as_bool(enabled, as_bool(row.enabled, True))
                row.updated_at = datetime.now()
            session.commit()
            return len(rows)

    def _rank_scene_entries(
        self,
        entries: List,
        query: Optional[str],
        scene_key: str,
        goods_id: Optional[int],
    ) -> List[tuple]:
        """
        对场景知识条目排序。

        商品专属作为第一排序键：所有专属条目排在通用条目之前，
        同组内按 final_score 降序。

        返回: [(entry, rule_score, vector_score, final_score, match_type), ...]
        """
        # 拆分专属 vs 通用
        specific: List = []
        generic: List = []
        for entry in entries:
            if goods_id is not None and entry.goods_id == goods_id:
                specific.append(entry)
            else:
                generic.append(entry)

        # 分别评分排序
        ranked_specific = self._score_entries(specific, query, goods_id, scene_key)
        ranked_generic = self._score_entries(generic, query, goods_id, scene_key)

        # 专属全部排在通用前面
        return ranked_specific + ranked_generic

    def _score_entries(
        self,
        entries: List,
        query: Optional[str],
        goods_id: Optional[int],
        scene_key: str = "",
    ) -> List[tuple]:
        """对一组条目评分并排序。返回 [(entry, rule_score, vector_score, final_score, match_type), ...]

        混合检索：规则评分 + 向量语义评分。
        """
        if not query or not query.strip():
            scored = []
            for entry in entries:
                score = self._priority_value(entry) * 10
                scored.append((entry, score, 0, score, "priority"))
            scored.sort(key=lambda x: -x[3])
            return scored

        query_clean = self._normalize_match_text(query)
        hints = self._query_intent_hints(query)
        pre_scored: List[tuple] = []  # (entry, rule_score, match_type)

        # ── 第一阶段：规则评分 ──
        for entry in entries:
            score = 0
            match_type = "none"
            matched = False

            # 1. aliases 精确匹配（最高权重）
            alias_score = self._alias_match_score(query_clean, entry.aliases or "")
            if alias_score > 0:
                score += alias_score
                match_type = "alias_exact" if alias_score >= 200 else "alias_partial"
                matched = True

            # 2. 简单关键词匹配
            keyword_score = self._keyword_match_score(query, entry)
            if keyword_score > 0 and match_type == "none":
                score += keyword_score
                match_type = "keyword"
                matched = True
            elif keyword_score > 0:
                score += keyword_score
                matched = True

            # 3. 意图调整（boost/penalize）
            intent_adjustment = self._intent_score_adjustment(hints, entry, scene_key, query=query_clean)
            score += intent_adjustment
            if intent_adjustment > 0:
                matched = True

            if not matched and not self.vector_retriever:
                continue

            # 4. priority 只对已匹配条目加分，避免无关高优先级条目污染结果
            if matched:
                score += self._priority_value(entry) * 10

            if score > 0 or (not matched and score == 0 and self.vector_retriever):
                pre_scored.append((entry, score, match_type))

        # ── 第二阶段：向量语义评分（混合检索） ──
        scored = self._apply_vector_scores(pre_scored, query, scene_key, goods_id)
        scored = [item for item in scored if item[3] > 0]
        scored = [
            item
            for item in scored
            if self._passes_retrieval_quality_gate(query, item[0], item[1], item[2], item[4])
        ]

        scored.sort(key=lambda x: -x[3])
        return scored

    _VECTOR_SCORE_THRESHOLD = 0.45
    _PRICE_KEYWORDS = frozenset(("价格", "多少钱", "优惠", "售价", "页面价格", "券", "九块九", "9块9", "9.9", "990元"))
    _PRICE_SECTION_KEYWORDS = frozenset(("价格", "多少钱", "优惠", "售价", "券"))

    def _apply_vector_scores(
        self,
        pre_scored: List[tuple],
        query: str,
        scene_key: str,
        goods_id: Optional[int],
    ) -> List[tuple]:
        """对已评分条目追加向量语义分。失败时静默回退到纯规则。

        保护规则：
        - vector_score < 0.45 不参与融合
        - 非价格 query + 价格类条目 → 不给 vector bonus
        - match_type 只在 vector_score >= 0.45 时标记为 hybrid/vector

        输入: [(entry, rule_score, match_type), ...]
        输出: [(entry, rule_score, vector_score, final_score, match_type), ...]
        """
        if not pre_scored or not self.vector_retriever:
            return [(e, rs, 0, rs, mt) for e, rs, mt in pre_scored]

        table_name = self._SCENE_TABLE_MAP.get(scene_key)
        if not table_name:
            return [(e, rs, 0, rs, mt) for e, rs, mt in pre_scored]

        # 生成 query embedding
        try:
            query_vec = self.vector_retriever._embed(query.strip())
        except Exception as exc:
            logger.debug(f"[hybrid] query embedding failed, fallback to rule-only: {_sanitize_for_log(exc)}")
            return [(e, rs, 0, rs, mt) for e, rs, mt in pre_scored]

        if not query_vec:
            return [(e, rs, 0, rs, mt) for e, rs, mt in pre_scored]

        # 判断 query 是否有价格意图
        q_lower = self._normalize_match_text(query)
        query_has_price_intent = any(kw in q_lower for kw in self._PRICE_KEYWORDS)

        import struct
        result = []
        for entry, rule_score, match_type in pre_scored:
            vector_score = 0.0
            try:
                with self.get_session() as session:
                    row = session.scalar(
                        select(SceneKnowledgeEmbedding).where(and_(
                            SceneKnowledgeEmbedding.scene == scene_key,
                            SceneKnowledgeEmbedding.knowledge_table == table_name,
                            SceneKnowledgeEmbedding.knowledge_id == entry.id,
                        ))
                    )
                if row and row.embedding:
                    entry_vec = struct.unpack(f"{row.embedding_dim}f", row.embedding)
                    vector_score = self._cosine_similarity(query_vec, entry_vec)
            except Exception as exc:
                logger.debug(
                    f"[hybrid] vector score skipped: scene={scene_key}, "
                    f"table={table_name}, knowledge_id={getattr(entry, 'id', None)}, error={_sanitize_for_log(exc)}"
                )

            # 保护 1: 低相似度不参与融合
            if vector_score < self._VECTOR_SCORE_THRESHOLD:
                result.append((entry, rule_score, vector_score, rule_score, match_type))
                continue

            # 保护 2: 非价格 query + 价格类条目 → 不给 vector bonus
            combined = " ".join(filter(None, [
                getattr(entry, "section_title", ""),
                getattr(entry, "sub_intent", ""),
                getattr(entry, "aliases", ""),
                getattr(entry, "answer", ""),
            ]))
            is_price_entry = any(kw in combined for kw in self._PRICE_SECTION_KEYWORDS)
            if is_price_entry and not query_has_price_intent:
                result.append((entry, rule_score, vector_score, rule_score, match_type))
                continue

            # 融合
            vector_bonus = int(vector_score * 500)
            final_score = rule_score + vector_bonus
            new_type = "hybrid" if match_type != "none" else "vector"
            result.append((entry, rule_score, vector_score, final_score, new_type))

        return result

    async def apply_vector_scores_async(
        self,
        pre_scored: List[tuple],
        query: str,
        scene_key: str,
        goods_id: Optional[int],
    ) -> List[tuple]:
        """Async-safe wrapper for vector scoring callers already on an event loop."""
        import asyncio

        return await asyncio.to_thread(
            self._apply_vector_scores,
            pre_scored,
            query,
            scene_key,
            goods_id,
        )

    @staticmethod
    def _knowledge_match_query(query: Optional[str]) -> str:
        """只用客户真实文本做知识匹配，避免商品卡片标题/价格污染检索。"""
        text = str(query or "").strip()
        marker = "客户消息："
        if marker not in text:
            return KnowledgeService._normalize_common_traditional(text)

        customer_part = text.split(marker, 1)[1]
        stop_markers = ("\n商品卡片：", "\n商品：", "\n订单信息：", "\n物流信息：")
        for stop in stop_markers:
            if stop in customer_part:
                customer_part = customer_part.split(stop, 1)[0]
        return KnowledgeService._normalize_common_traditional(customer_part.strip() or text)

    @staticmethod
    def _normalize_common_traditional(text: str) -> str:
        """Normalize common traditional Chinese terms seen in customer questions."""
        if not text:
            return text
        replacements = {
            "發": "发",
            "貨": "货",
            "遞": "递",
            "嗎": "吗",
            "幾": "几",
            "個": "个",
            "這": "这",
            "款": "款",
            "風": "风",
            "電": "电",
            "續": "续",
            "航": "航",
            "時": "时",
            "間": "间",
            "長": "长",
            "嗎": "吗",
            "麼": "么",
            "什麼": "什么",
            "沖": "冲",
            "滿": "满",
            "檔": "档",
            "顏": "颜",
            "色": "色",
            "質": "质",
            "保": "保",
            "開": "开",
            "關": "关",
            "聲": "声",
            "噪": "噪",
            "壞": "坏",
            "轉": "转",
            "葉": "叶",
            "繩": "绳",
            "無": "无",
            "帶": "带",
        }
        normalized = text
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        return normalized

    def _alias_match_score(self, query_clean: str, aliases: str) -> int:
        """aliases 匹配评分。"""
        if not query_clean or not aliases:
            return 0

        best_score = 0
        for alias in re.split(r"[/|;；\n\r]+", aliases):
            alias_clean = self._normalize_match_text(alias)
            if len(alias_clean) < 2:
                continue

            if alias_clean == query_clean:
                # 完全匹配
                best_score = max(best_score, 240 + min(len(alias_clean), 30))
            elif len(alias_clean) >= 4 and alias_clean in query_clean:
                # alias 是 query 的子串
                best_score = max(best_score, 115 + min(len(alias_clean), 12))
            elif len(query_clean) >= 6 and query_clean in alias_clean:
                # query 是 alias 的子串
                best_score = max(best_score, 80 + min(len(query_clean), 12))
            elif alias_clean in query_clean or query_clean in alias_clean:
                # 弱匹配
                best_score = max(best_score, 8 + min(len(alias_clean), 6))

        return best_score

    def _keyword_match_score(self, query: str, entry) -> int:
        """简单关键词匹配评分。"""
        words = self._search_terms(query)
        score = self._parameter_type_match_score(query, entry)
        score += self._version_name_match_score(query, entry)
        score += self._action_type_match_score(query, entry)
        score += self._complaint_type_match_score(query, entry)
        score += self._case_type_match_score(query, entry)
        if not words:
            return score

        return score + self._bm25_like_match_score(words, entry)

    @classmethod
    def _bm25_like_match_score(cls, words: List[str], entry) -> int:
        """轻量词项评分：比简单 contains 更重视别名/标题和较长关键词。"""
        if not words:
            return 0
        fields = (
            (getattr(entry, "aliases", "") or "", 12),
            (getattr(entry, "section_title", "") or "", 8),
            (getattr(entry, "sub_intent", "") or "", 8),
            (getattr(entry, "tags", "") or "", 5),
            (getattr(entry, "answer", "") or "", 3),
        )
        score = 0
        seen_terms: set[str] = set()
        for raw_word in words:
            term = cls._normalize_match_text(raw_word)
            if len(term) < 2 or term in seen_terms:
                continue
            seen_terms.add(term)
            term_score = 0
            length_bonus = min(len(term), 6)
            for text, weight in fields:
                clean_text = cls._normalize_match_text(text)
                if not clean_text or term not in clean_text:
                    continue
                term_score = max(term_score, weight + length_bonus)
            score += term_score
        return score

    _PARAMETER_TYPE_QUERY_HINTS = {}

    @classmethod
    def _configured_type_query_hints(cls, config_key: str, defaults: Dict[str, Any]) -> Dict[str, tuple[str, ...]]:
        configured = get_config(config_key, None)
        if configured is None:
            configured = defaults
        elif not isinstance(configured, dict):
            configured = defaults

        result: Dict[str, tuple[str, ...]] = {}
        for name, terms in configured.items():
            if isinstance(terms, str):
                terms = (terms,)
            if not isinstance(terms, (list, tuple, set)):
                continue
            cleaned = tuple(str(term or "").strip() for term in terms if str(term or "").strip())
            if cleaned:
                result[str(name)] = cleaned
        return result

    @classmethod
    def _query_parameter_types(cls, query: str) -> set[str]:
        clean = cls._normalize_match_text(query)
        types: set[str] = set()
        hints_by_type = cls._configured_type_query_hints(
            "knowledge.parameter_type_query_hints",
            cls._PARAMETER_TYPE_QUERY_HINTS,
        )
        for parameter_type, hints in hints_by_type.items():
            if any(cls._normalize_match_text(hint) in clean for hint in hints):
                types.add(parameter_type)
        if "battery_capacity" in types:
            types.discard("wind")
        if "duration" in types:
            types.discard("charging")
        return types

    @staticmethod
    def _entry_parameter_types(entry) -> set[str]:
        tags = str(getattr(entry, "tags", "") or "").lower()
        found: set[str] = set()
        for match in re.finditer(r"parameter[_-]?type\s*[:=]\s*([a-z0-9_-]+)", tags):
            value = match.group(1).strip()
            if value:
                found.add(value)
        return found

    _VERSION_NAME_QUERY_RE = re.compile(r"(?i)(?:^|[^a-z0-9])(?:\d+[a-z]|[a-z]\d+)(?:[^a-z0-9]|$)")

    @classmethod
    def _query_version_name_types(cls, query: str) -> set[str]:
        clean = cls._normalize_match_text(query)
        types: set[str] = set()
        if not clean:
            return types
        if cls._VERSION_NAME_QUERY_RE.search(clean):
            types.add("version_name")
        for token in cls._version_name_query_tokens():
            if token and token in clean:
                types.add("version_name")
                break
        return types

    @staticmethod
    def _entry_version_name_types(entry) -> set[str]:
        tags = str(getattr(entry, "tags", "") or "").lower()
        found: set[str] = set()
        for match in re.finditer(r"(?:parameter|case|model|version)[_-]?type\s*[:=]\s*([a-z0-9_-]+)", tags):
            value = match.group(1).strip()
            if value:
                found.add(value)
        if any(token in tags for token in ("version_name", "model_variant")):
            found.add("version_name")
        return found

    @classmethod
    def _version_name_match_score(cls, query: str, entry) -> int:
        query_types = cls._query_version_name_types(query)
        entry_types = cls._entry_version_name_types(entry)
        if not query_types or not entry_types:
            return 0
        if query_types & entry_types:
            return 180
        if entry_types:
            return -80
        if cls._entry_parameter_types(entry):
            return -120
        return 0

    @classmethod
    def _parameter_type_match_score(cls, query: str, entry) -> int:
        query_types = cls._query_parameter_types(query)
        entry_types = cls._entry_parameter_types(entry)
        if not query_types or not entry_types:
            return 0
        if query_types & entry_types:
            return 120
        return -80

    _ACTION_TYPE_QUERY_HINTS = {}

    @classmethod
    def _query_action_types(cls, query: str) -> set[str]:
        clean = cls._normalize_match_text(query)
        types: set[str] = set()
        hints_by_type = cls._configured_type_query_hints(
            "knowledge.action_type_query_hints",
            cls._ACTION_TYPE_QUERY_HINTS,
        )
        for action_type, hints in hints_by_type.items():
            if any(cls._normalize_match_text(hint) in clean for hint in hints):
                types.add(action_type)
        if {"fulfillment_exception", "order_change"} & types:
            types.discard("product_attribute")
            types.discard("fault_handling")
        return types

    @staticmethod
    def _entry_action_types(entry) -> set[str]:
        tags = str(getattr(entry, "tags", "") or "").lower()
        found: set[str] = set()
        for match in re.finditer(r"action[_-]?type\s*[:=]\s*([a-z0-9_-]+)", tags):
            value = match.group(1).strip()
            if value:
                found.add(value)
        return found

    @classmethod
    def _action_type_match_score(cls, query: str, entry) -> int:
        query_types = cls._query_action_types(query)
        entry_types = cls._entry_action_types(entry)
        if not query_types or not entry_types:
            return 0
        if query_types & entry_types:
            return 150
        return -90

    _COMPLAINT_TYPE_QUERY_HINTS = {}

    @classmethod
    def _query_complaint_types(cls, query: str) -> set[str]:
        clean = cls._normalize_match_text(query)
        types: set[str] = set()
        hints_by_type = cls._configured_type_query_hints(
            "knowledge.complaint_type_query_hints",
            cls._COMPLAINT_TYPE_QUERY_HINTS,
        )
        for complaint_type, hints in hints_by_type.items():
            if any(cls._normalize_match_text(hint) in clean for hint in hints):
                types.add(complaint_type)
        return types

    @staticmethod
    def _entry_complaint_types(entry) -> set[str]:
        tags = str(getattr(entry, "tags", "") or "").lower()
        found: set[str] = set()
        for match in re.finditer(r"complaint[_-]?type\s*[:=]\s*([a-z0-9_-]+)", tags):
            value = match.group(1).strip()
            if value:
                found.add(value)
        return found

    @classmethod
    def _complaint_type_match_score(cls, query: str, entry) -> int:
        query_types = cls._query_complaint_types(query)
        entry_types = cls._entry_complaint_types(entry)
        if not query_types:
            return 0
        if entry_types & query_types:
            return 220
        if entry_types:
            return -100
        if cls._entry_parameter_types(entry):
            return -140
        return 0

    _CASE_TYPE_QUERY_HINTS = {}

    @classmethod
    def _query_case_types(cls, query: str) -> set[str]:
        clean = cls._normalize_match_text(query)
        types: set[str] = set()
        hints_by_type = cls._configured_type_query_hints(
            "knowledge.case_type_query_hints",
            cls._CASE_TYPE_QUERY_HINTS,
        )
        for case_type, hints in hints_by_type.items():
            if any(cls._normalize_match_text(hint) in clean for hint in hints):
                types.add(case_type)
        return types

    @staticmethod
    def _entry_case_types(entry) -> set[str]:
        tags = str(getattr(entry, "tags", "") or "").lower()
        found: set[str] = set()
        for match in re.finditer(r"case[_-]?type\s*[:=]\s*([a-z0-9_-]+)", tags):
            value = match.group(1).strip()
            if value:
                found.add(value)
        return found

    @classmethod
    def _case_type_match_score(cls, query: str, entry) -> int:
        query_types = cls._query_case_types(query)
        entry_types = cls._entry_case_types(entry)
        if not query_types:
            if "tutorial" in entry_types:
                return -120
            return 0
        if entry_types & query_types:
            return 180
        if "tutorial" in entry_types:
            return -180
        if entry_types:
            return -80
        if cls._entry_parameter_types(entry):
            return -120
        return 0

    @classmethod
    def _passes_retrieval_quality_gate(
        cls,
        query: str,
        entry,
        rule_score: int,
        vector_score: float,
        match_type: str,
    ) -> bool:
        """过滤仅靠向量捞上来、但和 query 的结构化类型明显冲突的候选。"""
        if match_type == "alias_exact":
            return True
        if rule_score > 0:
            return True
        if vector_score <= 0:
            return False

        checks = (
            (cls._query_case_types(query), cls._entry_case_types(entry)),
            (cls._query_action_types(query), cls._entry_action_types(entry)),
            (cls._query_parameter_types(query), cls._entry_parameter_types(entry)),
            (cls._query_complaint_types(query), cls._entry_complaint_types(entry)),
        )
        has_query_type = False
        for query_types, entry_types in checks:
            if not query_types:
                continue
            has_query_type = True
            if entry_types and not (query_types & entry_types):
                return False

        if has_query_type:
            return True

        query_terms = {cls._normalize_match_text(term) for term in cls._search_terms(query)}
        query_terms = {term for term in query_terms if len(term) >= 2}
        if not query_terms:
            return vector_score >= 0.65

        entry_text = cls._normalize_match_text(
            " ".join(
                str(part or "")
                for part in (
                    getattr(entry, "section_title", ""),
                    getattr(entry, "sub_intent", ""),
                    getattr(entry, "aliases", ""),
                    getattr(entry, "answer", ""),
                    getattr(entry, "tags", ""),
                )
            )
        )
        if any(term in entry_text for term in query_terms):
            return True

        return vector_score >= 0.72

    # ── 意图识别 + 评分调整 ──────────────────────────────────────

    _LOGISTICS_QUERY_KW = ("快递", "物流", "包裹", "几小时到", "什么时候到", "到哪了", "到了吗", "发了吗", "寄出了", "还有多久到")
    _BATTERY_COMPLAINT_KW = ()
    _WRONG_MISSING_KW = (
        "发错货", "发错颜色", "发错了", "错发", "颜色错", "颜色发错",
        "少了", "少了一个", "少了个", "少发", "少发了", "少发了一个", "漏发", "漏发了",
        "缺件", "缺少", "配件少",
    )
    _NOTE_CHANGE_KW = (
        "备注一下", "备注发", "帮我改一下", "改一下地址", "能改地址", "更改收货地址", "改颜色", "换颜色",
        "别发错", "不要发错", "别弄错", "不要弄错", "混色", "混发", "两个颜色", "发两个颜色",
        "一黑一白", "一白一黑", "一绿一蓝", "一蓝一绿",
    )
    _MIX_COLOR_WORDS = ("颜色", "色")
    _PRICE_QUERY_KW = ("价格", "多少钱", "几块", "九块", "9块", "9.9", "990元", "太贵", "优惠价", "售价")
    _WIND_QUERY_KW = ()
    _BATTERY_SIZE_QUERY_KW = ("容量多大", "容量是多少")
    _BATTERY_DURATION_QUERY_KW = ()
    _ARRIVAL_TIME_QUERY_KW = ("几天到货", "多久到货", "什么时候到", "多久能到", "几天能到", "到货", "送达")
    _NOISE_FAULT_QUERY_KW = ("噪音", "吵", "滋滋声", "异响", "声音大", "声音不正常")
    _COOLING_QUERY_KW = ()
    # ── 轻量级意图分类关键词 ──
    _INTENT_LOGISTICS_KW = (
        "发什么快递", "什么快递", "发哪家", "快递公司", "发货地", "哪里发货",
        "今天能发吗", "什么时候发货", "发中通吗", "发极兔吗", "发圆通吗",
        "发顺丰吗", "能指定快递吗", "几天到", "多久到", "什么时候到",
    )
    _INTENT_ACCESSORY_KW = ("配件", "送什么", "赠品", "带什么", "包装里有什么")
    _INTENT_COLOR_STOCK_KW = (
        "颜色", "有什么颜色", "哪个颜色", "库存", "有货吗", "什么颜色",
    )
    _INTENT_AFTERSALE_FAULT_KW = (
        "不转", "坏了", "异响", "滋滋声", "声音大", "还吵", "风小",
        "充不进电", "没电", "用不了", "开不了", "没反应", "打不开",
        "不出风", "噪音", "响", "松动",
    )
    _INTENT_WIND_POWER_KW = ()
    _INTENT_PRICE_KW = (
        "多少钱", "价格", "几块", "贵", "优惠", "券", "便宜",
        "打折", "活动价",
    )

    @classmethod
    def _intent_keywords(cls, name: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
        configured = get_config(f"knowledge.intent_keywords.{name}", None)
        if configured is None:
            configured = fallback
        elif not isinstance(configured, (list, tuple, set)):
            return fallback
        keywords = tuple(str(item or "").strip() for item in configured if str(item or "").strip())
        return keywords

    @classmethod
    def _classify_query_intent(cls, query: str) -> set:
        """轻量级 query 意图分类（纯规则，不调用 LLM）。"""
        q = str(query or "").strip()
        intents: set = set()
        if not q:
            return intents

        # battery_capacity 优先于 wind_power（"电池多大" 不是风力问题）
        if any(kw in q for kw in cls._intent_keywords("battery_size_query", cls._BATTERY_SIZE_QUERY_KW)):
            intents.add("battery_capacity")

        # battery_duration
        if any(kw in q for kw in cls._intent_keywords("battery_duration_query", cls._BATTERY_DURATION_QUERY_KW)):
            intents.add("battery_duration")

        # logistics_delivery
        if any(kw in q for kw in cls._intent_keywords("logistics_delivery", cls._INTENT_LOGISTICS_KW)):
            intents.add("logistics_delivery")

        # accessory
        if any(kw in q for kw in cls._intent_keywords("accessory", cls._INTENT_ACCESSORY_KW)):
            intents.add("accessory")

        # color_stock
        if any(kw in q for kw in cls._intent_keywords("color_stock", cls._INTENT_COLOR_STOCK_KW)):
            intents.add("color_stock")

        # wind_power（排除已归类为电池容量或续航的 query）
        if (
            "battery_capacity" not in intents
            and "battery_duration" not in intents
            and any(kw in q for kw in cls._intent_keywords("wind_power", cls._INTENT_WIND_POWER_KW))
        ):
            intents.add("wind_power")

        # price
        if any(kw in q for kw in cls._intent_keywords("price", cls._INTENT_PRICE_KW)):
            intents.add("price")

        # aftersale_fault
        if any(kw in q for kw in cls._intent_keywords("aftersale_fault", cls._INTENT_AFTERSALE_FAULT_KW)):
            intents.add("aftersale_fault")

        # noise_fault（噪音售后子类）
        if any(kw in q for kw in cls._intent_keywords("noise_fault", cls._NOISE_FAULT_QUERY_KW)):
            intents.add("noise_fault")

        return intents

    @classmethod
    def _query_intent_hints(cls, query: str) -> set:
        """识别 query 的意图标签集合，用于评分调整。"""
        q = str(query or "").strip()
        hints = set()
        # 物流/到货（排除"到了"这种出现在答案中的通用词）
        if any(kw in q for kw in cls._intent_keywords("logistics_query", cls._LOGISTICS_QUERY_KW)):
            hints.add("logistics")
        # 到货时效查询：几天到货/多久到/什么时候到/到货/送达
        if any(kw in q for kw in cls._intent_keywords("arrival_time", cls._ARRIVAL_TIME_QUERY_KW)):
            hints.add("arrival_time")
        # 售后续航投诉（区别于参数咨询）
        if any(kw in q for kw in cls._intent_keywords("battery_complaint", cls._BATTERY_COMPLAINT_KW)):
            hints.add("battery_complaint")
        # 错发/少件（不含"多少"）
        if any(kw in q for kw in cls._intent_keywords("wrong_missing", cls._WRONG_MISSING_KW)):
            hints.add("wrong_missing")
        # 备注/改地址/改颜色
        if any(kw in q for kw in cls._intent_keywords("note_change", cls._NOTE_CHANGE_KW)):
            hints.add("note_change")
        color_hits = sum(1 for kw in cls._intent_keywords("mix_color_words", cls._MIX_COLOR_WORDS) if kw in q)
        if color_hits >= 2 and any(kw in q for kw in ("一个", "一件", "1个", "1件", "各一", "别发错", "不要发错", "混色", "混发", "发一个", "发两个")):
            hints.add("note_change")
        if any(kw in q for kw in cls._intent_keywords("price_query", cls._PRICE_QUERY_KW)):
            hints.add("price_query")
        if any(kw in q for kw in cls._intent_keywords("wind_query", cls._WIND_QUERY_KW)):
            # Issue 7: "电池多大" 含 "多大" 但意图是电池，不应标记为风力查询
            if any(kw in q for kw in cls._intent_keywords("battery_size_query", cls._BATTERY_SIZE_QUERY_KW)):
                hints.add("battery_size_query")
            else:
                hints.add("wind_query")
        elif any(kw in q for kw in cls._intent_keywords("battery_size_query", cls._BATTERY_SIZE_QUERY_KW)):
            hints.add("battery_size_query")
        if any(kw in q for kw in cls._intent_keywords("battery_duration_query", cls._BATTERY_DURATION_QUERY_KW)):
            hints.add("battery_duration_query")
        # 版本/型号名 token：交给通用 version_name 机制处理。
        if cls._query_version_name_types(q):
            hints.add("version_name")

        # 合并轻量级意图分类结果
        classified = cls._classify_query_intent(q)
        if classified:
            hints |= classified
            logger.debug("[检索意图] query_chars={} intents={}".format(len(q), ",".join(sorted(classified))))

        # cooling_query（制冷/制冰/半导体意图）
        if any(kw in q for kw in cls._intent_keywords("cooling_query", cls._COOLING_QUERY_KW)):
            hints.add("cooling_query")

        return hints

    @classmethod
    def _intent_score_adjustment(cls, hints: set, entry, scene_key: str = "", query: str = "") -> int:
        """根据意图标签对条目进行加分/减分。返回调整值（可正可负）。"""
        section = (entry.section_title or "").lower()
        sub_intent = (entry.sub_intent or "").lower()
        answer = (entry.answer or "").lower()
        aliases = (entry.aliases or "").lower()
        combined = f"{section} {sub_intent} {answer} {aliases}"
        query_lower = str(query or "").lower()

        adj = 0

        adj += cls._intent_score_adjustment_from_rules(
            hints=hints,
            query=query,
            scene_key=scene_key,
            section=section,
            sub_intent=sub_intent,
            answer=answer,
            aliases=aliases,
            combined=combined,
        )

        # ── 版本名查询：非版本条目降权 ──
        version_tokens = cls._version_name_query_tokens()
        is_version_query = "version_name" in hints or any(kw in query_lower for kw in version_tokens)
        if is_version_query:
            is_version_entry = any(
                kw in combined
                for kw in ("版本名称", "版本区别", "型号名称", "版本", "规格名称", "型号", *version_tokens)
            )
            is_gear_or_button = any(kw in combined for kw in (
                "加减按键", "按键用途", "开关机教程", "正面按键",
            ))
            if is_gear_or_button and not is_version_entry:
                adj -= 1200
            if is_version_entry:
                adj += 400
            # 版本查询命中非续航/版本 section → 降权
            is_version_section = any(kw in section for kw in ("续航", "版本", "电池", "参数", "型号", "规格"))
            if not is_version_section:
                adj -= 400

        return adj

    @staticmethod
    def _version_name_query_tokens() -> tuple[str, ...]:
        configured = get_config("agent.version_name_tokens", None)
        if configured is None:
            configured = ()
        elif not isinstance(configured, (list, tuple, set)):
            configured = ()
        return tuple(str(token or "").strip().lower() for token in configured if str(token or "").strip())

    def format_scene_results(self, results: List[Dict[str, Any]]) -> str:
        """将 search_scene_knowledge 结果格式化为 Agent 可读字符串。"""
        if not results:
            return "未找到相关知识。"

        parts = []
        for i, item in enumerate(results, 1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("section_title") or "").strip()
            sub_intent = str(item.get("sub_intent") or "")
            answer = str(item.get("answer") or "")
            score = item.get("score", 0)
            match_type = str(item.get("match_type") or "")
            goods_id = item.get("goods_id")
            tags = str(item.get("tags") or "").strip()
            source_type = str(item.get("source_type") or "").strip()

            header = f"{i}. "
            if title:
                header += title
            elif sub_intent:
                header += sub_intent
            else:
                header += "命中知识"
            if goods_id:
                header += f" [商品{goods_id}]"
            header += f" (score={score}, {match_type})"

            parts.append(header)
            parts.append(f"  {answer}")
            metadata = []
            if tags:
                metadata.append(f"标签：{tags}")
            if source_type:
                metadata.append(f"来源：{source_type}")
            if match_type:
                metadata.append(f"匹配：{match_type}")
            for line in metadata:
                parts.append(f"  {line}")
            parts.append("")

        if not parts:
            return "未找到相关知识。"
        return "\n".join(parts).strip()
