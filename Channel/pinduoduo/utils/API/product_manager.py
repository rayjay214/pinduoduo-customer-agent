from ..base_request import BaseRequest
import math

from utils.config_values import as_bool


def _error_log_summary(error_msg):
    return f"error_chars={len(str(error_msg or ''))}"


def _first_present(data, *keys, default=None):
    if not isinstance(data, dict):
        return default
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return default


def _first_list(data, *keys):
    for key in keys:
        value = data.get(key) if isinstance(data, dict) else None
        if isinstance(value, list):
            return value
    return []


def _coerce_cent_price(value):
    if value in (None, ""):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if "." in text:
            try:
                numeric = float(text)
                return numeric * 100 if math.isfinite(numeric) else None
            except ValueError:
                return None
    try:
        numeric = int(value)
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return numeric
    except (TypeError, ValueError, OverflowError):
        try:
            numeric = float(value)
            return numeric * 100 if math.isfinite(numeric) else None
        except (TypeError, ValueError):
            return None


def _format_cent_price(value):
    numeric = _coerce_cent_price(value)
    if numeric is None:
        return None
    return f"{numeric / 100:.2f}"


def _format_marketing_tags(goods_tag):
    if not isinstance(goods_tag, dict):
        return ""
    marketing_tags = goods_tag.get('marketingTags') or goods_tag.get('marketing_tags') or []
    if not isinstance(marketing_tags, list):
        return ""
    return ', '.join(str(tag) for tag in marketing_tags if str(tag).strip())


class ProductManager(BaseRequest):
    """
    拼多多商品管理API
    提供商品列表查询和商品详情获取功能
    """

    def __init__(self, shop_id: str = None, user_id: str = None, cookies=None):
        """
        初始化商品管理器

        Args:
            shop_id: 店铺ID，用于从数据库获取cookies
            user_id: 用户ID，用于从数据库获取cookies
            cookies: 登录cookies，如果直接传入则不需要从数据库获取
        """
        super().__init__(shop_id=shop_id, user_id=user_id)
        if cookies:
            if not self.update_cookies(cookies):
                self.logger.warning("初始化商品管理器时传入的 cookies 无效，已保留原 cookies")

    def get_product_list(self, page=1, size=10):
        """
        获取店铺商品列表

        Args:
            page (int): 页码，默认1
            size (int): 每页数量，默认10

        Returns:
            dict: 商品列表结果，格式如下：
                {
                    "success": True/False,
                    "products": [
                        {
                            "goods_id": int,
                            "goods_name": str,
                            "thumb_url": str,       # 商品缩略图
                            "price": float,         # 价格
                            "sold_quantity": int,   # 已售数量
                            "goods_type": int,      # 商品类型
                            "tag": str,             # 商品标签
                        },
                        ...
                    ],
                    "total": int,  # 总数量
                    "page": int,   # 当前页码
                    "error_msg": str  # 仅在失败时包含
                }
        """
        # 构建请求URL
        url = "https://mms.pinduoduo.com/latitude/goods/recommendGoods"

        # 构建请求数据
        data = {
            "uid": "",
            "pageNum": page,
            "pageSize": size
        }

        headers = self._build_mms_browser_headers(
            url=url,
            payload=data,
            require_anti_content=True,
        )

        # 发起请求
        result = self.post(url, json_data=data, headers=headers)

        if isinstance(result, dict) and as_bool(result.get("success"), False):
            if not isinstance(result.get("result"), dict):
                error_msg = f"商品列表响应 result 格式异常: {type(result.get('result')).__name__}"
                self.logger.error(error_msg)
                return {
                    "success": False,
                    "error_msg": error_msg,
                    "products": [],
                    "total": 0,
                    "page": page
                }
            # 解析商品列表
            products_data = self._parse_product_list(result)
            return {
                "success": True,
                "products": products_data.get("products", []),
                "total": products_data.get("total", 0),
                "page": page
            }
        else:
            error_msg = result.get('errorMsg') if isinstance(result, dict) else "获取商品列表失败"
            self.logger.error(f"获取商品列表失败: {_error_log_summary(error_msg)}")
            return {
                "success": False,
                "error_msg": error_msg,
                "products": [],
                "total": 0,
                "page": page
            }

    def get_product_detail(self, goods_id):
        """
        根据商品ID获取商品详细信息

        Args:
            goods_id (int): 商品ID

        Returns:
            dict: 商品详情结果，格式如下：
                {
                    "success": True/False,
                    "product_info": {
                        "goods_id": int,
                        "goods_name": str,
                        "specifications": str/dict,
                        "price": float,
                        "description": str,
                        # TODO: 根据实际API响应添加更多字段
                    },
                    "error_msg": str  # 仅在失败时包含
                }
        """
        if not goods_id:
            self.logger.error("商品ID不能为空")
            return {"success": False, "error_msg": "商品ID不能为空"}

        # 构建请求URL
        url = "https://mms.pinduoduo.com/glide/v2/mms/query/commit/on_shop/detail"

        # 构建请求数据
        data = {"goods_id": goods_id}

        headers = self._build_mms_browser_headers(
            url=url,
            payload=data,
            require_anti_content=True,
            include_client_hints=False,
        )

        # 发起请求
        result = self.post(url, json_data=data, headers=headers)

        if isinstance(result, dict) and as_bool(result.get("success"), False):
            if not isinstance(result.get("result"), dict):
                error_msg = f"商品详情响应 result 格式异常: {type(result.get('result')).__name__}"
                self.logger.error(error_msg)
                return {
                    "success": False,
                    "error_msg": error_msg
                }
            # 解析商品详细信息
            product_info = self._parse_product_detail(result)
            return {
                "success": True,
                "product_info": product_info
            }
        else:
            error_msg = result.get('errorMsg') if isinstance(result, dict) else "获取商品详情失败"
            self.logger.error(f"获取商品详情失败 (goods_id={goods_id}): {_error_log_summary(error_msg)}")
            return {
                "success": False,
                "error_msg": error_msg
            }

    def _parse_product_list(self, response_data):
        """
        解析商品列表响应数据

        Args:
            response_data (dict): API响应数据

        Returns:
            dict: 解析后的商品列表数据
        """
        try:
            result_data = response_data.get('result', {})
            goods_list = _first_list(
                result_data,
                "onSaleGoods",
                "goodsList",
                "goods_list",
                "list",
                "items",
            )

            products = []
            for goods in goods_list:
                if not isinstance(goods, dict):
                    continue
                # 价格：使用区间价格，最低价-最高价
                min_price = _first_present(goods, 'minOnSaleGroupPrice', 'min_on_sale_group_price', 'minPrice', 'min_price')
                max_price = _first_present(goods, 'maxOnSaleGroupPrice', 'max_on_sale_group_price', 'maxPrice', 'max_price')
                min_price_num = _coerce_cent_price(min_price)
                max_price_num = _coerce_cent_price(max_price)
                if min_price_num is not None and max_price_num is not None and min_price_num != max_price_num:
                    price_str = f"{min_price_num / 100:.2f}-{max_price_num / 100:.2f}"
                elif min_price_num is not None:
                    price_str = _format_cent_price(min_price_num)
                else:
                    price_str = None

                # 提取商品标签
                goods_tag = goods.get('goodsTag') or goods.get('goods_tag') or {}
                tag_str = _format_marketing_tags(goods_tag)

                product = {
                    "goods_id": _first_present(goods, 'goodsId', 'goods_id'),
                    "goods_name": _first_present(goods, 'goodsName', 'goods_name', default=''),
                    "thumb_url": _first_present(goods, 'thumbUrl', 'thumb_url', default=''),
                    "price": price_str,
                    "price_min": min_price,
                    "price_max": max_price,
                    "sold_quantity": _first_present(goods, 'soldQuantity', 'sold_quantity', default=0),
                    "sold_quantity_30d": _first_present(goods, 'soldQuantity30d', 'sold_quantity_30d', default=0),
                    "quantity": _first_present(goods, 'quantity', default=0),  # 库存
                    "goods_type": _first_present(goods, 'goodsType', 'goods_type', default=''),
                    "is_spike": _first_present(goods, 'isSpike', 'is_spike', default=False),  # 是否秒杀
                    "support_customize": _first_present(goods, 'supportCustomize', 'support_customize', default=False),  # 是否支持定制
                    "goods_url": _first_present(goods, 'goodsUrl', 'goods_url', default=''),  # 商品链接
                    "tag": tag_str,
                }
                products.append(product)

            return {
                "products": products,
                "total": result_data.get('total', len(products))
            }

        except Exception as e:
            self.logger.error(f"解析商品列表失败: {self._sanitize_exception_for_log(e)}")
            return {
                "products": [],
                "total": 0
            }

    def _parse_product_detail(self, response_data):
        """
        解析商品详情响应数据

        Args:
            response_data (dict): API响应数据

        Returns:
            dict: 解析后的商品详情
        """
        try:
            result_data = response_data.get('result', {})

            # 提取规格信息
            specifications = []
            skus = result_data.get('skus', [])

            if skus:
                for sku in skus:
                    if not isinstance(sku, dict):
                        continue
                    # 获取规格组合信息
                    specs = sku.get('spec') or sku.get('specs') or []
                    if specs:
                        spec_text = []
                        for spec_item in specs:
                            if not isinstance(spec_item, dict):
                                continue
                            parent_name = _first_present(spec_item, 'parent_name', 'parentName', default='')
                            spec_name = _first_present(spec_item, 'spec_name', 'specName', 'name', default='')
                            if parent_name and spec_name:
                                spec_text.append(f"{parent_name}: {spec_name}")
                            elif spec_name:
                                spec_text.append(spec_name)

                        if spec_text:
                            specifications.append(" | ".join(spec_text))

            # 提取分类信息作为规格补充
            cats = result_data.get('cats', [])
            if cats and isinstance(cats, list):
                # 过滤掉空值并组合分类信息
                valid_cats = [str(cat).strip() for cat in cats if str(cat or "").strip()]
                if valid_cats:
                    specifications.append(f"商品分类: {' > '.join(valid_cats)}")

            product_info = {
                "goods_id": _first_present(result_data, 'goods_id', 'goodsId'),
                "goods_name": _first_present(result_data, 'goods_name', 'goodsName', default=''),
                "specifications": specifications[:20]  # 最多显示20个规格信息
            }

            return product_info

        except Exception as e:
            self.logger.error(f"解析商品详情失败: {self._sanitize_exception_for_log(e)}")
            return {
                "goods_id": None,
                "goods_name": "解析失败",
                "specifications": []
            }
