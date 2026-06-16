from ..base_request import BaseRequest
from utils.config_values import as_bool


def _error_log_summary(error_msg):
    return f"error_chars={len(str(error_msg or ''))}"


class GetShopInfo(BaseRequest):
    def __init__(self, cookies=None):
        # 如果直接传入cookies，不需要从数据库获取
        super().__init__()
        if cookies:
            if not self.update_cookies(cookies):
                self.logger.warning("初始化店铺信息接口时传入的 cookies 无效，已保留原 cookies")
    
    def get_shop_info(self):
        url = "https://mms.pinduoduo.com/earth/api/merchant/queryMerchantInfoByMallId"

        result = self.post(url, json_data={})

        if not isinstance(result, dict):
            self.logger.error(f"获取店铺信息失败: 响应格式异常: {type(result).__name__}")
            return False
        
        if as_bool(result.get("success"), False):
            result_data = result.get('result', {})
            if not isinstance(result_data, dict):
                self.logger.error(f"获取店铺信息失败: 响应 result 格式异常: {type(result_data).__name__}")
                return False
            shop_id = result_data.get('mallId')
            shop_name = result_data.get('mallName')
            mallLogo = result_data.get('mallLogo')
            return shop_id, shop_name, mallLogo
        else:
            error_msg = result.get('errorMsg') if result else "获取店铺信息失败"
            self.logger.error(f"获取店铺信息失败: {_error_log_summary(error_msg)}")
            return False
