from ..base_request import BaseRequest
from utils.config_values import as_bool


def _error_log_summary(error_msg):
    return f"error_chars={len(str(error_msg or ''))}"


class AccountMonitor(BaseRequest):
    def __init__(self, cookies=None, shop_id=None, user_id=None, channel_name="pinduoduo"):
        # 如果直接传入cookies，同时支持传入账户信息用于自动重新登录
        super().__init__(shop_id=shop_id, user_id=user_id, channel_name=channel_name)
        if cookies:
            if not self.update_cookies(cookies):
                self.logger.warning("初始化在线状态接口时传入的 cookies 无效，已保留原 cookies")
    def set_csstatus(self, status: str):
        url = 'https://mms.pinduoduo.com/plateau/chat/set_csstatus'
        
        data = {
            "data": {
                "cmd": "set_csstatus",
                "status": status
            },
            "client": "WEB"
        }
        
        # 使用基类的post方法
        result = self.post(url, json_data=data)

        if not isinstance(result, dict):
            self.logger.error(f"账号 设置状态失败: 响应格式异常: {type(result).__name__}")
            return False
        
        if result and as_bool(result.get("success"), False):
            return True
        else:
            error_msg = result.get('errorMsg') if result else "设置状态失败"
            self.logger.error(f"账号 设置状态失败: {_error_log_summary(error_msg)}")
            return False
            

   



