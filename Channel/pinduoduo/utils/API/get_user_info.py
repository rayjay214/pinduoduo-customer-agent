from ..base_request import BaseRequest
from utils.config_values import as_bool


def _error_log_summary(error_msg):
    return f"error_chars={len(str(error_msg or ''))}"


class GetUserInfo(BaseRequest):
    def __init__(self, cookies=None):

        super().__init__()
        if cookies:
            if not self.update_cookies(cookies):
                self.logger.warning("初始化用户信息接口时传入的 cookies 无效，已保留原 cookies")
    def get_user_info(self):
        url = "https://mms.pinduoduo.com/janus/api/new/userinfo"
        
        result = self.post(url, data="")

        if not isinstance(result, dict):
            self.logger.error(f"获取用户信息失败: 响应格式异常: {type(result).__name__}")
            return False
        
        if as_bool(result.get("success"), False):
            result_data = result.get('result', {})
            if not isinstance(result_data, dict):
                self.logger.error(f"获取用户信息失败: 响应 result 格式异常: {type(result_data).__name__}")
                return False
            user_id = result_data.get('id')
            user_name = result_data.get('username')
            mall_id = result_data.get('mall_id')
            return user_id, user_name, mall_id
        else:
            error_msg = result.get('errorMsg') if result else "获取用户信息失败"
            self.logger.error(f"获取用户信息失败: {_error_log_summary(error_msg)}")
            return False
