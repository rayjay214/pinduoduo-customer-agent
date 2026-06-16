from ..base_request import BaseRequest


class GetToken(BaseRequest):
    def __init__(self, shop_id, user_id, channel_name="pinduoduo"):
        super().__init__(shop_id, user_id, channel_name)

    def get_token(self):
        """
        根据提供的店铺名获取对应的token
        Returns:
            str: 成功返回token字符串
            None: 获取失败返回None
        """
        url = "https://mms.pinduoduo.com/chats/getToken"
        payload = {'version': '3'}

        result = self.post(url, data=payload)
        
        if result:
            if not isinstance(result, dict):
                self.logger.error(
                    f"账号 {self.account_name} 无法从响应中获取token: {self._sanitize_text_for_log(result)}"
                )
                return None

            # 处理响应
            if result.get('token'):
                return result['token']
            nested_result = result.get('result')
            if isinstance(nested_result, dict) and nested_result.get('token'):
                return nested_result.get('token')
            else:
                self.logger.error(
                    f"账号 {self.account_name} 无法从响应中获取token: {self._sanitize_for_log(result)}"
                )
        
        return None



