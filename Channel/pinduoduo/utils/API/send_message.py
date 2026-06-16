from ..base_request import BaseRequest
from config import get_config
from utils.config_values import as_bool


DEFAULT_TRANSFER_REMARK = "客服升级处理"


def _result_dict(result):
    payload = (result or {}).get("result") if isinstance(result, dict) else None
    return payload if isinstance(payload, dict) else {}


def _result_log_summary(result):
    if not isinstance(result, dict):
        return f"result_type={type(result).__name__}"
    payload = _result_dict(result)
    success = result.get("success")
    return (
        f"result_type=dict success={success!r} "
        f"top_keys={len(result)} payload_type={type(payload).__name__} result_keys={len(payload)}"
    )


def _error_log_summary(error_msg):
    return f"error_chars={len(str(error_msg or ''))}"


class SendMessage(BaseRequest):
    def __init__(self, shop_id: str, user_id: str, channel_name: str = "pinduoduo"):
        super().__init__(shop_id, user_id, channel_name)
        
        # 检查账户信息是否正确加载
        if not hasattr(self, 'account_name'):
            self.logger.error(f"无法在数据库中找到账户: shop_id={shop_id}, user_id={user_id}")
            raise ValueError("找不到指定的账户信息")

    def send_text(self, recipient_uid, message_content):
        """
        发送文本消息
        """
        url = "https://mms.pinduoduo.com/plateau/chat/send_message"
        data = {
            "data": {
                "cmd": "send_message",
                "request_id": self.generate_request_id(),
                "message": {
                    "to": {
                        "role": "user",
                        "uid": recipient_uid
                    },
                    "from": {
                        "role": "mall_cs"
                    },
                    "content": message_content,
                    "msg_id": None,
                    "type": 0,
                    "is_aut": 0,
                    "manual_reply": 1,
                },
            },
            "client": "WEB"
        }

        result = self.post(url, json_data=data)
        if not isinstance(result, dict):
            self.logger.error(f"发送文本消息失败: {_result_log_summary(result)}")
            return {"success": False, "error_msg": "请求失败", "result": result}

        if as_bool(result.get("success"), False):
            result_payload = _result_dict(result)
            if result_payload.get("error_code") == 10002:
                error_msg = result_payload.get('error')
                self.logger.error(f"发送文本消息失败: {_error_log_summary(error_msg)}")
                return {"success": False, "error_msg": error_msg or "发送失败", "result": result_payload}
            else:
                return result
        error_msg = result.get("error_msg") or result.get("errorMsg") or _result_dict(result).get("error") or "请求失败"
        self.logger.error(f"发送文本消息失败: {_result_log_summary(result)}")
        return {"success": False, "error_msg": error_msg, "result": result}

 
        
    def send_image(self, recipient_uid, image_url):
        """
        发送图片消息
        """
        url = "https://mms.pinduoduo.com/plateau/chat/send_message"
        data = {
            "data": {
                "cmd": "send_message",
                "request_id": self.generate_request_id(),
                "message": {
                    "to": {
                        "role": "user",
                        "uid": recipient_uid
                    },
                    "from": {
                        "role": "mall_cs"
                    },
                    "content": image_url,
                    "msg_id": None,
                    "chat_type": "cs",
                    "type": 1,
                    "is_aut": 0,
                    "manual_reply": 1,
                }
            },
            "client": "WEB"
        }

        result = self.post(url, json_data=data)
        if isinstance(result, dict):
            if as_bool(result.get("success"), False):
                self.logger.debug(f"发送图片消息成功: {_result_log_summary(result)}")
                return result
            error_msg = result.get("error_msg") or result.get("errorMsg") or _result_dict(result).get("error") or "请求失败"
            self.logger.error(f"发送图片消息失败: {_result_log_summary(result)}")
            return {"success": False, "error_msg": error_msg, "result": result}
        return {"success": False, "error_msg": "请求失败", "result": result}


    def send_mallGoodsCard(self, recipient_uid, goods_id, biz_type: int = 2):
        """
        发送商城商品卡片消息

        Args:
            recipient_uid: 接收消息的用户UID
            goods_id: 商品ID
            biz_type: 业务类型，默认2（客服推荐商品）
        """
        url = "https://mms.pinduoduo.com/plateau/message/send/mallGoodsCard"
        data = {
            "uid": recipient_uid,
            "goods_id": goods_id,
            "biz_type": biz_type
        }

        headers = self._build_mms_browser_headers(
            url=url,
            payload=data,
            require_anti_content=True,
        )

        result = self.post(url, json_data=data, headers=headers)
        if isinstance(result, dict):
            if as_bool(result.get("success"), False):
                self.logger.info(f"商品卡片发送成功: goods_id={goods_id}, to={recipient_uid}, biz_type={biz_type}")
                return result
            else:
                self.logger.error(
                    f"商品卡片发送失败: {_error_log_summary(result.get('error_msg', '未知错误'))}"
                )
                return {
                    "success": False,
                    "error_msg": result.get("error_msg", "发送失败"),
                    "result": result,
                }
        return {"success": False, "error_msg": "请求失败", "result": result}


    def getAssignCsList(self):
        """
        获取分配的客服列表
        """
        url = "https://mms.pinduoduo.com/latitude/assign/getAssignCsList"
        data = {"wechatCheck": True}
        
        result = self.post(url, json_data=data)
        if not isinstance(result, dict):
            self.logger.error(f"获取分配的客服列表失败: 响应格式异常: {type(result).__name__}")
            return None

        if as_bool(result.get('success'), False):
            cs_list = _result_dict(result).get('csList')
            if isinstance(cs_list, dict):
                return cs_list
            self.logger.error(
                f"获取分配的客服列表失败: 响应缺少有效 csList, {_result_log_summary(result)}"
            )
            return None

        error_msg = _result_dict(result).get('error') if result else "请求失败"
        self.logger.error(f"获取分配的客服列表失败: {_error_log_summary(error_msg)}")
        return None


    def move_conversation(self, recipient_uid, cs_uid, remark=None):
        """
        转移会话
        """
        url = "https://mms.pinduoduo.com/plateau/chat/move_conversation"
        transfer_remark = str(
            remark
            if remark is not None
            else get_config("pinduoduo.transfer.default_remark", DEFAULT_TRANSFER_REMARK)
        ).strip() or DEFAULT_TRANSFER_REMARK
        data = {
            "data": {
                "cmd": "move_conversation",
                "request_id": self.generate_request_id(),
                "conversation": {
                    "csid": cs_uid,
                    "uid": recipient_uid,
                    "need_wx": False,
                    "remark": transfer_remark
                }
            },
            "client": "WEB"
        }
        
        result = self.post(url, json_data=data)
        if not isinstance(result, dict):
            return {"success": False, "error_msg": "请求失败", "result": result}

        if as_bool(result.get("success"), False):
            self.logger.debug(f"转移会话成功: {_result_log_summary(result)}")
            return result

        error_msg = result.get("error_msg") or result.get("errorMsg") or _result_dict(result).get("error") or "请求失败"
        self.logger.error(f"转移会话失败: {_result_log_summary(result)}")
        return {"success": False, "error_msg": error_msg, "result": result}
