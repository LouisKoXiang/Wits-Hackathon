import os
import json

class RedisMemory:
    """
    聊天記憶管理。
    根據環境變數決定使用 Redis 還是本機 memory
    - USE_REDIS=0 → 使用程式內的 dict
    - USE_REDIS=1 → 使用 Redis
    """

    def __init__(self):
        # 判斷是否啟用 Redis
        self.use_redis = os.getenv("USE_REDIS", "0") == "1"

        if self.use_redis:
            # 連接 Redis，連不到的話會在 API 層回報錯誤
            import redis
            self.client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                password=os.getenv("REDIS_PASSWORD", None),
                username="default",
                decode_responses=True,   # 自動把 bytes 轉成字串
            )
        else:
            # 本機暫存（重啟就會消失）
            self.memory_store = {}

    # 取出某個 session 的記憶
    def get_session(self, session_id: str):
        if self.use_redis:
            raw = self.client.get(f"chat_session:{session_id}")

            # 沒資料就給預設格式
            if not raw:
                return {
                    "messages": [],
                    "last_prediction": None,
                    "last_form_data": None
                }

            try:
                data = json.loads(raw)
            except Exception:
                return {
                    "messages": [],
                    "last_prediction": None,
                    "last_form_data": None
                }

            # 確保 key 都還在
            data.setdefault("messages", [])
            data.setdefault("last_prediction", None)
            data.setdefault("last_form_data", None)
            return data

        # in-memory 模式
        data = self.memory_store.get(session_id)

        if not data:
            return {
                "messages": [],
                "last_prediction": None,
                "last_form_data": None
            }

        data.setdefault("messages", [])
        data.setdefault("last_prediction", None)
        data.setdefault("last_form_data", None)
        return data

    # 寫入某個 session Data
    def save_session(self, session_id: str, data):
        if self.use_redis:
            self.client.set(
                f"chat_session:{session_id}",
                json.dumps(data, ensure_ascii=False),
                ex=60 * 60 * 12  # 12 小時
            )
        else:
            self.memory_store[session_id] = data

    # 清除某個 session
    def clear(self, session_id: str):
        if self.use_redis:
            self.client.delete(f"chat_session:{session_id}")
        else:
            self.memory_store.pop(session_id, None)
