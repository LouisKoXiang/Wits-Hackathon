import os
import json
from typing import List, Dict, Any, Optional

import redis
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/debug/redis", tags=["Debug-Redis"])

# 這支 API 專門給「有啟用 Redis」的情境用
USE_REDIS = os.getenv("USE_REDIS", "0") == "1"

if not USE_REDIS:
    _redis_client = None
else:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))

    try:
        _redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=REDIS_DB,
            decode_responses=True,
        )
    except Exception as e:
        raise RuntimeError(f"初始化 Redis 客戶端失敗: {e}")


@router.get("/keys")
def list_keys(pattern: str = Query("*", description="Redis key pattern，例如 'chat_session:*'")) -> Dict[str, Any]:
    """
    列出 Redis 目前的 key。
    """
    if not USE_REDIS or _redis_client is None:
        raise HTTPException(status_code=400, detail="目前未啟用 Redis（USE_REDIS != 1）。")

    keys: List[str] = []
    try:
        # scan_iter 不會一次拉所有 key
        for k in _redis_client.scan_iter(match=pattern, count=100):
            keys.append(k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"讀取 Redis keys 失敗: {e}")

    return {
        "pattern": pattern,
        "count": len(keys),
        "keys": keys,
    }


@router.get("/key/{key}")
def get_key_detail(key: str) -> Dict[str, Any]:
    """
    讀取某個 Redis key 的內容。

    """
    if not USE_REDIS or _redis_client is None:
        raise HTTPException(status_code=400, detail="目前未啟用 Redis（USE_REDIS != 1）。")

    try:
        raw = _redis_client.get(key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"讀取 Redis key 失敗: {e}")

    if raw is None:
        raise HTTPException(status_code=404, detail=f"Redis 找不到 key: {key}")

    parsed: Optional[Any] = None
    is_json = False
    try:
        parsed = json.loads(raw)
        is_json = True
    except Exception:
        parsed = None

    return {
        "key": key,
        "raw_value": raw,
        "is_json": is_json,
        "parsed": parsed,
    }
