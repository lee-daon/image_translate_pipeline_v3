import redis.asyncio as redis
import json
import logging
from typing import Dict, Any, Tuple, Optional

from core.config import REDIS_URL

logger = logging.getLogger(__name__)

_redis_client: redis.Redis = None

async def initialize_redis():
    """Redis 연결 풀을 초기화합니다."""
    global _redis_client
    try:
        logger.info(f"Connecting to Redis at {REDIS_URL}...")
        _redis_client = redis.from_url(REDIS_URL, decode_responses=False)
        await _redis_client.ping()
        logger.info("Successfully connected to Redis.")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
        raise

def get_redis_client() -> redis.Redis:
    """초기화된 Redis 클라이언트 인스턴스를 반환합니다."""
    if _redis_client is None:
        raise ConnectionError("Redis client is not initialized. Call initialize_redis() first.")
    return _redis_client

async def close_redis():
    """Redis 연결 풀을 종료합니다."""
    if _redis_client:
        logger.info("Closing Redis connection...")
        await _redis_client.close()
        logger.info("Redis connection closed.")

# 제거 또는 주석 처리: 기존 get_final_result
# async def get_final_result(request_id: str) -> Any | None: ...
