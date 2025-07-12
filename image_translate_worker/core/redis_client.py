import redis.asyncio as redis
import json
import logging
from typing import Optional, Callable, Awaitable

from core.config import REDIS_URL, SUCCESS_QUEUE, ERROR_QUEUE

logger = logging.getLogger(__name__)

_redis_client: redis.Redis = None
_task_completion_callback: Optional[Callable[[], Awaitable[None]]] = None

def set_task_completion_callback(callback: Callable[[], Awaitable[None]]):
    """작업 완료 시 호출할 콜백 함수를 설정합니다"""
    global _task_completion_callback
    _task_completion_callback = callback

async def _notify_task_completion():
    """작업 완료를 알립니다"""
    if _task_completion_callback:
        try:
            await _task_completion_callback()
        except Exception as e:
            logger.error(f"Failed to notify task completion: {e}")

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

async def enqueue_error_result(request_id: str, image_id: str, error_message: str):
    """에러 결과를 에러 큐에 추가합니다."""
    try:
        redis_client = get_redis_client()
        error_data = {
            "image_id": image_id,
            "error_message": error_message
        }
        error_json = json.dumps(error_data).encode('utf-8')
        await redis_client.rpush(ERROR_QUEUE, error_json)
        logger.info(f"[{request_id}] Error result enqueued to {ERROR_QUEUE}: {error_message}")
        
        # 작업 완료 알림
        await _notify_task_completion()
    except Exception as e:
        logger.error(f"[{request_id}] Failed to enqueue error result: {e}", exc_info=True)

async def enqueue_success_result(request_id: str, image_id: str, image_url: str, queue_name: str = SUCCESS_QUEUE):
    """성공 결과를 지정된 큐에 추가합니다."""
    try:
        redis_client = get_redis_client()
        success_data = {
            "image_id": image_id,
            "image_url": image_url
        }
        success_json = json.dumps(success_data).encode('utf-8')
        await redis_client.rpush(queue_name, success_json)
        logger.info(f"[{request_id}] Success result enqueued to {queue_name}: {image_url}")
        
        # 작업 완료 알림
        await _notify_task_completion()
    except Exception as e:
        logger.error(f"[{request_id}] Failed to enqueue success result: {e}", exc_info=True)


