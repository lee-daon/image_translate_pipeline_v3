import asyncio
import logging
from typing import Optional

import aiohttp

from .config import IMAGE_DOWNLOAD_MAX_RETRIES, IMAGE_DOWNLOAD_RETRY_DELAY

logger = logging.getLogger(__name__)

async def download_image_async(
    session: aiohttp.ClientSession, url: str, request_id: str
) -> Optional[bytes]:
    """
    이미지를 비동기적으로 다운로드합니다 (재시도 로직 포함).

    Args:
        session (aiohttp.ClientSession): aiohttp 클라이언트 세션.
        url (str): 다운로드할 이미지의 URL.
        request_id (str): 로깅을 위한 요청 ID.

    Returns:
        Optional[bytes]: 성공 시 이미지의 바이트 데이터, 실패 시 None.
    """
    if url.startswith('//'):
        url = 'https:' + url
        
    for attempt in range(IMAGE_DOWNLOAD_MAX_RETRIES):
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.read()
        except Exception as e:
            if attempt == IMAGE_DOWNLOAD_MAX_RETRIES - 1:
                logger.error(f"[{request_id}] URL에서 이미지 다운로드 최종 실패: {url}", exc_info=True)
                return None
            logger.warning(f"[{request_id}] 이미지 다운로드 재시도 ({attempt + 1}/{IMAGE_DOWNLOAD_MAX_RETRIES})... 에러: {e}")
            await asyncio.sleep(IMAGE_DOWNLOAD_RETRY_DELAY)
    return None 