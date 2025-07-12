import asyncio
import logging
from typing import Optional
import io

import aiohttp
from PIL import Image

from .config import IMAGE_DOWNLOAD_MAX_RETRIES, IMAGE_DOWNLOAD_RETRY_DELAY

logger = logging.getLogger(__name__)

fetchHeaders = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
    'Referer': 'https://detail.tmall.com/',
}

async def download_image_async(
    session: aiohttp.ClientSession, url: str, request_id: str
) -> Optional[bytes]:
    """
    이미지를 비동기적으로 다운로드합니다 (재시도 로직 포함).
    Content-Type을 확인하여 jpg/png가 아니면 jpg로 변환합니다.

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
            async with session.get(url, headers=fetchHeaders) as response:
                response.raise_for_status()
                
                content_type = response.headers.get('Content-Type', '').lower()
                image_bytes = await response.read()

                if 'image/jpeg' in content_type or 'image/png' in content_type:
                    return image_bytes
                else:
                    logger.warning(f"[{request_id}] Unexpected Content-Type '{content_type}' for URL {url}. Attempting to convert to JPEG.")
                    try:
                        with Image.open(io.BytesIO(image_bytes)) as img:
                            if img.mode in ('RGBA', 'P'):
                                img = img.convert('RGB')
                            
                            with io.BytesIO() as buffer:
                                img.save(buffer, format='JPEG', quality=95)
                                return buffer.getvalue()
                    except Exception as convert_exc:
                        logger.error(f"[{request_id}] 이미지 JPEG 변환 실패: {url}", exc_info=convert_exc)
                        return None # 변환 실패 시 None 반환

        except Exception as e:
            if attempt == IMAGE_DOWNLOAD_MAX_RETRIES - 1:
                logger.error(f"[{request_id}] URL에서 이미지 다운로드 최종 실패: {url}", exc_info=True)
                return None
            logger.warning(f"[{request_id}] 이미지 다운로드 재시도 ({attempt + 1}/{IMAGE_DOWNLOAD_MAX_RETRIES})... 에러: {e}")
            await asyncio.sleep(IMAGE_DOWNLOAD_RETRY_DELAY)
            
    return None 