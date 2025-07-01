import logging
from datetime import datetime
import numpy as np
import cv2

from core.config import RESIZE_TARGET_SIZE
from hosting.r2hosting import R2ImageHosting

logger = logging.getLogger(__name__)

def handle_no_chinese_text_sync(
    image_bytes: bytes, 
    original_url: str, 
    request_id: str, 
    image_id: str, 
    is_long: bool,
    r2_hosting_instance: R2ImageHosting
) -> str:
    """
    중국어 텍스트가 없을 때 이미지를 리사이즈하고 R2에 업로드하는 동기 함수.
    
    Args:
        image_bytes (bytes): 원본 이미지의 바이트 데이터.
        original_url (str): 원본 이미지 URL.
        request_id (str): 요청 ID.
        image_id (str): 이미지 ID.
        is_long (bool): 긴 이미지 여부.
        r2_hosting_instance (R2ImageHosting): R2 호스팅을 위한 공유 인스턴스.
        
    Returns:
        str: 업로드된 이미지의 URL. 실패 시 원본 URL.
    """
    try:
        img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: 
            logger.warning(f"[{request_id}] Failed to decode image for resizing.")
            return original_url
        
        h, w = img.shape[:2]
        target_h, target_w = (int(h * (864 / w)), 864) if is_long else RESIZE_TARGET_SIZE
        resized = cv2.resize(img, (target_w, target_h))

        date_str = datetime.now().strftime('%Y-%m-%d')
        product_id, rem = (image_id.split('-', 1) + [""])[:2]
        final_id = f"{rem}-{request_id[:5]}" if rem else f"{image_id}-{request_id[:5]}"
        
        res = r2_hosting_instance.upload_image_from_array(
            resized, final_id, f'translated_image/{date_str}/{product_id}', '.jpg', 90,
            {"request_id": request_id, "image_id": image_id, "type": "resized_no_text"}
        )
        return res["url"] if res.get("success") else original_url
    except Exception as e:
        logger.error(f"[{request_id}] Error in no-Chinese-text handler: {e}", exc_info=True)
        return original_url 