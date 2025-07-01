import numpy as np
from typing import Tuple

def crop_padding(
    image_np: np.ndarray, 
    original_size_before_padding: Tuple[int, int]
) -> np.ndarray:
    """
    전처리 시 추가된 패딩을 제거하여 원본 콘텐츠 영역만 남깁니다.

    Args:
        image_np (np.ndarray): 패딩이 추가된 이미지 (예: 512x512).
        original_size_before_padding (Tuple[int, int]): 패딩 전 이미지 크기 (너비, 높이).

    Returns:
        np.ndarray: 패딩이 제거된 이미지.
    """
    padded_h, padded_w = image_np.shape[:2]
    original_w, original_h = original_size_before_padding

    top = (padded_h - original_h) // 2
    left = (padded_w - original_w) // 2
    
    bottom = top + original_h
    right = left + original_w

    return image_np[top:bottom, left:right]
