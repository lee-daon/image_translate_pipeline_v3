# 파일명: inpainting_pipeline/modules/postprocessing/simple_upscaler.py
import cv2
import numpy as np
from typing import Union

def upscale_simple(image_np: np.ndarray, scale_factor: Union[int, float]) -> np.ndarray:
    """
    단순 보간법(Cubic)을 사용하여 이미지를 확대합니다.

    Args:
        image_np (np.ndarray): 확대할 BGR 이미지.
        scale_factor (Union[int, float]): 확대 배율 (1.0보다 커야 함).

    Returns:
        np.ndarray: 확대된 이미지.
    """
    if scale_factor <= 1.0:
        return image_np
        
    h, w = image_np.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    return cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)