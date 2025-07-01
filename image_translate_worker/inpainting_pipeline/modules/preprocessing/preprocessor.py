import cv2
import numpy as np
from typing import Tuple

def preprocess_image(
    image_np: np.ndarray, target_size: int = 512
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], int]:
    """
    이미지를 조건에 따라 축소하고, 목표 크기에 맞게 패딩합니다.

    Args:
        image_np (np.ndarray): BGR 이미지 NumPy 배열.
        target_size (int): 모델 입력 목표 크기 (기본값: 512).

    Returns:
        Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], int]: 
        (처리된 이미지, 패딩 전 이미지 크기(w, h), 원본 크기(w, h), 축소 배율) 튜플.
    """
    # CPU 디노이징 적용 (Bilateral Filter)
    image_np = cv2.bilateralFilter(
        src=image_np, 
        d=9,             # Pixel neighborhood diameter
        sigmaColor=50,   # Filter sigma in the color space
        sigmaSpace=50    # Filter sigma in the coordinate space
    )

    h, w = image_np.shape[:2]
    original_size = (w, h)
    max_dim = max(h, w)

    # `max_dim`을 `target_size`로 나누어 올림 계산하여 동적으로 축소 배율 결정
    # 예: max_dim=800, target_size=512 -> (800+511)//512 = 2배 축소
    # 예: max_dim=1200, target_size=512 -> (1200+511)//512 = 3배 축소
    # 예: max_dim=2200, target_size=512 -> (2200+511)//512 = 5배 축소
    if max_dim > target_size:
        # 정수 올림 나눗셈
        scale_factor = (max_dim + target_size - 1) // target_size
    else:
        scale_factor = 1

    if scale_factor > 1:
        new_w, new_h = int(w / scale_factor), int(h / scale_factor)
        resized_img = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized_img = image_np

    # 목표 크기에 맞게 검은색으로 패딩
    rh, rw = resized_img.shape[:2]
    size_before_padding = (rw, rh) # (너비, 높이) 순서

    top = (target_size - rh) // 2
    bottom = target_size - rh - top
    left = (target_size - rw) // 2
    right = target_size - rw - left
    
    padded_img = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    return padded_img, size_before_padding, original_size, scale_factor
