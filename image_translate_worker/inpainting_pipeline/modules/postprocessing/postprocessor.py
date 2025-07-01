import cv2
import numpy as np
import onnxruntime
import logging
import math
from typing import Tuple

# 내부 모듈 임포트
from .resize import crop_padding
from .simple_upscaler import upscale_simple
from .upscaler import upscale_with_onnx

# AI 업스케일러의 배율을 상수로 정의
AI_UPSCALE_FACTOR = 2

def run_postprocessing(
    inpainted_image: np.ndarray,
    size_before_padding: Tuple[int, int],
    scale_factor: int,
    upscale_model_path: str,
) -> np.ndarray:
    """
    인페인팅된 이미지에 대한 전체 후처리 파이프라인을 실행합니다.
    AI 업스케일링을 위해 이미지를 64배수로 패딩하고, 처리 후 패딩을 제거합니다.
    """
    # 1. 패딩 제거하여 원본 비율의 이미지 복원
    restored_image = crop_padding(inpainted_image, size_before_padding)

    if scale_factor <= 1:
        return restored_image

    # 2. AI 업스케일링을 위해 64배수 크기로 패딩
    h, w = restored_image.shape[:2]
    target_h = math.ceil(h / 64) * 64
    target_w = math.ceil(w / 64) * 64
    pad_h = target_h - h
    pad_w = target_w - w

    if pad_h > 0 or pad_w > 0:
        logging.info(f"AI 업스케일링을 위해 패딩 추가: {w}x{h} -> {target_w}x{target_h}")
        image_to_upscale = cv2.copyMakeBorder(restored_image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    else:
        image_to_upscale = restored_image

    # 3. AI 업스케일링 수행
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        upscale_session = onnxruntime.InferenceSession(upscale_model_path, providers=providers)
        
        upscaled_padded_image = upscale_with_onnx(
            session=upscale_session,
            image_np=image_to_upscale,
        )
    except Exception as e:
        logging.error(f"AI 업스케일링 실패: {e}. 단순 리사이즈로 대체합니다.")
        return upscale_simple(restored_image, scale_factor)

    # 4. AI 업스케일링 후 패딩 제거
    if pad_h > 0 or pad_w > 0:
        h_upscaled, w_upscaled = h * AI_UPSCALE_FACTOR, w * AI_UPSCALE_FACTOR
        current_image = upscaled_padded_image[:h_upscaled, :w_upscaled, :]
        logging.info(f"AI 업스케일링 후 패딩 제거: {upscaled_padded_image.shape[1]}x{upscaled_padded_image.shape[0]} -> {current_image.shape[1]}x{current_image.shape[0]}")
    else:
        current_image = upscaled_padded_image

    # 5. 최종 크기를 원본 배율에 맞게 조절
    remaining_scale = scale_factor / AI_UPSCALE_FACTOR

    if remaining_scale > 1.0:
        logging.info(f"AI 업스케일링 후 추가 리사이즈 (배율: {remaining_scale:.2f})")
        final_image = upscale_simple(current_image, remaining_scale)
    else:
        final_image = current_image

    return final_image
