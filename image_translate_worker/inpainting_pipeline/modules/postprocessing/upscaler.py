import numpy as np
import onnxruntime
import logging

def upscale_with_onnx(
    session: onnxruntime.InferenceSession,
    image_np: np.ndarray,
) -> np.ndarray:
    """
    ONNX 모델을 사용하여 이미지 전체를 한 번에 업스케일링합니다.
    (메모리 부족에 주의)

    Args:
        session (onnxruntime.InferenceSession): 업스케일링 모델의 ONNX 세션.
        image_np (np.ndarray): 업스케일링할 BGR 이미지(uint8) NumPy 배열.

    Returns:
        np.ndarray: 업스케일링된 BGR 이미지(uint8) NumPy 배열.
    """
    logging.info(f"ONNX 모델로 전체 이미지({image_np.shape[1]}x{image_np.shape[0]}) 업스케일링 시작...")
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 모델 입력에 맞게 float16 타입으로 변환하고 [0, 1] 범위로 정규화
    img_preprocessed = np.transpose(image_np, (2, 0, 1)).astype(np.float16) / 255.0
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    # ONNX 런타임으로 추론 실행
    result = session.run([output_name], {input_name: img_batch})[0]

    # 모델 출력을 다시 [0, 255] 범위의 uint8 이미지로 변환
    output_image = (np.transpose(result[0], (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)

    logging.info(f"업스케일링 완료. 최종 크기: {output_image.shape[1]}x{output_image.shape[0]}")
    return output_image
