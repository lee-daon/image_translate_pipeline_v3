import numpy as np
import onnxruntime
from typing import List

def inpaint_batch_gpu(
    session: onnxruntime.InferenceSession,
    image_np_list: List[np.ndarray],
    mask_np_list: List[np.ndarray],
) -> List[np.ndarray]:
    """
    ONNX 모델을 사용하여 이미지 배치를 받아 GPU에서 inpainting을 수행합니다.
    입력 이미지와 마스크는 모두 512x512 크기라고 가정합니다.

    Args:
        session: onnxruntime.InferenceSession 객체.
        image_np_list: 512x512 크기의 BGR 이미지(uint8) NumPy 배열 리스트.
        mask_np_list: 512x512 크기의 흑백 마스크(uint8) NumPy 배열 리스트.

    Returns:
        inpainting이 완료된 512x512 크기의 BGR 이미지(uint8) NumPy 배열 리스트.
    """
    # 모델의 메타데이터에서 입력/출력 이름 및 타입 가져오기
    input_details = session.get_inputs()
    image_input_name = input_details[0].name
    mask_input_name = input_details[1].name
    output_name = session.get_outputs()[0].name
    
    # 모델의 입력 타입에 따라 데이터 타입 결정 (float16 또는 float32)
    model_input_type = input_details[0].type
    if model_input_type == 'tensor(float16)':
        dtype = np.float16
    else:
        dtype = np.float32

    preprocessed_images = []
    preprocessed_masks = []

    for image_np, mask_np in zip(image_np_list, mask_np_list):
        # 전처리: 0-1 범위 정규화 및 CHW 변환
        img_preprocessed = np.transpose(image_np, (2, 0, 1)) / 255.0
        mask_preprocessed = np.expand_dims(mask_np, axis=0) / 255.0
        mask_preprocessed[mask_preprocessed > 0] = 1.0
        
        preprocessed_images.append(img_preprocessed)
        preprocessed_masks.append(mask_preprocessed)

    # 전처리된 이미지들을 하나의 배치로 합치고, 모델에 맞는 타입으로 변환
    img_batch = np.stack(preprocessed_images, axis=0).astype(dtype)
    mask_batch = np.stack(preprocessed_masks, axis=0).astype(dtype)
    
    # 모델의 입력 이름에 맞춰 데이터 제공
    input_feed = {
        image_input_name: img_batch,
        mask_input_name: mask_batch,
    }

    # 배치 추론 실행
    results = session.run([output_name], input_feed)[0]

    # 후처리: 결과 배치를 개별 이미지로 분리
    output_images = []
    for result_item in results:
        # 모델 출력은 이미 [0, 255] 범위로 clamp 되어 있으므로 타입만 변환
        output_image_np = np.transpose(result_item, (1, 2, 0))
        output_image_np = output_image_np.astype(np.uint8)
        output_images.append(output_image_np)
        
    return output_images
