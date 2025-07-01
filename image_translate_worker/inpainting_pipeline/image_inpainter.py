import onnxruntime
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List, Iterator
import logging
import os
import cv2
import numpy as np
from concurrent.futures import as_completed

# 파이프라인 전반에 사용할 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_models_on_gpu(
    inpaint_model_path: str
) -> Optional[onnxruntime.InferenceSession]:
    """
    인페인팅 ONNX 모델을 GPU에 로드합니다.

    Args:
        inpaint_model_path (str): 인페인팅 ONNX 모델 파일 경로.

    Returns:
        onnxruntime.InferenceSession: 성공 시 인페인팅 세션을, 실패 시 None을 반환합니다.
    """
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
        logging.warning("CUDAExecutionProvider를 사용할 수 없습니다. CPU로 대체합니다.")
        providers = ['CPUExecutionProvider']
    else:
        logging.info("CUDAExecutionProvider를 사용하여 GPU에서 모델을 실행합니다.")

    inpaint_session = None

    try:
        if not os.path.exists(inpaint_model_path):
            logging.error(f"인페인팅 모델을 찾을 수 없습니다: {inpaint_model_path}")
            return None
        logging.info(f"인페인팅 모델 로딩: {inpaint_model_path}")
        inpaint_session = onnxruntime.InferenceSession(inpaint_model_path, providers=providers)
        logging.info("인페인팅 모델을 성공적으로 로드했습니다.")
    except Exception as e:
        logging.error(f"인페인팅 모델 로딩 실패: {e}")
        return None
    
    return inpaint_session

def setup_thread_pool(max_workers: int = 4) -> ThreadPoolExecutor:
    """
    동시 처리를 위한 스레드 풀을 설정합니다.

    Args:
        max_workers (int): 스레드 풀의 최대 스레드 수.

    Returns:
        ThreadPoolExecutor: 생성된 스레드 풀 실행자 객체.
    """
    logging.info(f"최대 {max_workers}개의 스레드로 스레드 풀을 설정합니다.")
    return ThreadPoolExecutor(max_workers=max_workers)

# --- 파이프라인 모듈 임포트 ---
from .modules.preprocessing.preprocessor import preprocess_image
from .modules.inpaint_gpu.batch_inpainting import inpaint_batch_gpu
from .modules.postprocessing.postprocessor import run_postprocessing

# --- 모델 경로 상수 ---
# 이 파일의 위치를 기준으로 패키지 루트 디렉토리를 동적으로 찾습니다.
# 이렇게 하면 패키지를 다른 곳으로 옮겨도 경로가 깨지지 않습니다.
PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPAINT_MODEL = os.path.join(PIPELINE_ROOT, "modules/inpaint_gpu/models/lama_512_fp32.onnx")
DEFAULT_UPSCALE_MODEL = os.path.join(PIPELINE_ROOT, "modules/postprocessing/models/2x_ModernSpanimationV1_fp16_op17.onnx")

class ImageInpainter:
    """
    이미지 인페인팅 및 후처리 파이프라인을 관리하는 메인 클래스.
    """
    def __init__(self, executor: Optional[ThreadPoolExecutor] = None, max_workers: int = 4):
        """
        ImageInpainter 초기화. 모델 로드 및 스레드 풀을 설정합니다.
        외부 스레드 풀 실행자를 받아 공유할 수 있습니다.
        
        Args:
            executor (Optional[ThreadPoolExecutor]): 공유할 스레드 풀 실행자.
            max_workers (int): `executor`가 제공되지 않을 경우 생성할 스레드 풀의 최대 스레드 수.
        """
        self.inpaint_session = load_models_on_gpu(DEFAULT_INPAINT_MODEL)
        if not self.inpaint_session:
            raise ValueError("인페인팅 모델 로딩에 실패했습니다. 파이프라인을 시작할 수 없습니다.")
        
        if executor:
            self.executor = executor
            self._created_executor = False
            logging.info("외부 스레드 풀을 공유하여 사용합니다.")
        else:
            self.executor = setup_thread_pool(max_workers)
            self._created_executor = True

    def process_images(
        self, 
        image_list: List[np.ndarray], 
        mask_list: List[np.ndarray], 
        batch_size: int = 8
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        이미지 목록을 받아 전체 인페인팅 파이프라인을 실행하고,
        (원본 인덱스, 처리 완료된 이미지) 튜플을 하나씩 반환(yield)합니다.

        Args:
            image_list (List[np.ndarray]): 처리할 BGR 이미지 NumPy 배열 리스트.
            mask_list (List[np.ndarray]): 사용할 마스크 NumPy 배열 리스트.
            batch_size (int): GPU 인페인팅 시 한 번에 처리할 배치 크기.

        Yields:
            Iterator[Tuple[int, np.ndarray]]: (원본 인덱스, 최종 처리된 이미지) 튜플.
        """
        
        num_images = len(image_list)
        if num_images == 0:
            return

        # 1. 전처리 (병렬 실행)
        logging.info(f"{num_images}개의 이미지에 대한 병렬 전처리를 시작합니다...")
        
        # as_completed는 순서를 보장하지 않으므로, future와 원본 인덱스를 매핑
        preprocess_futures = {
            self.executor.submit(preprocess_image, image_list[i]): i 
            for i in range(num_images)
        }
        
        # 결과를 담을 리스트를 미리 생성하여 순서를 보장
        preprocessed_results = [None] * num_images
        for future in as_completed(preprocess_futures):
            original_index = preprocess_futures[future]
            try:
                preprocessed_results[original_index] = future.result()
            except Exception as e:
                logging.error(f"이미지 {original_index} 전처리 중 오류 발생: {e}")
                preprocessed_results[original_index] = (None, None, None, None) # 실패 시 플레이스홀더
        
        # 전처리 결과 분리
        preprocessed_images, sizes_before_padding, _, scale_factors = zip(*preprocessed_results)
        
        # 마스크도 동일한 scale_factor를 사용하여 전처리
        processed_masks = []
        for i, mask in enumerate(mask_list):
            scale = scale_factors[i]
            if scale > 1:
                h, w = mask.shape[:2]
                new_w, new_h = int(w / scale), int(h / scale)
                resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            else:
                resized_mask = mask
            
            rh, rw = resized_mask.shape[:2]
            top = (512 - rh) // 2
            bottom = 512 - rh - top
            left = (512 - rw) // 2
            right = 512 - rw - left
            processed_masks.append(cv2.copyMakeBorder(
                resized_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0]
            ))
        
        logging.info("모든 이미지와 마스크의 전처리가 완료되었습니다.")

        # 2. & 3. 인페인팅과 후처리를 병렬로 실행
        logging.info(f"GPU 인페인팅과 CPU 후처리를 병렬로 시작합니다 (배치 크기: {batch_size})...")
        
        postprocess_futures = {}
        for i in range(0, num_images, batch_size):
            batch_end = min(i + batch_size, num_images)
            logging.info(f"  - 인페인팅 배치 처리 중: {i+1}-{batch_end} / {num_images}")
            
            # GPU에서 한 배치 인페인팅
            image_batch = list(preprocessed_images[i:batch_end])
            mask_batch = list(processed_masks[i:batch_end])
            inpainted_batch = inpaint_batch_gpu(self.inpaint_session, image_batch, mask_batch)
            
            # 인페인팅이 끝난 배치를 즉시 후처리 작업으로 제출
            logging.info(f"  - 후처리 작업 제출 중: {i+1}-{batch_end} / {num_images}")
            for j, inpainted_img in enumerate(inpainted_batch):
                original_index = i + j
                future = self.executor.submit(
                    run_postprocessing, 
                    inpainted_img, 
                    sizes_before_padding[original_index], 
                    scale_factors[original_index],
                    DEFAULT_UPSCALE_MODEL,
                )
                postprocess_futures[future] = original_index

        # 모든 후처리 작업이 제출된 후, 완료되는 순서대로 결과를 반환
        logging.info("모든 작업이 제출되었습니다. 완료되는 대로 결과를 반환합니다...")
        for future in as_completed(postprocess_futures):
            original_index = postprocess_futures[future]
            try:
                result_image = future.result()
                logging.info(f"이미지 {original_index + 1}의 후처리가 완료되어 반환합니다.")
                yield original_index, result_image
            except Exception as e:
                logging.error(f"이미지 {original_index + 1} 후처리 중 오류 발생: {e}")

    def close(self):
        """파이프라인 종료 시 스레드 풀을 안전하게 닫습니다."""
        if self._created_executor:
            logging.info("내부적으로 생성된 스레드 풀을 종료합니다.")
            self.executor.shutdown()
        else:
            logging.info("외부 스레드 풀을 공유하므로 종료하지 않습니다.")
