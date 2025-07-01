import logging
import os
import io
import asyncio
import tempfile
import shutil
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageFile

# 잘린 이미지 파일도 로드할 수 있도록 허용
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

class OcrProcessor:
    """
    PaddleOCR 모델을 사용하여 이미지에서 텍스트를 추출하는 캡슐화된 클래스.
    CPU 작업과 GPU 작업을 별도의 Executor에서 처리하여 최적화합니다.
    """
    def __init__(self, loop, cpu_executor, gpu_executor, jpeg_quality: int = 95):
        """
        OcrProcessor를 초기화합니다.
        Args:
            loop: 비동기 이벤트 루프.
            cpu_executor: 이미지 전처리 등 CPU 바운드 작업을 위한 실행자.
            gpu_executor: OCR 모델 로딩 및 추론 등 GPU 바운드 작업을 위한 실행자.
            jpeg_quality: JPG 변환 시 사용할 품질.
        """
        self.loop = loop
        self.cpu_executor = cpu_executor
        self.gpu_executor = gpu_executor
        self.jpeg_quality = jpeg_quality
        self.ocr_model = None
        self.temp_dir = tempfile.mkdtemp(prefix="ocr_processor_")
        logger.info(f"OCR Processor's temp directory created: {self.temp_dir}")

    async def initialize_model(self):
        """GPU 실행자에서 PaddleOCR 모델을 비동기적으로 로드합니다."""
        logger.info("Initializing PaddleOCR model in GPU executor...")
        await self.loop.run_in_executor(
            self.gpu_executor, self._load_model_sync
        )
        logger.info("PaddleOCR model initialized successfully.")

    def _load_model_sync(self):
        """[동기] PaddleOCR 모델을 로드하는 내부 함수."""
        try:
            self.ocr_model = PaddleOCR(
                det_algorithm="DB",
                det_model_dir="/root/.paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer",
                det_max_side_len=1504,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                det_db_unclip_ratio=2.0,
                use_dilation=False,
                rec_algorithm="SVTR_LCNet",
                rec_model_dir="/root/.paddleocr/whl/rec/ch/ch_PP-OCRv3_rec_infer",
                rec_image_shape='3, 64, 480',
                rec_char_type='ch',
                max_text_length=25,
                use_space_char=True,
                drop_score=0.5,
                lang="ch",
                use_gpu=True,
                use_fp16=True,
                show_log=True,
                gpu_mem=1000,
                precision='fp32',
                max_batch_size=10
            )
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR model: {e}", exc_info=True)
            raise

    def _prepare_image_sync(self, image_bytes: bytes, image_id: str) -> np.ndarray:
        """[동기] 임시 파일을 사용하지 않고 메모리 내에서 이미지를 준비합니다."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()

            if image.format not in ['JPEG', 'PNG']:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                output_buffer = io.BytesIO()
                image.save(output_buffer, format="JPEG", quality=self.jpeg_quality)
                image = Image.open(output_buffer)
                image.load()

            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Error preparing image {image_id} from bytes: {e}", exc_info=True)
            raise

    def _run_ocr_sync(self, img_array: np.ndarray) -> list:
        """[동기] NumPy 배열에 대해 OCR을 실행하고, 결과를 애플리케이션 포맷에 맞게 변환합니다."""
        if self.ocr_model is None:
            raise RuntimeError("OCR model is not initialized.")
        
        # 1. PaddleOCR 실행 (결과는 [box, (text, score)] 튜플 포함)
        raw_result = self.ocr_model.ocr(img_array, True)

        if raw_result is None or not raw_result or raw_result[0] is None:
            return []
        
        # PaddleOCR v4는 결과 리스트를 한 번 더 감쌀 수 있으므로 내부 리스트를 추출
        result_list = raw_result[0] if isinstance(raw_result[0], list) and isinstance(raw_result[0][0], list) else raw_result
        
        # 2. (text, score) 튜플을 [text, score] 리스트로 변환 (중요!)
        # 기존 JSON 직렬화/역직렬화 과정에서 발생했던 암시적 변환을 명시적으로 재현합니다.
        processed_result = []
        for line in result_list:
            # line[1]이 (text, score) 튜플인 경우, 이를 list로 변환합니다.
            if isinstance(line, list) and len(line) == 2 and isinstance(line[1], tuple):
                line[1] = list(line[1])
            processed_result.append(line)

        return processed_result

    async def process_image(self, image_bytes: bytes, image_id: str, request_id: str) -> list:
        """이미지 바이트에 대해 전체 OCR 파이프라인(전처리, OCR)을 실행합니다."""
        logger.info(f"[{request_id}] Starting OCR pipeline for image: {image_id}")
        
        try:
            img_array = await self.loop.run_in_executor(
                self.cpu_executor, self._prepare_image_sync, image_bytes, image_id
            )
            logger.debug(f"[{request_id}] Image prepared for OCR. Shape: {img_array.shape}")

            ocr_result = await self.loop.run_in_executor(
                self.gpu_executor, self._run_ocr_sync, img_array
            )
            logger.info(f"[{request_id}] OCR processed. Found {len(ocr_result)} text boxes.")
            
            return ocr_result
        except Exception as e:
            logger.error(f"[{request_id}] An error occurred in OCR pipeline for {image_id}: {e}", exc_info=True)
            raise

    def close(self):
        """임시 디렉토리 등 리소스를 정리합니다."""
        try:
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up OCR temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up OCR temp directory on close: {e}") 