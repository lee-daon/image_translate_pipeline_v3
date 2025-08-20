import logging
import os
import io
import tempfile
import shutil
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageFile

# config import 추가
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import OCR_SHOW_LOG

# 잘린 이미지 파일도 로드할 수 있도록 허용
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

class OcrProcessor:
    """
    PaddleOCR 모델을 사용하여 이미지에서 텍스트를 추출하는 캡슐화된 클래스.
    CPU 작업과 GPU 작업을 별도의 Executor에서 처리하여 최적화합니다.
    """
    def __init__(self, loop, cpu_executor, gpu_executor):
        """
        OcrProcessor를 초기화합니다.
        Args:
            loop: 비동기 이벤트 루프.
            cpu_executor: 이미지 전처리 등 CPU 바운드 작업을 위한 실행자.
            gpu_executor: OCR 모델 로딩 및 추론 등 GPU 바운드 작업을 위한 실행자.
        """
        self.loop = loop
        self.cpu_executor = cpu_executor
        self.gpu_executor = gpu_executor
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
        """[동기] PaddleOCR 모델을 로드하는 내부 함수 (PaddleOCR v3.x 호환)."""
        try:
            # v3.x API: 디바이스와 문서 처리 부가 모델 비활성화, v5 서버 모델 사용
            # 모델 디렉토리 수동 지정 대신, 모델 이름으로 로드
            self.ocr_model = PaddleOCR(
                device="gpu:0",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_detection_model_name="PP-OCRv5_server_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                text_rec_score_thresh=0.85,
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
                image.save(output_buffer, format="JPEG", quality=100)
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
        """[동기] NumPy 배열에 대해 OCR을 실행하고, 결과를 [[box], [text, score]] 리스트로 변환합니다.

        PaddleOCR v3.x의 predict API만 사용합니다. 예외는 상위로 전파합니다.
        """
        if self.ocr_model is None:
            raise RuntimeError("OCR model is not initialized.")

        pages = self.ocr_model.predict(img_array)
        if not pages:
            return []

        page = pages[0]

        # 정답: page 자체에서 필드 추출 (res 없이 직접 보유)
        parse_path = "page"
        res_candidate = page

        def pick(container, key):
            if container is None:
                return None
            if isinstance(container, dict):
                return container.get(key)
            return getattr(container, key, None)

        rec_texts = pick(res_candidate, "rec_texts") or pick(page, "rec_texts") or []
        rec_scores = pick(res_candidate, "rec_scores") or pick(page, "rec_scores") or []
        rec_polys = pick(res_candidate, "rec_polys") or pick(page, "rec_polys") or []

        # numpy → list 변환 보정
        def to_list_safe(val):
            try:
                if isinstance(val, np.ndarray):
                    return val.tolist()
            except Exception:
                pass
            return val

        rec_texts = to_list_safe(rec_texts)
        rec_scores = to_list_safe(rec_scores)
        rec_polys = to_list_safe(rec_polys)

        # 수집된 경로/타입/길이 로깅
        try:
            logger.debug(
                "predict parse mode=%s; types: texts=%s scores=%s polys=%s; lens: t=%s s=%s p=%s",
                parse_path,
                type(rec_texts).__name__, type(rec_scores).__name__, type(rec_polys).__name__,
                (len(rec_texts) if hasattr(rec_texts, "__len__") else "-"),
                (len(rec_scores) if hasattr(rec_scores, "__len__") else "-"),
                (len(rec_polys) if hasattr(rec_polys, "__len__") else "-")
            )
        except Exception:
            pass

        if not (len(rec_texts) == len(rec_scores) == len(rec_polys)):
            logger.warning(
                "Predict result length mismatch: texts=%d scores=%d polys=%d",
                len(rec_texts), len(rec_scores), len(rec_polys)
            )

        processed_result = []
        for poly, text, score in zip(rec_polys, rec_texts, rec_scores):
            processed_result.append([to_list_safe(poly), [str(text), float(score)]])
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