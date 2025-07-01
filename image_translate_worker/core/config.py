import os

# Redis 설정
REDIS_URL = os.environ.get("REDIS_URL")

# 로깅 설정
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# === Shared Memory 설정 ===

# === Translation 설정 ===
TRANSLATE_TEXT_RESULT_HASH_PREFIX = "itp_tr_"

# === Worker 공통 설정 ===
# CPU 집약적 작업을 처리할 스레드 수 (시스템 코어 수에 맞춰 조절)
CPU_WORKER_COUNT = int(os.environ.get("CPU_WORKER_COUNT", "16"))
# 동시에 처리할 수 있는 최대 작업 수
MAX_CONCURRENT_TASKS = int(os.environ.get("MAX_CONCURRENT_TASKS", "100"))

# === Inpainting Worker 설정 ===
# 입력 큐 (이제 전체 파이프라인의 시작점이 됨)
PROCESSOR_TASK_QUEUE = "img:translate:tasks"
# 성공/에러 결과를 전달하는 큐
SUCCESS_QUEUE = "img:translate:success"
ERROR_QUEUE = "img:translate:error"
HOSTING_TASKS_QUEUE = SUCCESS_QUEUE # 레거시 호환성을 위한 별칭

# 텍스트 영역 주변 패딩 픽셀 수
MASK_PADDING_PIXELS = int(os.environ.get("MASK_PADDING_PIXELS", "1"))

# 인페인팅 배치 크기
# operate_worker가 인페인팅을 트리거하기 위해 모으는 작업 수
WORKER_COLLECT_BATCH_SIZE = int(os.environ.get("WORKER_COLLECT_BATCH_SIZE", "16"))
# Inpainting 파이프라인 내부에서 GPU에 한번에 올리는 이미지 수
INPAINTER_GPU_BATCH_SIZE = int(os.environ.get("INPAINTER_GPU_BATCH_SIZE", "4"))
# 배치가 다 차지 않았을 때, 처리를 시작하기까지 대기하는 최대 시간 (초)
WORKER_BATCH_MAX_WAIT_TIME_SECONDS = float(os.environ.get("WORKER_BATCH_MAX_WAIT_TIME_SECONDS", "5.0"))

# === 이미지 처리 설정 (from OCR Worker) ===
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "95"))  # JPEG 변환 품질

# === Rendering Pipeline 설정 ===
# 폰트 파일 경로
FONT_PATH = os.environ.get("FONT_PATH", "/app/rendering_pipeline/modules/fonts/GmarketSansTTFBold.ttf")
# 리사이즈 목표 크기 (is_long=false일 때 사용)
RESIZE_TARGET_SIZE = (
    int(os.environ.get("RESIZE_TARGET_HEIGHT", "1024")), 
    int(os.environ.get("RESIZE_TARGET_WIDTH", "1024"))
)

# GPU 설정
USE_CUDA = os.environ.get("USE_CUDA", "1") == "1"

# === HTTP 클라이언트 설정 ===
# 이미지 다운로드 재시도 횟수
IMAGE_DOWNLOAD_MAX_RETRIES = int(os.environ.get("IMAGE_DOWNLOAD_MAX_RETRIES", "3"))
# 재시도 간격 (초)
IMAGE_DOWNLOAD_RETRY_DELAY = int(os.environ.get("IMAGE_DOWNLOAD_RETRY_DELAY", "2"))
