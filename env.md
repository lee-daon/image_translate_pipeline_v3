# .env 파일 대신 사용하는 환경변수 설정 파일입니다.
# 실제 배포 환경에서는 이 파일의 이름을 .env로 변경하고, 민감한 정보(API 키 등)를 안전하게 관리해야 합니다.

# 로깅 설정
LOG_LEVEL="INFO"

# Worker 공통 설정
CPU_WORKER_COUNT="16"
MAX_CONCURRENT_TASKS="100"

# Inpainting Worker 설정
MASK_PADDING_PIXELS="1"
WORKER_COLLECT_BATCH_SIZE="16"
INPAINTER_GPU_BATCH_SIZE="4"
WORKER_BATCH_MAX_WAIT_TIME_SECONDS="5.0"

# 이미지 처리 설정
JPEG_QUALITY="95"

# Rendering Pipeline 설정
FONT_PATH="/app/rendering_pipeline/modules/fonts/GmarketSansTTFBold.ttf"
RESIZE_TARGET_HEIGHT="1024"
RESIZE_TARGET_WIDTH="1024"

# GPU 설정
USE_CUDA="1"

# HTTP 클라이언트 설정
IMAGE_DOWNLOAD_MAX_RETRIES="3"
IMAGE_DOWNLOAD_RETRY_DELAY="2"

# R2 및 Gemini API 키는 .env 파일에서 로드
# 이 값들은 실제 키로 채워야 합니다.
R2_ENDPOINT=""
CLOUDFLARE_ACCESS_KEY_ID=""
CLOUDFLARE_SECRET_KEY=""
R2_BUCKET_NAME=""
R2_DOMAIN=""
