import os

# Redis 설정
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

# 호스팅 작업 큐 (이미지 파일 저장 요청을 받는 큐)
HOSTING_TASKS_QUEUE = os.environ.get("HOSTING_TASKS_QUEUE", "img:translate:success")

# 공유 메모리 설정
# 공유 메모리 이름에 사용할 접두사 (프로세스 간 충돌 방지)
SHM_NAME_PREFIX = "img_shm_"

# 로깅 설정
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# 이미지 출력 설정
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output/translated")

