# 1. 베이스 이미지로 NVIDIA CUDA 12.3.2, cuDNN 9 런타임 이미지를 선택합니다.
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# 2. 시스템 의존성 및 Python 설치
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip libgl1-mesa-glx libglib2.0-0 libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 의존성 설치
# 모든 의존성을 단일 RUN 명령어로 설치하여 버전 충돌을 원천적으로 방지합니다.
COPY base_requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install \
        --no-cache-dir \
        --ignore-installed \
        -r base_requirements.txt \
        --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 5. 필요한 모든 모듈 복사
COPY inpainting_pipeline /app/inpainting_pipeline
COPY core /app/core
COPY dispatching_pipeline /app/dispatching_pipeline
COPY hosting /app/hosting
COPY rendering_pipeline /app/rendering_pipeline
COPY ocr_pipeline /app/ocr_pipeline
COPY worker.py /app/worker.py

# 6. 환경 변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
# R2, Redis 등 기타 환경변수는 docker-compose.yml에서 설정

# 8. 워커 실행
CMD ["python3", "/app/worker.py"]
