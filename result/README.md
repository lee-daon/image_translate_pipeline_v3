# Result 모듈

이 모듈은 이미지 처리 파이프라인의 결과 저장을 담당하는 독립 실행 가능한 컴포넌트입니다. 호스팅 큐에서 이미지 데이터를 받아 로컬 파일로 저장하는 역할을 수행합니다.

## 주요 기능

- Redis 큐에서 호스팅 작업을 가져와 처리
- 공유 메모리(SHM) 또는 URL에서 이미지 데이터 로드
- 로컬 디렉토리에 이미지 파일 저장 (JPEG 형식)
- 처리 결과를 JSON 파일로 기록
- 자원 관리 및 정리

## 프로젝트 구조

```
result/
├── returner.py          # 메인 워커 애플리케이션
├── core/                # 내장 코어 모듈
│   ├── __init__.py
│   ├── config.py        # 설정 관리
│   ├── redis_client.py  # Redis 클라이언트
│   ├── shm_manager.py   # 공유 메모리 관리
│   └── image_utils.py   # 이미지 처리 유틸리티
├── dockerfile           # Docker 빌드 파일
├── requirements.txt     # Python 패키지 의존성
└── README.md           # 이 파일
```

## 의존성

필요한 패키지들은 `requirements.txt`에 명시되어 있습니다:

```
opencv-python>=4.8.0
numpy>=1.21.0
python-dotenv>=0.21.0
redis>=4.5.0
aiohttp>=3.8.0
```

## 환경 변수 설정

`.env` 파일에 다음 환경 변수들을 설정할 수 있습니다:

- `REDIS_URL`: Redis 서버 URL (기본값: redis://localhost:6379)
- `HOSTING_TASKS_QUEUE`: 호스팅 작업 큐 이름 (기본값: hosting_tasks)
- `OUTPUT_DIR`: 출력 디렉토리 경로 (기본값: ./output/translated)

- `LOG_LEVEL`: 로그 레벨 (기본값: INFO)

## 사용 방법

### 단독 실행

1. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```

2. 환경 변수 설정 (선택사항):
   ```bash
   # .env 파일 생성
   echo "REDIS_URL=redis://localhost:6379" > .env
   echo "OUTPUT_DIR=./output/translated" >> .env
   ```

3. 모듈 실행:
   ```bash
   python returner.py
   ```

### Docker 사용

```bash
# 이미지 빌드
docker build -t image-result-worker .

# 컨테이너 실행 (볼륨 마운트 포함)
docker run -v ./output:/app/output \
           -e REDIS_URL=redis://your-redis-host:6379 \
           image-result-worker
```

## 아키텍처

### 메인 컴포넌트

- `ImageResultWorker`: 메인 워커 클래스
  - `_get_image_from_shm()`: 공유 메모리에서 이미지 로드
  - `process_hosting_task()`: 파일 저장 작업 처리
  - `start_worker()`: 워커 프로세스 시작

### 코어 모듈

- `core.image_utils.ImageUtils`: 이미지 처리 유틸리티
  - `download_image_from_url()`: URL에서 이미지 다운로드
  - `save_image_to_file()`: 로컬 파일로 이미지 저장
  - `save_result_to_json()`: 결과를 JSON 파일에 기록

- `core.redis_client`: Redis 연결 관리
- `core.shm_manager`: 공유 메모리 관리
- `core.config`: 설정 값 관리

## 출력 구조

### 디렉토리 구조
```
output/
└── translated/
    ├── image_001_20241201_143022.jpg
    ├── image_002_20241201_143045.jpg
    └── results.json
```

### 파일 명명 규칙
- 이미지 파일: `{image_id}_{timestamp}.jpg`
- 타임스탬프 형식: `YYYYMMDD_HHMMSS`

### 결과 JSON 구조
```json
[
  {
    "timestamp": "2024-12-01T14:30:22.123456",
    "request_id": "req_001",
    "image_id": "image_001",
    "file_path": "/path/to/output/translated/image_001_20241201_143022.jpg",
    "status": "completed"
  }
]
```

## API 구조

### Redis 큐 메시지 형식

워커는 다음 두 가지 방식의 이미지 소스를 지원합니다:

#### 1. 공유 메모리 방식
```json
{
  "request_id": "req_001",
  "image_id": "image_001",
  "shm_info": {
    "shm_name": "img_shm_12345",
    "shape": [1024, 1024, 3],
    "dtype": "uint8",
    "size": 3145728
  }
}
```

#### 2. URL 방식
```json
{
  "request_id": "req_002",
  "image_id": "image_002",
  "image_url": "https://example.com/image.jpg"
}
```

## 로깅

모듈은 상세한 로깅을 제공하며, 다음 정보들을 기록합니다:
- 작업 시작/완료 상태
- 이미지 소스 타입 (SHM/URL)
- 오류 및 예외 상황
- 파일 저장 결과
- 리소스 관리 상태

## 독립 실행

이 모듈은 완전히 독립적으로 실행 가능하며, 필요한 모든 코어 기능이 내장되어 있습니다. 외부 의존성 없이 Redis 서버만 있으면 실행할 수 있습니다.
