# Redis Queue 구조 (업데이트됨)

## 📊 전체 데이터 흐름

```
외부 요청 → OCR Worker → Operate Worker → Result Returner
     ↓           ↓             ↓              ↓
img:translate:tasks → ocr:results → img:translate:success → 최종 결과
     ↓           ↓             ↓
     ↓   img:translate:error ← img:translate:error ← (에러 발생 시)
     ↓           ↓             ↓
    에러 처리 ← 에러 처리 ← 에러 처리
```

## 🔄 워커별 큐 사용

### 1. OCR Worker (`ocr_worker/worker.py`)

#### 📥 입력 큐
- **큐 이름**: `img:translate:tasks` (**변경됨**)
- **동작**: `redis.blpop([OCR_TASK_QUEUE], timeout=1)`
- **데이터 구조**:
```json
{
  "request_id": "요청 ID",
  "image_url": "이미지 URL",
  "image_id": "이미지 ID", 
  "is_long": boolean
}
```

#### 📤 출력 큐
- **성공 시**: `ocr:results` (내부 통신용)
- **실패 시**: `img:translate:error` (**새로 추가**)

#### 📤 출력 데이터 구조
**성공 시 (`ocr:results`)**:
```json
{
  "request_id": "요청 ID",
  "image_id": "이미지 ID",
  "image_url": "이미지 URL",
  "is_long": boolean,
  "ocr_result": [
    [
      [좌표배열],
      ["텍스트", 신뢰도]
    ]
  ]
}
```

**실패 시 (`img:translate:error`)**:
```json
{
  "request_id": "요청 ID",
  "image_id": "이미지 ID",
  "error_message": "에러 메시지",
  "timestamp": 1234567890
}
```

### 2. Operate Worker (`operate_worker/worker.py`)

#### 📥 입력 큐  
- **큐 이름**: `ocr:results` (== `PROCESSOR_TASK_QUEUE`)
- **동작**: `redis.blpop([PROCESSOR_TASK_QUEUE], timeout=1)`
- **데이터 구조**: OCR Worker 출력과 동일

#### 📤 출력 큐
- **성공 시**: `img:translate:success` (**변경됨**, 기존 `hosting:tasks`)
- **실패 시**: `img:translate:error` (**새로 추가**)

#### 📤 출력 데이터 구조
**성공 시 (`img:translate:success`)**:
```json
{
  "request_id": "요청 ID",
  "image_id": "이미지 ID", 
  "image_url": "최종 렌더링된 이미지 URL"
}
```

**실패 시 (`img:translate:error`)**:
```json
{
  "request_id": "요청 ID",
  "image_id": "이미지 ID",
  "error_message": "에러 메시지",
  "timestamp": 1234567890
}
```

## 🏗️ Operate Worker 내부 큐 (asyncio.Queue)

### 추론 큐들
- **`inference_queue_short`**: Short 타입 GPU 추론 대기열
- **`inference_queue_long`**: Long 타입 GPU 추론 대기열
- **`postprocessing_queue`**: 후처리 작업 대기열

### 처리 흐름
```
입력(ocr:results) → 전처리 → inference_queue → GPU처리 → postprocessing_queue → 렌더링 → 출력(img:translate:success)
     ↓              ↓              ↓              ↓              ↓                ↓
 에러 처리     에러 처리      에러 처리      에러 처리      에러 처리        에러 처리
     ↓              ↓              ↓              ↓              ↓                ↓
img:translate:error (각 단계에서 실패 시)
```

## 📋 큐 설정값

### Redis 큐 (워커 간 통신)
| 큐 이름 | 용도 | 생산자 | 소비자 |
|---------|------|--------|--------|
| `img:translate:tasks` | **번역 작업 요청** | 외부 시스템 | OCR Worker |
| `ocr:results` | OCR 결과 (내부 통신) | OCR Worker | Operate Worker |
| `img:translate:success` | **성공 결과** | Operate Worker | Result Returner |
| `img:translate:error` | **에러 결과** ⭐ | 두 워커 모두 | 에러 처리 시스템 |

### asyncio.Queue (Operate Worker 내부)
| 큐 이름 | 크기 제한 | 용도 |
|---------|-----------|------|
| `inference_queue_short` | 30 | Short 타입 GPU 추론 대기 |
| `inference_queue_long` | 30 | Long 타입 GPU 추론 대기 |  
| `postprocessing_queue` | 50 | 후처리 작업 대기 |

## 🚨 에러 처리 시점

### OCR Worker 에러 케이스
- 이미지 다운로드 실패
- OCR 처리 실패  
- JSON 파싱 실패
- 기타 예상치 못한 에러

### Operate Worker 에러 케이스
- 이미지 다운로드 실패
- 마스크 생성 실패
- 전처리 실패
- GPU 추론 실패
- 후처리 실패
- 번역 실패
- 렌더링 실패
- 이미지 업로드 실패
- JSON 파싱 실패

## 🎯 핵심 변경사항

1. **명확한 네이밍**: 
   - `ocr:tasks` → `img:translate:tasks`
   - `hosting:tasks` → `img:translate:success`

2. **에러 추적 가능**: 
   - 새로운 `img:translate:error` 큐로 모든 실패 케이스 추적
   - `request_id`, `image_id`, `error_message`, `timestamp` 포함

3. **완전한 커버리지**: 
   - 두 워커의 모든 주요 실패 지점에서 에러 큐로 전송
   - 더 이상 "조용히 실패"하는 케이스 없음

4. **백워드 호환성**: 
   - `HOSTING_TASKS_QUEUE = SUCCESS_QUEUE`로 레거시 코드 지원
