# Image Translate Pipeline
## 개요

이 프로젝트는 이미지 번역 파이프라인을 구현하며, 여러 단계의 이미지 처리 작업을 효율적으로 처리하기 위해 Redis 큐와 워커 패턴을 활용합니다. 시스템은 **OCR 워커**, **Operate 워커**, **Result 워커** 3개의 핵심 워커로 구성되어 워커 간 통신 오버헤드를 줄이고 아키텍처를 단순화했습니다.

## 현재 아키텍처: 큐-워커 패턴

1.  **OCR 워커 (`ocr_worker`):**
    *   **역할:** API 서버로부터 `ocr_tasks` 큐를 통해 이미지 처리 요청을 받습니다. PaddleOCR을 사용해 이미지에서 텍스트를 추출합니다.
    *   **출력:** 추출된 텍스트 정보(`ocr_result`)를 `processor_tasks` 큐에 넣어 Operate 워커에게 전달합니다.

2.  **Operate 워커 (`operate_worker`):**
    *   **역할:** 파이프라인의 핵심으로, OCR 결과를 받아 **번역, 인페인팅, 렌더링**을 모두 처리하는 통합 워커입니다.
    *   **핵심 기술:** 내장된 `inpainting_pipeline` 모듈과 공유 `ThreadPoolExecutor`를 사용하여 모든 작업을 효율적으로 관리합니다. 이미지 데이터는 프로세스 내부 메모리에서 직접 전달되어 I/O 오버헤드를 최소화합니다.

3.  **Result 워커 (`result`):**
    *   **역할:** `hosting_tasks` 큐를 통해 최종 결과 이미지 URL을 수신하고, 메인 서버에 작업 완료를 알리는 등 후속 처리를 담당합니다.

## Operate 워커 상세 동작 원리

`operate_worker`는 단일 워커지만, 내부는 여러 컴포넌트와 단계로 구성된 효율적인 비동기 파이프라인입니다.

### 1. 초기화 및 시작

1.  **공유 스레드 풀 생성:** 워커가 시작될 때, 모든 CPU 집약적 작업을 처리할 공유 `ThreadPoolExecutor`가 생성됩니다.
2.  **컴포넌트 초기화:**
    *   `ImageInpainter`: 인페인팅 파이프라인의 메인 클래스로, 워커의 공유 스레드 풀을 받아 초기화됩니다.
    *   `ResultChecker`: 번역과 인페인팅 결과의 동기화를 담당하며, 마찬가지로 공유 스레드 풀을 사용합니다.
    *   `aiohttp.ClientSession`: 이미지 다운로드 및 외부 API 호출을 위한 비동기 HTTP 클라이언트입니다.
3.  **내부 워커 태스크 시작:**
    *   **Redis 리스너:** `processor_tasks` 큐에서 작업을 받아옵니다.
    *   **배치 프로세서:** 주기적으로 또는 배치가 가득 차면 인페인팅 파이프라인을 실행합니다.

### 2. 단일 작업 처리 흐름

1.  **작업 수신:** Redis 리스너가 `processor_tasks` 큐에서 OCR 결과가 포함된 작업을 가져옵니다.
2.  **작업 분기:**
    *   **번역할 텍스트가 없는 경우:** 이미지를 리사이즈하여 R2에 업로드하고, 최종 URL을 `hosting_tasks` 큐로 바로 전달합니다.
    *   **번역할 텍스트가 있는 경우:** 번역과 인페인팅 경로를 병렬로 진행합니다.

#### 2-1. 번역 경로 (비동기 I/O 위주)

*   `logic.text_translate` 모듈이 외부 번역 API (Gemini)를 비동기적으로 호출합니다.
*   번역 결과는 `ResultChecker`의 내부 메모리(`translation_results`)에 저장됩니다.

#### 2-2. 인페인팅 경로 (CPU 스레드 풀 위주)

*   **마스크 생성:** `generate_mask_pure_sync` 함수가 **공유 스레드 풀**에서 실행되어 마스크(Numpy 배열)를 생성합니다.
*   **배치 추가:** 생성된 이미지와 마스크 배열이 작업 정보와 함께 `AsyncInpaintingWorker`의 `task_batch` 리스트에 직접 추가됩니다.

### 3. 통합 처리 및 동기화

*   **배치 처리:** '배치 프로세서'가 `ImageInpainter.process_images`를 호출하여 배치에 쌓인 모든 이미지와 마스크를 한 번에 처리합니다.
    *   `ImageInpainter`는 내부적으로 전처리, ONNX 추론, 후처리를 모두 수행하며, 이 과정 역시 **공유 스레드 풀**을 활용합니다.
*   **결과 동기화:** 처리된 인페인팅 이미지(Numpy 배열)는 `ResultChecker`의 내부 메모리(`inpainting_results`)에 저장됩니다.
*   `ResultChecker`는 특정 `request_id`에 대한 번역과 인페인팅 결과가 모두 준비되면, 최종 렌더링 작업을 트리거합니다.

### 4. 최종 렌더링

*   `RenderingProcessor`가 **공유 스레드 풀**에서 실행됩니다.
*   인페인팅된 이미지 배열 위에 번역된 텍스트를 그려 최종 이미지를 생성합니다.
*   완성된 이미지는 R2 스토리지에 업로드되고, 이 이미지의 URL이 `hosting_tasks` Redis 큐에 추가되어 **Result 워커**에게 전달됩니다.

## Redis 큐 및 데이터 스키마

1.  **`ocr_tasks` (List):** API 서버 → OCR 워커
2.  **`processor_tasks` (List):** OCR 워커 → Operate 워커
3.  **`hosting_tasks` (List):** Operate 워커 → Result 워커
4.  **`error_queue` (List):** 모든 워커 → 에러 로깅/처리 시스템

*중간 결과(번역, 인페인팅)는 더 이상 Redis에 저장되지 않고, Operate 워커의 내부 메모리에서 직접 관리됩니다.*

