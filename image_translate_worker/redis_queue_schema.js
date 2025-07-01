// Redis 큐 스키마 - Image Translation Pipeline (개선된 아키텍처)

// ===== 1. 워커 간 큐 (Redis List) =====
// 작업 흐름: ocr_worker -> operate_worker
// Queue: processor_tasks (BLPOP으로 수신)
const processor_tasks = {
    "request_id": "647c390b-6279-4633-aa9a-c657379488f8",
    "image_url": "https://img.alicdn.com/example.jpg",
    "image_id": "resized_6.jpg",
    "is_long": true,
    "ocr_result": [
        [
            [[383.0, 79.0], [481.0, 79.0], [481.0, 154.0], [383.0, 154.0]], // box coordinates
            ["05", 0.9856277704238892] // [text, confidence]
        ],
        [
            [[379.0, 162.0], [486.0, 162.0], [486.0, 198.0], [379.0, 198.0]],
            ["卖点介绍", 0.998233437538147]
        ],
        [
            [[213.0, 250.0], [657.0, 250.0], [657.0, 317.0], [213.0, 317.0]],
            ["360°套包工艺", 0.9992042183876038]
        ]
    ]
};

// 작업 흐름: operate_worker -> result_worker
// Queue: hosting_tasks (BLPOP으로 수신)
const hosting_tasks = {
    "request_id": "request_id",
    "image_id": "image_id",
    "image_url": "https://r2.example.com/rendered/final_image.jpg" // 최종 렌더링된 이미지 URL
};


// ===== 2. Operate 워커 내부 메모리 저장소 (Python Dict) =====
// ResultChecker 클래스 내에서 관리됨

// a) 번역 결과 저장소
// result_checker.translation_results[request_id]
const translation_result_internal = {
    "image_id": "image-filename.jpg",
    "image_url": "https://img.alicdn.com/example.jpg",
    "translate_result": [
        {
            "box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
            "translated_text": "번역된 텍스트",
            "original_char_count": 15
        }
    ]
};

// b) 인페인팅 결과 저장소  
// result_checker.inpainting_results[request_id]
const inpainting_result_internal = {
    "request_id": "request_id",
    "image_id": "image_id",
    "is_long": true,
    "inpainted_image": "numpy_array_of_inpainted_image" // 메모리 상의 NumPy 배열 참조
};


// ===== 3. 처리 흐름 요약 =====
/*
워커 기반 이미지 번역 파이프라인 (최종 아키텍처):

1.  **`processor_tasks` 큐 (Redis List):**
    -   OCR Worker가 작업을 완료하고, OCR 결과가 포함된 Task를 이 큐에 넣습니다.
    -   Operate Worker가 이 큐를 리스닝하며 작업을 가져옵니다.

2.  **Operate Worker 내부 처리 (In-Memory):**
    -   하나의 작업이 들어오면 **번역**과 **마스크 생성**이 병렬로 실행됩니다.
    -   **번역 경로:**
        -   외부 번역 API를 비동기 호출합니다.
        -   번역 결과는 `ResultChecker`의 내부 메모리(`translation_results`)에 저장됩니다.
    -   **인페인팅 경로:**
        -   CPU 스레드 풀에서 마스크(Numpy 배열)를 생성합니다.
        -   생성된 이미지와 마스크 배열은 다른 정보와 함께 `AsyncInpaintingWorker`의 배치(`task_batch`) 리스트에 추가됩니다.
    -   **통합 처리:**
        -   배치가 채워지면, `ImageInpainter`가 배치 전체를 받아 인페인팅을 수행합니다.
        -   처리된 인페인팅 이미지(Numpy 배열)는 `ResultChecker`의 내부 메모리(`inpainting_results`)에 저장됩니다.

3.  **결과 동기화 및 렌더링 (In-Memory):**
    -   `ResultChecker`는 특정 `request_id`에 대해 번역과 인페인팅 결과가 모두 준비되면, 최종 렌더링 작업을 스레드 풀에 제출합니다.
    -   렌더링이 완료된 이미지는 R2 스토리지에 업로드됩니다.

4.  **`hosting_tasks` 큐 (Redis List):**
    -   최종 렌더링된 이미지의 URL이 이 큐에 추가됩니다.
    -   Result Worker가 이 작업을 받아 후속 처리를 진행합니다.

**개선 사항:**
-   **데이터 전달:** 더 이상 공유 메모리(SHM)나 임시 파일을 사용하지 않습니다. 모든 이미지 데이터는 프로세스 내부 메모리에서 Numpy 배열로 직접 전달되어 I/O 오버헤드를 제거하고 성능을 극대화합니다.
-   **구조 단순화:** 복잡한 내부 큐 시스템과 Redis Hash를 사용한 중간 결과 저장을 모두 제거하여, 코드의 복잡성을 낮추고 유지보수성을 향상시켰습니다.
*/