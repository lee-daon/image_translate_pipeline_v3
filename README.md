# Image Translate Pipeline v3

## 개요

v3는 단일 워커 컨테이너로 전체 파이프라인(OCR → 번역 → 인페인팅 → 렌더링 → 업로드)을 처리합니다. 외부 시스템은 Redis에 작업을 넣고, 성공/실패 큐로 결과를 받습니다.

## 퍼포먼스
- RTX3060기준 
- 1장 처리-> 1000ms/1장
- 40장 처리-> 250ms/1장
         

## 구성요소
- **컨테이너**: `image_translate_worker` 하나로 통합
- **메인 엔트리**: `image_translate_worker/worker.py`
- **OCR**: `image_translate_worker/ocr_pipeline/worker.py` (PaddleOCR v3.x, PP-OCRv5 server det/rec)
- **인페인팅**: `image_translate_worker/inpainting_pipeline`
- **렌더링**: `image_translate_worker/rendering_pipeline`
- **번역/마스크생성**: `image_translate_worker/dispatching_pipeline`

## 요구사항
- NVIDIA GPU 및 Docker

## 빠른 시작
1) 빌드/실행
```bash
docker compose up --build

## simulation에다가 이미지 url작성->실행
## 자동 생성된 output폴더에서 결과 확인

```

2) 작업 넣기 (입력 큐)
- 큐 이름: `img:translate:tasks`
- 메시지 JSON 스키마:
```json
{
  "image_url": "https://example.com/sample.jpg",
  "image_id": "sample.jpg",
  "is_long": false
}
```
3) 결과 받기 (출력 큐)
- 성공 큐: `img:translate:success`
```json
{
  "image_id": "sample.jpg",
  "image_url": "https://r2.example.com/rendered/abcd.jpg"
}
```
- 실패 큐: `img:translate:error`
```json
{
  "image_id": "sample.jpg",
  "error_message": "에러 설명"
}
```

## 내부 동작 (요약)
1) `worker.py`가 Redis의 `img:translate:tasks`를 BLPOP으로 수신합니다.
2) 이미지 다운로드 → OCR 실행 → 중국어 텍스트 필터링 → 번역 및 마스크 생성 병렬 처리
3) 인페인팅 배치 처리 후 렌더링 → R2 업로드
4) 최종 URL을 성공 큐로, 실패 시 에러 메시지를 실패 큐로 푸시

## 주요 파일
- `image_translate_worker/worker.py`: 이벤트 루프, 배치 처리, 큐 입출력, 에러 처리
- `image_translate_worker/ocr_pipeline/worker.py`: PaddleOCR 로딩/추론
- `image_translate_worker/rendering_pipeline/rendering.py`: 텍스트 렌더링
- `image_translate_worker/dispatching_pipeline/*`: 마스크, 번역 등 보조 로직
- `image_translate_worker/core/redis_client.py`: Redis 연결/결과 푸시
- `image_translate_worker/core/config.py`: 환경변수/상수 정의

## 환경변수 (일부 발췌)
- 환경변수 목록.txt 참고

## Redis 큐 상세
- 입력: `img:translate:tasks` (JSON: `image_url`, `image_id`, `is_long`)
- 성공: `img:translate:success` (JSON: `image_id`, `image_url`)
- 실패: `img:translate:error` (JSON: `image_id`, `error_message`)

## 변경 요약 (v2 → v3)
- 다중 워커 구조 → 단일 컨테이너로 통합
- [v2에서 v3 개선안 문서](document/v2에서v3개선안.md) 참고

