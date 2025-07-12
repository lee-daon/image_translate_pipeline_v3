# 이미지 번역 워커 큐 데이터 스키마

이 문서는 이미지 번역 워커 시스템에서 사용하는 각 Redis 큐의 데이터 구조(스키마)를 정의합니다. 외부 번역 파이프라인과 원활한 데이터 교환을 위해 이 스키마를 반드시 준수해야 합니다.

---

### 1. 번역 작업 요청 큐 (`image:translation:queue`)

우리 내부 시스템에서 특정 상품의 이미지 번역이 필요할 때 작업을 요청하는 큐입니다.

- **큐 이름:** `image:translation:queue`
- **데이터 형식:** JSON

#### 필드 설명
- `type` (string, 필수): 번역할 이미지의 종류. 'main_image', 'detail_image', 'option_image' 중 하나여야 합니다.
- `userId` (number|string, 필수): 작업을 요청한 사용자의 고유 ID.
- `productId` (number|string, 필수): 번역할 이미지가 속한 상품의 고유 ID.

#### 예시
```json
{
  "type": "main_image",
  "userId": 1,
  "productId": 868447429276
}
```

---

### 2. 외부 번역 파이프라인 작업 전송 큐 (`img:translate:tasks`)

`mainworker`가 개별 이미지 번역 작업을 생성하여 외부 번역 파이프라인으로 전송하는 큐입니다.

- **큐 이름:** `img:translate:tasks`
- **데이터 형식:** JSON

#### 필드 설명
- `image_url` (string, 필수): 번역할 원본 이미지의 URL.
- `image_id` (string, 필수): 시스템에서 생성한 고유 이미지 식별자. 이 ID는 결과 수신 시 원본 작업을 식별하는 데 사용되므로, **절대 변경되어서는 안 됩니다.**
- `is_long` (boolean, 필수): 상세 이미지처럼 긴 이미지인지 여부.

#### 예시
```json
{
  "image_url": "https://example.com/path/to/image.jpg",
  "image_id": "868447429276-1-main-1",
  "is_long": false
}
```

---

### 3. 번역 성공 결과 수신 큐 (`img:translate:success`)

외부 번역 파이프라인이 이미지 번역을 성공적으로 완료했을 때, 그 결과를 우리 시스템으로 전달하는 큐입니다.

- **큐 이름:** `img:translate:success`
- **데이터 형식:** JSON

#### **[중요]** 필드 설명
- `image_id` (string, 필수): 작업 전송 큐로 받았던 **고유 이미지 식별자를 그대로 반환**해야 합니다.
- `image_url` (string, 필수): 번역이 완료된 **결과 이미지의 URL**.

#### 예시
```json
{
  "image_id": "868447429276-1-main-1",
  "image_url": "https://result.loopton.com/path/to/translated_image.jpg"
}
```
> **주의:** `image_url` 필드가 누락될 경우, 우리 시스템은 결과를 처리하지 못하고 해당 작업의 카운트 감소가 누락될 수 있습니다.

---

### 4. 번역 실패 결과 수신 큐 (`img:translate:error`)

외부 번역 파이프라인에서 작업 처리 중 오류가 발생했을 때, 실패 정보를 전달하는 큐입니다.

- **큐 이름:** `img:translate:error`
- **데이터 형식:** JSON

#### 필드 설명
- `image_id` (string, 필수): 실패한 작업의 고유 이미지 식별자를 그대로 반환해야 합니다.
- `error_message` (string, 필수): 실패 원인을 설명하는 오류 메시지.

#### 예시
```json
{
  "image_id": "868447429276-2-detail-1",
  "error_message": "Failed to download image from source."
}
```
