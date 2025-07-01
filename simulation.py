import asyncio
import redis.asyncio as redis
import json
import uuid
import argparse

# 기본 설정값
DEFAULT_REDIS_URL = "redis://localhost:6380"  # Docker Redis 포트 변경
OCR_TASK_QUEUE = "img:translate:tasks"

# 이미지 URL 목록 (is_long=false)
SHORT_IMAGE_URLS = [
    "https://image.loopton.com/resized_12.jpg",
    "https://image.loopton.com/resized_des_1745402182900/resized_10.jpg",
    "https://image.loopton.com/resized_des_1745402182900/resized_11.jpg",
    "https://image.loopton.com/resized_des_1745402182900/resized_12.jpg"
]

# 이미지 URL 목록 (is_long=true)
LONG_IMAGE_URLS = [
    "https://image.loopton.com/resized_des_1745402182900/resized_9.jpg",
    "https://image.loopton.com/resized_des_1745402182900/resized_13.jpg",
    "https://image.loopton.com/resized_des_1745402182900/resized_14.jpg",
    "https://image.loopton.com/resized_des_1745402182900/resized_15.jpg"
]

async def send_ocr_task(redis_client, image_url: str, is_long: bool):
    """Redis에 OCR 작업을 전송합니다."""
    try:
        request_id = str(uuid.uuid4())
        image_id = image_url.split('/')[-1]  # URL에서 파일명 추출
        
        task_data = {
            "request_id": request_id,
            "image_url": image_url,
            "image_id": image_id,
            "is_long": is_long
        }
        
        # JSON으로 직렬화하여 Redis 큐에 추가
        task_json = json.dumps(task_data).encode('utf-8')
        await redis_client.rpush(OCR_TASK_QUEUE, task_json)
        
        print(f"작업 전송 완료: {image_id} (is_long={is_long}), request_id: {request_id}")
        return request_id
        
    except Exception as e:
        print(f"작업 전송 오류 ({image_url}): {e}")
        return None

async def main(redis_url):
    """Redis에 OCR 작업들을 전송합니다."""
    try:
        # Redis 연결
        redis_client = redis.from_url(redis_url, decode_responses=False)
        await redis_client.ping()
        print(f"Redis 연결 성공: {redis_url}")
        
        # 모든 이미지 URL 수집 (짧은 이미지 + 긴 이미지)
        all_tasks = []
        
        # 짧은 이미지들 (is_long=False)
        for url in SHORT_IMAGE_URLS:
            all_tasks.append(send_ocr_task(redis_client, url, False))
            
        # 긴 이미지들 (is_long=True)
        for url in LONG_IMAGE_URLS:
            all_tasks.append(send_ocr_task(redis_client, url, True))
        
        # 모든 작업을 병렬로 전송
        results = await asyncio.gather(*all_tasks)
        
        print("\n--- 모든 작업 전송 완료 ---")
        successful_requests = sum(1 for r in results if r is not None)
        total_images = len(SHORT_IMAGE_URLS) + len(LONG_IMAGE_URLS)
        print(f"총 {total_images}개 이미지 중 {successful_requests}개 작업 전송 성공")
        print(f"짧은 이미지: {len(SHORT_IMAGE_URLS)}개")
        print(f"긴 이미지: {len(LONG_IMAGE_URLS)}개")
        
        # Redis 연결 닫기
        await redis_client.close()
        
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redis에 OCR 작업 전송 시뮬레이션")
    parser.add_argument("--redis_url", default=DEFAULT_REDIS_URL, help="Redis 연결 URL")
    
    args = parser.parse_args()
    
    # 이벤트 루프 정책 설정 (Windows 환경에서 ProactorEventLoop 사용 권장)
    import os
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main(args.redis_url))
