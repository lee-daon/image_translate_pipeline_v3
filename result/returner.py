import json
import logging
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv

# 코어 모듈 임포트 (로컬 core 폴더에서)
from core.config import HOSTING_TASKS_QUEUE, OUTPUT_DIR
from core.redis_client import get_redis_client, initialize_redis, close_redis
from core.image_utils import ImageUtils

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageResultWorker:
    """
    호스팅 큐에서 이미지를 가져와서 로컬에 파일로 저장하는 워커
    """
    
    def __init__(self):
        """초기화"""
        self.redis = get_redis_client()
        self.image_utils = ImageUtils(OUTPUT_DIR)
        
        # 공유 메모리 객체 추적 (정리용)
        self.active_shm_objects = []
        
        logger.info(f"ImageResultWorker 초기화 완료 - 출력 디렉토리: {OUTPUT_DIR}")

    async def _test_redis_connection(self):
        """Redis 연결 테스트"""
        try:
            await self.redis.ping()
            logger.info("Redis 연결 성공")
            return True
        except Exception as e:
            logger.error(f"Redis 연결 실패: {str(e)}")
            return False
        
    async def process_hosting_task(self, task_data: Dict[str, Any]):
        """
        호스팅 작업 처리 (로컬 파일 저장)
        
        Args:
            task_data: 호스팅 작업 데이터 (shm_info 또는 image_url 방식 모두 지원)
        """
        image_id = task_data.get("image_id", "unknown")
        image_url = task_data.get("image_url")  # 이미지 URL
        
        # 로그 메시지
        logger.info(f"task_data: {image_id}")
        
        try:
            
            # 이미지 로드 (shm_info 우선, 없으면 image_url 사용)
            image_array = None
            error = ""
            
                    
            if image_array is None and image_url:
                # URL에서 이미지 다운로드
                image_array = await self.image_utils.download_image_from_url(image_url)
                if image_array is None:
                    error = "URL에서 이미지 다운로드 실패"
                    
            if image_array is None:
                logger.error(f"이미지 로드 실패: {error}")
                return
                
            # 이미지를 로컬 파일로 저장
            success, result = await self.image_utils.save_image_to_file(image_array, image_id)
            
            # 저장 성공 확인
            if not success:
                logger.error(f"이미지 저장 실패: {result}")
                return
                
            # 결과를 JSON 파일에 저장
            file_path = result
            await self.image_utils.save_result_to_json(image_id, file_path)
            
            logger.info(f"파일 저장 완료: {image_id}")
                
        except Exception as e:
            logger.error(f"파일 저장 작업 처리 중 오류: {str(e)}", exc_info=True)
            
    async def start_worker(self):
        """
        파일 저장 워커 시작
        """
        logger.info(f"이미지 파일 저장 워커 시작 - 큐: {HOSTING_TASKS_QUEUE}")
        
        # Redis 연결 테스트
        if not await self._test_redis_connection():
            logger.error("Redis 연결 실패로 워커를 시작할 수 없습니다")
            return
        
        logger.info(f"큐에서 작업 대기 중...")
        
        while True:
            try:
                # 큐에서 작업 가져오기 (timeout=0, 무한 대기)
                task = await self.redis.blpop(HOSTING_TASKS_QUEUE, timeout=0)
                
                queue_name_bytes, task_data_bytes = task
                logger.info(f"새 작업 수신 - 데이터 크기: {len(task_data_bytes)} bytes")
                
                try:
                    task_data = json.loads(task_data_bytes.decode('utf-8'))
                    
                    # 작업 처리 (작업마다 새 태스크 생성)
                    asyncio.create_task(self.process_hosting_task(task_data))
                    
                except json.JSONDecodeError as e:
                    logger.error(f"작업 데이터 디코딩 오류: {task_data_bytes}. 오류: {e}")
                except Exception as proc_e:
                    logger.error(f"작업 처리 중 오류: {proc_e}", exc_info=True)
                
            except asyncio.CancelledError:
                logger.info("워커 태스크 취소됨")
                break
            except Exception as e:
                logger.error(f"작업 처리 루프 중 오류: {str(e)}", exc_info=True)
                logger.info(f"5초 후 재시도...")
                await asyncio.sleep(5)
        
        logger.info("이미지 파일 저장 워커 종료")

async def main():
    """메인 함수"""
    logger.info("Redis 초기화 중...")
    
    try:
        # Redis 초기화
        await initialize_redis()
        logger.info("Redis 초기화 완료")
    except Exception as e:
        logger.error(f"Redis 초기화 실패: {str(e)}")
        return
    
    worker_task = None
    
    try:
        # 워커 초기화 및 시작
        worker = ImageResultWorker()
        worker_task = asyncio.create_task(worker.start_worker())
        
        # 워커 태스크 완료 대기
        await worker_task
        
    except asyncio.CancelledError:
        logger.info("메인 태스크 취소됨")
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt 수신, 워커 종료 중...")
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류: {str(e)}", exc_info=True)
    finally:
        # 워커 태스크 취소
        if worker_task and not worker_task.done():
            logger.info("워커 태스크 취소 중...")
            worker_task.cancel()
            
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
        
        # Redis 연결 종료
        await close_redis()
        logger.info("Redis 연결 종료. 프로그램 종료.")

if __name__ == "__main__":
    asyncio.run(main())
