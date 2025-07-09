import json
import logging
import asyncio
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# 코어 모듈 임포트 (로컬 core 폴더에서)
from core.config import REDIS_URL, HOSTING_TASKS_QUEUE, OUTPUT_DIR, JPEG_QUALITY
from core.redis_client import get_redis_client, initialize_redis, close_redis
from core.shm_manager import get_array_from_shm, cleanup_shm
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
        self.image_utils = ImageUtils(OUTPUT_DIR, JPEG_QUALITY)
        
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
    
    async def _get_image_from_shm(self, shm_info: Dict[str, Any]) -> Tuple[Optional[np.ndarray], str]:
        """
        공유 메모리에서 이미지 로드
        
        Args:
            shm_info: 공유 메모리 정보 (shm_name, shape, dtype, size 포함)
            
        Returns:
            (이미지 배열, 오류 메시지) 튜플
        """
        try:
            if isinstance(shm_info, str):
                # 과거 버전 호환성을 위해 문자열(shm_name)만 받는 경우 처리
                logger.warning(f"문자열 shm_info 수신: {shm_info}. 기본값 사용.")
                shm_info = {
                    'shm_name': shm_info,
                    'shape': [1024, 1024, 3],  # 기본 해상도
                    'dtype': 'uint8',
                    'size': 1024 * 1024 * 3
                }
            
            # 필수 필드 확인
            if not shm_info.get('shm_name'):
                raise ValueError("shm_info에 'shm_name' 누락")
                
            # 공유 메모리에서 배열 가져오기
            img_array, existing_shm = get_array_from_shm(shm_info)
            
            # 정리를 위해 목록에 추가
            self.active_shm_objects.append(existing_shm)
            
            # 배열 복사본 반환
            return img_array.copy(), ""
            
        except Exception as e:
            shm_name = shm_info.get('shm_name', 'unknown') if isinstance(shm_info, dict) else str(shm_info)
            error_msg = f"공유 메모리 {shm_name}에서 이미지 로드 중 오류: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg
    
    async def _cleanup_shm_resources(self):
        """공유 메모리 자원 정리"""
        for shm in self.active_shm_objects:
            try:
                shm.close()
            except Exception as e:
                logger.warning(f"공유 메모리 객체 닫기 중 오류: {str(e)}")
        
        # 목록 비우기
        self.active_shm_objects.clear()
    
    async def process_hosting_task(self, task_data: Dict[str, Any]):
        """
        호스팅 작업 처리 (로컬 파일 저장)
        
        Args:
            task_data: 호스팅 작업 데이터 (shm_info 또는 image_url 방식 모두 지원)
        """
        request_id = task_data.get("request_id", "unknown")
        image_id = task_data.get("image_id", "unknown")
        shm_info = task_data.get("shm_info")  # 전체 shm_info 객체
        image_url = task_data.get("image_url")  # 이미지 URL
        
        # 로그 메시지
        source_type = "SHM" if shm_info else "URL" if image_url else "None"
        logger.info(f"task_data: {image_url}")
        logger.info(f"[{request_id}] 파일 저장 작업 시작 - 이미지: {image_id}, 소스: {source_type}")
        
        try:
            # 필수 데이터 확인
            if not all([request_id != "unknown", image_id != "unknown"]):
                logger.error(f"[{request_id}] 필수 기본 데이터 누락: request_id={request_id}, image_id={image_id}")
                return
                
            if not (shm_info or image_url):
                logger.error(f"[{request_id}] 이미지 소스 누락: shm_info와 image_url 모두 없음")
                return
            
            # 이미지 로드 (shm_info 우선, 없으면 image_url 사용)
            image_array = None
            error = ""
            
            if shm_info:
                # 공유 메모리에서 이미지 로드
                image_array, error = await self._get_image_from_shm(shm_info)
                if image_array is None:
                    logger.warning(f"[{request_id}] SHM 로드 실패, URL 방식으로 시도: {error}")
                    
            if image_array is None and image_url:
                # URL에서 이미지 다운로드
                image_array = await self.image_utils.download_image_from_url(image_url, request_id)
                if image_array is None:
                    error = "URL에서 이미지 다운로드 실패"
                    
            if image_array is None:
                logger.error(f"[{request_id}] 이미지 로드 실패: {error}")
                return
                
            # 이미지를 로컬 파일로 저장
            success, result = await self.image_utils.save_image_to_file(image_array, image_id)
            
            # 저장 성공 확인
            if not success:
                logger.error(f"[{request_id}] 이미지 저장 실패: {result}")
                return
                
            # 결과를 JSON 파일에 저장
            file_path = result
            await self.image_utils.save_result_to_json(image_id, file_path, request_id)
            
            logger.info(f"[{request_id}] 파일 저장 완료: {image_id}")
                
        except Exception as e:
            logger.error(f"[{request_id}] 파일 저장 작업 처리 중 오류: {str(e)}", exc_info=True)
            
        finally:
            # 공유 메모리 리소스 정리 (SHM을 사용한 경우만)
            if shm_info:
                await self._cleanup_shm_resources()
                
                # SHM 이름으로 unlink 호출
                try:
                    shm_name = shm_info.get('shm_name') if isinstance(shm_info, dict) else None
                    if shm_name:
                        cleanup_shm(shm_name)
                except Exception as unlink_e:
                    logger.error(f"[{request_id}] SHM unlink 오류 {shm_name}: {unlink_e}")

    async def start_worker(self, poll_interval: float = 1.0):
        """
        파일 저장 워커 시작
        
        Args:
            poll_interval: 큐 폴링 간격 (초)
        """
        logger.info(f"이미지 파일 저장 워커 시작 - 큐: {HOSTING_TASKS_QUEUE}")
        
        # Redis 연결 테스트
        if not await self._test_redis_connection():
            logger.error("Redis 연결 실패로 워커를 시작할 수 없습니다")
            return
        
        logger.info(f"큐에서 작업 대기 중...")
        
        while True:
            try:
                # 큐에서 작업 가져오기
                task = await self.redis.blpop(HOSTING_TASKS_QUEUE, timeout=int(poll_interval))
                
                if task:
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
                logger.info(f"{poll_interval}초 후 재시도...")
                await asyncio.sleep(poll_interval)
        
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
