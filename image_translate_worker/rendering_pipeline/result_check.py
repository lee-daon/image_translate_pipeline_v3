import json
import logging
import asyncio
from typing import Dict, Any, Optional
import concurrent.futures
import os
import cv2
import numpy as np

# 로깅 설정
logger = logging.getLogger(__name__)

class ResultChecker:
    def __init__(self, cpu_executor: concurrent.futures.ThreadPoolExecutor,
                 rendering_processor, http_session):
        """
        번역 결과와 인페인팅 결과를 내부 메모리에서 확인하고 렌더링 작업을 ThreadPool에 제출하는 클래스
        """
        self.cpu_executor = cpu_executor
        self.rendering_processor = rendering_processor
        self.http_session = http_session
        
        self.translation_results: Dict[str, Dict] = {}
        self.inpainting_results: Dict[str, Dict] = {}   
        self.result_lock = asyncio.Lock()

    async def save_translation_result(self, request_id: str, data: dict):
        """번역 결과를 내부 메모리에 저장하고 렌더링 가능성 확인"""
        async with self.result_lock:
            self.translation_results[request_id] = data
            logger.debug(f"[{request_id}] Translation result saved to memory")
            await self._check_and_trigger_rendering(request_id)

    async def save_inpainting_result(self, request_id: str, data: dict):
        """인페인팅 결과를 내부 메모리에 저장하고 렌더링 가능성 확인"""
        async with self.result_lock:
            # data 에는 'inpainted_image' (numpy 배열)가 포함되어 있음
            self.inpainting_results[request_id] = data
            logger.debug(f"[{request_id}] Inpainting result (numpy array) saved to memory")
            await self._check_and_trigger_rendering(request_id)

    async def _check_and_trigger_rendering(self, request_id: str):
        """두 결과가 모두 준비되면 렌더링 트리거"""
        translation_data = self.translation_results.get(request_id)
        inpainting_data = self.inpainting_results.get(request_id)
        
        if translation_data and inpainting_data:
            logger.info(f"[{request_id}] Both results ready, triggering rendering")
            
            try:
                await self._trigger_rendering_internal(request_id, translation_data, inpainting_data)
            except Exception as e:
                logger.error(f"[{request_id}] Error in rendering trigger: {e}", exc_info=True)
            finally:
                # 사용한 데이터 정리
                self.translation_results.pop(request_id, None)
                self.inpainting_results.pop(request_id, None)
                logger.debug(f"[{request_id}] Results cleaned from memory")

    async def _trigger_rendering_internal(self, request_id: str, translation_data: dict, inpainting_data: dict):
        """메모리에서 직접 받은 이미지 배열로 렌더링 실행"""
        inpainted_image = inpainting_data.get("inpainted_image")
        if inpainted_image is None:
            logger.error(f"[{request_id}] Inpainted image (numpy array) not found")
            return

        original_image_url = translation_data.get("image_url")
        if not original_image_url:
            raise ValueError("Original image URL not found")
            
        original_image_bytes = await self._download_image_async(original_image_url, request_id)
        if not original_image_bytes:
            raise ValueError("Failed to download original image")

        rendering_task_data = {
            "request_id": request_id,
            "image_id": translation_data.get("image_id"),
            "translate_data": translation_data,
            "inpainted_image": inpainted_image, # 스레드에 바로 전달
            "original_image_bytes": original_image_bytes,
            "is_long": inpainting_data.get("is_long", False)
        }
        
        logger.info(f"[{request_id}] Submitting rendering task to ThreadPool")
        self.cpu_executor.submit(self.rendering_processor.process_rendering_sync, rendering_task_data)
            
    # 레거시 호환성 메서드는 더 이상 필요 없음

    async def _download_image_async(self, image_url: str, request_id: str) -> Optional[bytes]:
        if not self.http_session:
            logger.error(f"[{request_id}] HTTP Session is not initialized.")
            return None
        try:
            if image_url.startswith('//'):
                image_url = 'https:' + image_url
            max_retries = 3
            retry_delay = 2
            for attempt in range(max_retries):
                try:
                    async with self.http_session.get(image_url) as response:
                        response.raise_for_status()
                        return await response.read()
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        logger.warning(f"[{request_id}] Download error ({e}), retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        raise e
        except Exception as e:
            logger.error(f"[{request_id}] Download failed after retries: {e}", exc_info=True)
            return None
