import os
import sys
import json
import logging
import signal
import asyncio
import concurrent.futures
import uuid
from typing import List, Dict, Tuple, Optional
from functools import partial

import numpy as np
import cv2
import redis
import aiohttp

# OpenCV 스레드 수 제한 (ThreadPoolExecutor와의 경합 방지)
cv2.setNumThreads(2)

# --- 경로 설정 ---
WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKER_DIR)

# --- 신규 파이프라인 및 기존 모듈 임포트 ---
from inpainting_pipeline.image_inpainter import ImageInpainter
from ocr_pipeline.worker import OcrProcessor
from core.config import (
    LOG_LEVEL,
    WORKER_COLLECT_BATCH_SIZE,
    INPAINTER_GPU_BATCH_SIZE,
    WORKER_BATCH_MAX_WAIT_TIME_SECONDS,
    PROCESSOR_TASK_QUEUE,
    CPU_WORKER_COUNT,
    JPEG_QUALITY,
    MAX_CONCURRENT_TASKS,
    MAX_PENDING_TASKS,
    SHUTDOWN_MAX_WAIT_SECONDS
)
from core.redis_client import initialize_redis, close_redis, get_redis_client, enqueue_error_result, enqueue_success_result, set_task_completion_callback
from core.image_downloader import download_image_async
from dispatching_pipeline.mask import filter_chinese_ocr_result, generate_mask_pure_sync
from dispatching_pipeline.text_translate import process_and_save_translation
from dispatching_pipeline.resize_handler import handle_no_chinese_text_sync
from hosting.r2hosting import R2ImageHosting
from rendering_pipeline.result_check import ResultChecker
from rendering_pipeline.rendering import RenderingProcessor

# 로깅 설정
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('asyncio').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# === 전역 작업 카운터 (서버 안정화용) ===
pending_tasks_count_8473 = 0
tasks_count_lock_8473 = asyncio.Lock()

async def _decrease_pending_count():
    """전역 작업 카운터를 감소시킵니다"""
    global pending_tasks_count_8473
    async with tasks_count_lock_8473:
        pending_tasks_count_8473 -= 1
    logger.debug(f"Task completed, pending count: {pending_tasks_count_8473}")

class AsyncInpaintingWorker:
    """inpainting_pipeline과 내부 메모리를 사용하는 통합 비동기 워커"""
    
    def __init__(self):
        self.cpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=CPU_WORKER_COUNT, thread_name_prefix="cpu-worker"
        )
        self.gpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gpu-worker"
        )
        self.concurrent_task_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
        
        self.inpainter = ImageInpainter(executor=self.cpu_executor)
        self.ocr_processor: Optional[OcrProcessor] = None
        
        self.batch_lock = asyncio.Lock()
        self.task_batch: List[Tuple[np.ndarray, np.ndarray, Dict, np.ndarray]] = []
        self.batch_trigger = asyncio.Event()

        self.r2_hosting = R2ImageHosting()
        self.http_session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._workers: List[asyncio.Task] = []
        self.rendering_processor: Optional[RenderingProcessor] = None
        self.result_checker: Optional[ResultChecker] = None

    async def start_workers(self):
        self._running = True
        self.http_session = aiohttp.ClientSession()
        main_loop = asyncio.get_running_loop()

        self.ocr_processor = OcrProcessor(
            loop=main_loop, 
            cpu_executor=self.cpu_executor,
            gpu_executor=self.gpu_executor,
            jpeg_quality=JPEG_QUALITY
        )
        await self.ocr_processor.initialize_model()
        logger.info("✅ OCR Processor initialized.")

        self.rendering_processor = RenderingProcessor(loop=main_loop)
        self.result_checker = ResultChecker(
            cpu_executor=self.cpu_executor,
            rendering_processor=self.rendering_processor,
            http_session=self.http_session
        )
        logger.info("✅ Rendering modules initialized.")

        redis_listener = asyncio.create_task(self._redis_listener_worker("redis-listener"))
        batch_processor = asyncio.create_task(self._inpainting_batch_processor("inpainting-processor"))
        self._workers = [redis_listener, batch_processor]
        logger.info(f"🚀 Started {len(self._workers)} main workers.")

    async def stop_workers(self):
        self._running = False
        if self.inpainter:
            self.inpainter.close()
        if self.ocr_processor:
            self.ocr_processor.close()
        
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        
        if self.http_session:
            await self.http_session.close()
        
        self.cpu_executor.shutdown(wait=True)
        self.gpu_executor.shutdown(wait=True)
        logger.info("Stopped all workers and shut down thread pools.")

    async def run_cpu_task(self, func, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(
            self.cpu_executor, partial(func, *args, **kwargs)
        )

    async def process_task_from_redis(self, task_data: dict):
        """Redis에서 받은 작업을 처리하여 전체 파이프라인을 실행"""
        # request_id가 없으면 새로 생성하여 task_data에 추가
        request_id = task_data.get("request_id")
        if not request_id:
            request_id = str(uuid.uuid4())
            task_data["request_id"] = request_id
            logger.info(f"Generated new request_id: {request_id} for image_id: {task_data.get('image_id')}")
            
        image_id = task_data.get("image_id")
        try:
            image_url = task_data.get("image_url")
            
            # 1. OCR 실행 (리팩토링으로 로직 위치 이동)
            image_bytes = await download_image_async(self.http_session, image_url, request_id)
            if image_bytes is None:
                await enqueue_error_result(request_id, image_id, "Image download failed")
                return

            ocr_result = await self.ocr_processor.process_image(image_bytes, image_id, request_id)
            
            # 2. 원래의 task_data 구조를 완벽하게 재현
            task_data['ocr_result'] = ocr_result

            # 3. 여기부터는 원래 operate_worker의 로직을 그대로 실행
            is_long = task_data.get("is_long", False)
            filtered_ocr = filter_chinese_ocr_result(task_data.get("ocr_result") or [], request_id)

            if not filtered_ocr:
                final_url = await self.run_cpu_task(
                    handle_no_chinese_text_sync, 
                    image_bytes, 
                    image_url, 
                    request_id, 
                    image_id, 
                    is_long,
                    self.r2_hosting
                )
                await enqueue_success_result(request_id, image_id, final_url or image_url)
                return
                
            # 마스크 생성 (래퍼 함수 없이 직접 호출)
            mask_gen_result = await self.run_cpu_task(
                generate_mask_pure_sync, image_bytes, filtered_ocr, request_id
            )
            image_array, mask_array = mask_gen_result

            # 번역 작업 (기존 코드와 같이 filtered_ocr을 task_data에 추가)
            task_data["filtered_ocr_result"] = filtered_ocr
            await process_and_save_translation(task_data, image_url, self.result_checker)

            # 인페인팅 배치에 추가
            task_info = {"request_id": request_id, "image_id": image_id, "is_long": is_long}
            async with self.batch_lock:
                self.task_batch.append((image_array, mask_array, task_info, image_array))
            
            if len(self.task_batch) >= WORKER_COLLECT_BATCH_SIZE:
                self.batch_trigger.set()
                
        except Exception as e:
            logger.error(f"[{request_id}] Error in task processing: {e}", exc_info=True)
            await enqueue_error_result(request_id, image_id, f"Task processing error: {str(e)}")
        finally:
            # 예외가 발생하더라도 카운터는 감소되어야 함 (단, enqueue_error_result에서 이미 감소시킨 경우 제외)
            # 하지만 enqueue 함수들이 이미 카운터를 관리하므로 여기서는 semaphore만 해제
            self.concurrent_task_semaphore.release()

    async def _redis_listener_worker(self, name: str):
        """Redis에서 작업을 가져와 처리"""
        global pending_tasks_count_8473
        logger.info(f"Worker '{name}' started, listening on '{PROCESSOR_TASK_QUEUE}'.")
        while self._running:
            try:
                # 대기 작업 수가 최대치를 넘으면 최대치 아래로 내려올 때까지 기다림
                while True:
                    async with tasks_count_lock_8473:
                        current_pending = pending_tasks_count_8473
                    if current_pending < MAX_PENDING_TASKS:
                        break
                    logger.warning(f"Too many pending tasks ({current_pending}), waiting for completion...")
                    await asyncio.sleep(1)
                    if not self._running:
                        return

                await self.concurrent_task_semaphore.acquire()
                try:
                    # timeout=0으로 설정하여 작업이 올 때까지 무한정 대기 (가장 효율적)
                    task_tuple = await get_redis_client().blpop([PROCESSOR_TASK_QUEUE], timeout=0)

                    # blpop(timeout=0)은 항상 값을 반환하므로 task_tuple은 None이 될 수 없음
                    async with tasks_count_lock_8473:
                        pending_tasks_count_8473 += 1
                    logger.debug(f"Task received, pending count: {pending_tasks_count_8473}")

                    task_data = json.loads(task_tuple[1].decode('utf-8'))
                    asyncio.create_task(self.process_task_from_redis(task_data))

                except (redis.exceptions.RedisError, json.JSONDecodeError) as e:
                    logger.error(f"Error in '{name}': {e}", exc_info=True)
                    self.concurrent_task_semaphore.release()
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    # 작업자 종료 시 대기 중이던 semaphore를 해제
                    self.concurrent_task_semaphore.release()
                    break
            except Exception as e:
                # 예상치 못한 에러 발생 시 semaphore 누수 방지
                if self.concurrent_task_semaphore.locked():
                    self.concurrent_task_semaphore.release()
                logger.critical(f"Unexpected error in '{name}': {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _inpainting_batch_processor(self, name: str):
        """주기적으로 또는 트리거에 의해 깨어나 배치 처리"""
        logger.info(f"Worker '{name}' started.")
        while self._running:
            try:
                await asyncio.wait_for(self.batch_trigger.wait(), timeout=WORKER_BATCH_MAX_WAIT_TIME_SECONDS)
            except asyncio.TimeoutError:
                pass
            finally:
                self.batch_trigger.clear()
            
            async with self.batch_lock:
                if not self.task_batch:
                    continue
                batch_to_process = list(self.task_batch)
                self.task_batch.clear()
            
            await self._process_batch(batch_to_process)

    async def _process_batch(self, batch: List[Tuple[np.ndarray, np.ndarray, Dict, np.ndarray]]):
        if not batch: return

        images, masks, tasks_info, original_image_arrays = zip(*batch)
        request_ids = [info['request_id'] for info in tasks_info]
        logger.info(f"Processing batch of {len(batch)} tasks. IDs: {request_ids}")

        try:
            results_iterator = self.inpainter.process_images(
                image_list=list(images), mask_list=list(masks),
                batch_size=INPAINTER_GPU_BATCH_SIZE
            )
            
            result_tasks = [
                self._handle_inpainting_result(tasks_info[idx], result_img, original_image_arrays[idx])
                for idx, result_img in results_iterator
            ]
            await asyncio.gather(*result_tasks)

        except Exception as e:
            logger.error(f"Failed to process inpainting batch: {e}", exc_info=True)
            error_tasks = [
                enqueue_error_result(task['request_id'], task['image_id'], "Inpainting batch failed")
                for task in tasks_info
            ]
            await asyncio.gather(*error_tasks)

    async def _handle_inpainting_result(self, task_info: Dict, result_image: np.ndarray, original_image_array: np.ndarray):
        """인페인팅 결과를 ResultChecker로 전달"""
        request_id = task_info["request_id"]
        try:
            inpainting_data = {
                **task_info, 
                "inpainted_image": result_image,
                "original_image_array": original_image_array
            }
            await self.result_checker.save_inpainting_result(request_id, inpainting_data)
            logger.info(f"[{request_id}] Inpainting result (numpy array) saved and forwarded.")
        except Exception as e:
            logger.error(f"[{request_id}] Failed to handle inpainting result: {e}", exc_info=True)
            await enqueue_error_result(request_id, task_info['image_id'], "Result handling failed")

async def main():
    global pending_tasks_count_8473
    worker = AsyncInpaintingWorker()
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)
    
    try:
        await initialize_redis()
        
        # Redis client에 작업 완료 콜백 설정
        set_task_completion_callback(_decrease_pending_count)
        
        await worker.start_workers()
        logger.info("🚀 Inpainting Worker started successfully.")
        await stop_event.wait()

    finally:
        logger.info("Shutting down workers...")
        
        # Graceful shutdown: 대기 작업이 있으면 최대 100초까지 기다림
        async with tasks_count_lock_8473:
            current_pending = pending_tasks_count_8473
        
        if current_pending == 0:
            logger.info("No pending tasks, shutting down immediately.")
        else:
            logger.info(f"Waiting for {current_pending} pending tasks to complete (max {SHUTDOWN_MAX_WAIT_SECONDS}s)...")
            
            for i in range(SHUTDOWN_MAX_WAIT_SECONDS):
                async with tasks_count_lock_8473:
                    current_pending = pending_tasks_count_8473
                
                if current_pending == 0:
                    logger.info(f"All pending tasks completed after {i+1}s.")
                    break
                    
                await asyncio.sleep(1)
                
                if (i + 1) % 10 == 0:  # 10초마다 로그 출력
                    logger.info(f"Still waiting... {current_pending} tasks pending ({i+1}s elapsed)")
            else:
                logger.warning(f"Shutdown timeout reached. {current_pending} tasks still pending.")
        
        await worker.stop_workers()
        await close_redis()
        logger.info("Worker shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Critical error: {e}", exc_info=True)
        exit(1)
