import os
import json
import time
import logging
import numpy as np
import cv2
import asyncio
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
from functools import partial
import redis
import aiohttp
from PIL import Image, ImageDraw, ImageFont

from core.config import (
    RESIZE_TARGET_SIZE,
    FONT_PATH,
    JPEG_QUALITY2
)
from core.redis_client import get_redis_client, enqueue_error_result, enqueue_success_result
from hosting.r2hosting import R2ImageHosting

# 렌더링 관련 모듈 임포트
from rendering_pipeline.modules.selectTextColor import TextColorSelector
from rendering_pipeline.modules.textsize import TextSizeCalculator

# 로깅 설정
logger = logging.getLogger(__name__)



class RenderingProcessor:
    """
    ThreadPool에서 실행되는 렌더링 처리기
    """
    
    def __init__(self, loop, font_path: str = FONT_PATH):
        """
        RenderingProcessor 초기화
        
        Args:
            loop: 메인 asyncio 이벤트 루프
            font_path: 사용할 폰트 파일 경로
        """
        self.main_loop = loop
        # 폰트 파일 경로 저장
        self.font_path = font_path
        if not os.path.exists(self.font_path):
            logger.warning(f"폰트 파일을 찾을 수 없습니다: {self.font_path}")
        
        # 렌더링 워커용 폰트 캐시
        self.font_cache: Dict[int, ImageFont.FreeTypeFont] = {}
        
        # 필요한 모듈 인스턴스 생성
        self.text_color_selector = TextColorSelector()
        try:
            self.text_size_calculator = TextSizeCalculator(font_path=self.font_path)
        except Exception as e:
            logger.error(f"Failed to initialize TextSizeCalculator: {e}", exc_info=True)
            self.text_size_calculator = None 
        
        # R2 호스팅 인스턴스 (최종 결과용만)
        self.r2_hosting = R2ImageHosting()
        
        logger.info("RenderingProcessor 초기화 완료")
    

    def _get_font(self, size: int) -> Optional[ImageFont.FreeTypeFont]:
        """폰트 로드 및 캐싱"""
        if size <= 0:
             logger.warning(f"Requested font size {size} is invalid. Returning None.")
             return None

        if size not in self.font_cache:
            try:
                self.font_cache[size] = ImageFont.truetype(self.font_path, size)
            except IOError as e:
                logger.error(f"Worker failed to load font '{self.font_path}' at size {size}: {e}")
                return None
            except Exception as e:
                logger.error(f"Worker encountered unexpected error loading font at size {size}: {e}", exc_info=True)
                return None
        return self.font_cache[size]

    def process_rendering_sync(self, task_data: dict):
        """렌더링 처리 (순수 동기 함수 - ThreadPool에서 실행)"""
        request_id = task_data.get("request_id", "unknown")
        
        try:
            logger.info(f"[{request_id}] Starting rendering in ThreadPool")
            
            image_id = task_data["image_id"]
            translate_data = task_data["translate_data"]
            inpainted_image = task_data["inpainted_image"]
            original_image_bytes = task_data["original_image_bytes"]
            is_long = task_data["is_long"]
            
            # 원본 이미지 디코딩
            original_img_array = np.frombuffer(original_image_bytes, dtype=np.uint8)
            original_image = cv2.imdecode(original_img_array, cv2.IMREAD_COLOR)
            
            if original_image is None:
                raise ValueError("Failed to decode original image")
            
            # 이미지 크기 및 스케일링 처리
            height_scale = 1.0
            width_scale = 1.0
            image_for_text_color_selection = original_image
            rendered_image = inpainted_image
            
            if not is_long:
                # Short: 1024x1024 고정
                original_h, original_w = inpainted_image.shape[:2]
                target_h, target_w = RESIZE_TARGET_SIZE
                
                # 리사이즈
                image_for_text_color_selection = cv2.resize(original_image, (target_w, target_h))
                rendered_image = cv2.resize(inpainted_image, (target_w, target_h))
                
                if original_h > 0 and original_w > 0:
                    height_scale = target_h / original_h
                    width_scale = target_w / original_w
                
                # translate_data의 box 좌표 스케일링
                translate_data = self._scale_bounding_boxes(translate_data, width_scale, height_scale)

            else:
                # Long: 가로 864px 고정, 세로 비율 유지 (원본 크기와 상관없이 무조건 리사이즈)
                original_h, original_w = inpainted_image.shape[:2]
                target_w = 864  # 가로 864px 고정
                scale = target_w / original_w  # 원본이 작으면 확대, 크면 축소
                target_h = int(original_h * scale)
                
                # 무조건 864px로 리사이즈 (작은 이미지도 확대)
                image_for_text_color_selection = cv2.resize(original_image, (target_w, target_h))
                rendered_image = cv2.resize(inpainted_image, (target_w, target_h))
                
                # translate_data의 box 좌표 스케일링
                translate_data = self._scale_bounding_boxes(translate_data, scale, scale)

            # 폰트 크기 계산
            if self.text_size_calculator:
                try:
                    translate_data = self.text_size_calculator.calculate_font_sizes(translate_data)
                    logger.debug(f"[{request_id}] Font sizes calculated")
                except Exception as e:
                    logger.error(f"[{request_id}] Font size calculation failed: {e}")
            
            # 텍스트 색상 선택
            try:
                translate_data = self.text_color_selector.select_text_color(
                    request_id=request_id,
                    translate_data=translate_data,
                    original_image=image_for_text_color_selection,
                    inpainted_image=rendered_image
                )
                logger.debug(f"[{request_id}] Text colors selected")
            except Exception as e:
                logger.error(f"[{request_id}] Text color selection failed: {e}")
            
            # 텍스트 렌더링
            rendered_image = self._draw_texts_on_image(rendered_image, translate_data)
            
            # 최종 결과를 R2에 업로드
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # image_id에서 productId 추출 (첫 번째 '-'로 분리)
            if '-' in image_id:
                product_id = image_id.split('-', 1)[0]
                remaining_part = image_id.split('-', 1)[1]
            else:
                product_id = image_id
                remaining_part = ""
            
            # 파일명 구성: remaining_part + '-' + request_id의 첫 5글자
            final_image_id = f"{remaining_part}-{request_id[:5]}" if remaining_part else f"{image_id}-{request_id[:5]}"
            
            upload_result = self.r2_hosting.upload_image_from_array(
                image_array=rendered_image,
                image_id=final_image_id,
                sub_path=f'translated_image/{current_date}/{product_id}',
                file_ext='.jpg',
                quality=JPEG_QUALITY2,
                metadata={
                    "request_id": request_id,
                    "image_id": image_id
                }
            )
            
            if upload_result["success"]:
                final_image_url = upload_result["url"]
                logger.info(f"[{request_id}] Final rendering uploaded: {final_image_url}")
                
                # 메인 이벤트 루프에서 비동기 함수를 안전하게 실행
                coro = self._send_to_hosting_queue(request_id, image_id, final_image_url)
                asyncio.run_coroutine_threadsafe(coro, self.main_loop)
            else:
                logger.error(f"[{request_id}] Final upload failed: {upload_result.get('error')}")
                # 업로드 실패 시 에러 큐로 전송
                coro = enqueue_error_result(request_id, image_id, f"Upload failed: {upload_result.get('error')}")
                asyncio.run_coroutine_threadsafe(coro, self.main_loop)
                
        except Exception as e:
            logger.error(f"[{request_id}] Rendering error in ThreadPool: {e}", exc_info=True)
            # 렌더링 전반적인 오류 시 에러 큐로 전송
            coro = enqueue_error_result(request_id, task_data.get("image_id", "N/A"), f"Rendering error: {str(e)}")
            asyncio.run_coroutine_threadsafe(coro, self.main_loop)

    async def _send_to_hosting_queue(self, request_id: str, image_id: str, image_url: str):
        """성공 큐에 최종 결과 전송"""
        try:
            await enqueue_success_result(request_id, image_id, image_url)
            logger.info(f"[{request_id}] Final result sent to success queue: {image_url}")
        except Exception as e:
            logger.error(f"[{request_id}] Failed to send to success queue: {e}", exc_info=True)

    def _scale_bounding_boxes(self, translate_data: dict, width_scale: float, height_scale: float) -> dict:
        """바운딩 박스 좌표를 주어진 비율로 스케일링합니다."""
        if "translate_result" in translate_data:
            for item in translate_data["translate_result"]:
                if "box" in item and item["box"]:
                    original_box_coords = np.array(item["box"])
                    scaled_box_coords = original_box_coords.astype(np.float32)
                    scaled_box_coords[:, 0] *= width_scale
                    scaled_box_coords[:, 1] *= height_scale
                    item["box"] = scaled_box_coords.tolist()
        return translate_data

    def _draw_texts_on_image(self, image: np.ndarray, translate_data: dict) -> np.ndarray:
        """PIL을 사용하여 모든 텍스트를 이미지에 효율적으로 렌더링합니다."""
        if not translate_data or "translate_result" not in translate_data:
            return image

        try:
            # BGR -> RGB 변환을 한 번만 수행
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            # 텍스트 렌더링을 위한 배치 처리
            for item in translate_data["translate_result"]:
                text = item.get("translated_text")
                box = item.get("box")
                text_color = item.get("text_color")
                font_size = item.get("font_size_px", 20)

                if not (text and box and text_color):
                    continue

                font = self._get_font(max(1, int(font_size)))
                if font is None:
                    continue
                
                # 박스 좌표를 정수형으로 변환 (한 번만)
                box_np = np.array(box, dtype=np.int32)
                x_min, y_min = box_np.min(axis=0)
                x_max, y_max = box_np.max(axis=0)
                width = x_max - x_min
                height = y_max - y_min

                # RGB 색상 변환 (한 번만)
                rgb_text_color = (text_color.get("r", 0), text_color.get("g", 0), text_color.get("b", 0))

                # 멀티라인 텍스트 처리 최적화
                lines = text.split('\n')
                if len(lines) == 1:
                    # 단일 라인 텍스트 - 빠른 경로
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    line_width = text_bbox[2] - text_bbox[0]
                    line_height = text_bbox[3] - text_bbox[1]
                    
                    text_x = x_min + (width - line_width) // 2
                    text_y = y_min + (height - line_height) // 2
                    
                    draw.text((text_x, text_y), text, fill=rgb_text_color, font=font)
                else:
                    # 멀티라인 텍스트 - 기존 로직 유지하되 최적화
                    total_text_width = 0
                    total_text_height = 0
                    line_heights = []
                    line_widths = []

                    # 모든 라인의 크기를 한 번에 계산
                    for line in lines:
                        text_bbox = draw.textbbox((0, 0), line, font=font)
                        line_width = text_bbox[2] - text_bbox[0]
                        line_height = text_bbox[3] - text_bbox[1]
                        
                        line_widths.append(line_width)
                        line_heights.append(line_height)
                        total_text_width = max(total_text_width, line_width)
                        total_text_height += line_height

                    # 텍스트 시작 위치 계산
                    text_x = x_min + (width - total_text_width) // 2
                    text_y = y_min + (height - total_text_height) // 2

                    # 각 라인 렌더링
                    current_y = text_y
                    for i, line in enumerate(lines):
                        # 각 라인의 중앙 정렬
                        line_x = x_min + (width - line_widths[i]) // 2
                        draw.text((line_x, current_y), line, fill=rgb_text_color, font=font)
                        current_y += line_heights[i]
            
            # RGB -> BGR 변환을 한 번만 수행
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.error(f"Text rendering error: {e}", exc_info=True)
            return image

