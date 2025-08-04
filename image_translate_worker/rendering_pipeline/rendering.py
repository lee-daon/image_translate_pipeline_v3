import os

import logging
import numpy as np
import cv2
import asyncio
from datetime import datetime
from typing import Dict, Optional
from PIL import Image, ImageDraw, ImageFont

from core.config import (
    RESIZE_TARGET_SIZE,
    FONT_PATH,
    MASK_PADDING_PIXELS
)
from core.redis_client import enqueue_error_result, enqueue_success_result
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

    def _get_clean_sampling_mask(self, current_box_np: np.ndarray, global_inpainted_mask: np.ndarray, height: int, width: int):
        """
        주어진 박스 주변의 '깨끗한' 샘플링 마스크와 인페인트 영역 마스크를 생성합니다.
        
        Args:
            current_box_np: 현재 처리 중인 박스의 numpy 배열
            global_inpainted_mask: 모든 인페인팅 영역을 포함하는 마스크
            height: 이미지 높이
            width: 이미지 너비
            
        Returns:
            (np.ndarray, np.ndarray): (깨끗한 샘플링 마스크, 현재 박스의 인페인트 영역 마스크)
        """
        # 공통 설정
        sampling_start_offset = MASK_PADDING_PIXELS + 4
        sampling_ring_thickness = 25
        
        # 색상 보정 적용 영역은 MASK_PADDING_PIXELS + 1 만큼 확장
        correction_area_padding = MASK_PADDING_PIXELS + 1
        kernel_size_inpainted = (correction_area_padding * 2) + 1
        
        # 현재 박스 기준 마스크 생성
        current_box_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(current_box_mask, [current_box_np], 255)
        
        # 현재 박스의 인페인팅 영역 (색상 보정 적용 대상)
        inpainted_area_mask = cv2.dilate(current_box_mask, np.ones((kernel_size_inpainted, kernel_size_inpainted), np.uint8))
        
        # 잠재적 샘플링 링 계산
        kernel_size_outer = ((sampling_start_offset + sampling_ring_thickness) * 2) + 1
        outer_ring_mask = cv2.dilate(current_box_mask, np.ones((kernel_size_outer, kernel_size_outer), np.uint8))
        kernel_size_inner = (sampling_start_offset * 2) + 1
        inner_ring_mask = cv2.dilate(current_box_mask, np.ones((kernel_size_inner, kernel_size_inner), np.uint8))
        potential_sampling_ring = cv2.bitwise_and(outer_ring_mask, cv2.bitwise_not(inner_ring_mask))
        
        # 다른 인페인팅 영역을 제외하여 '깨끗한' 샘플링 링만 남김
        clean_sampling_mask = cv2.bitwise_and(potential_sampling_ring, cv2.bitwise_not(global_inpainted_mask))
        
        return clean_sampling_mask, inpainted_area_mask

    def _correct_global_color(self, original_image: np.ndarray, inpainted_image: np.ndarray, translate_data: dict, request_id: str) -> np.ndarray:
        """
        각 텍스트 박스 주변의 '로컬' 색상을 사용하여 인페인팅된 영역의 색감을 보정합니다.
        원본과 인페인팅된 이미지의 동일한 주변부 영역을 비교하여 색상 변형량을 계산하고, 이를 인페인트 영역에 역적용합니다.
        """
        try:
            height, width = original_image.shape[:2]

            # 1. LAB 색상 공간으로 변환
            original_lab = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
            inpainted_lab = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2LAB)
            l_inpainted, a_inpainted, b_inpainted = cv2.split(inpainted_lab)

            # 2. 모든 텍스트 박스를 포함하는 전체 인페인팅 영역 마스크 생성 (오염 지도)
            all_boxes_base_mask = np.zeros((height, width), dtype=np.uint8)
            if "translate_result" in translate_data:
                for item in translate_data["translate_result"]:
                    if item.get("box"):
                        box = np.array(item["box"], dtype=np.int32)
                        if cv2.contourArea(box) > 0:
                            cv2.fillPoly(all_boxes_base_mask, [box], 255)
            
            # 오염 지도는 더 넓게 설정하여 안전 마진을 확보
            global_mask_padding = MASK_PADDING_PIXELS + 3
            kernel_size_global = (global_mask_padding * 2) + 1
            global_inpainted_mask = cv2.dilate(all_boxes_base_mask, np.ones((kernel_size_global, kernel_size_global), np.uint8))

            # 3. 각 박스를 순회하며 색상 보정 적용
            for item in translate_data.get("translate_result", []):
                if not item.get("box"):
                    continue

                box_np = np.array(item["box"], dtype=np.int32)
                if cv2.contourArea(box_np) < 1:
                    continue

                # 3-1. '깨끗한' 샘플링 마스크와 현재 박스의 인페인트 마스크 가져오기
                clean_sampling_mask, inpainted_area_mask = self._get_clean_sampling_mask(
                    box_np, global_inpainted_mask, height, width
                )

                # 3-2. 픽셀 추출
                source_pixels = original_lab[clean_sampling_mask > 0]
                reference_pixels = inpainted_lab[clean_sampling_mask > 0]

                if source_pixels.shape[0] < 50 or reference_pixels.shape[0] < 50:
                    logger.warning(f"[{request_id}] Not enough clean pixels in sampling ring for local color correction. Skipping a box.")
                    continue
                
                # 3-3. 색상 통계 계산
                l_mean_src, a_mean_src, b_mean_src = np.mean(source_pixels, axis=0)
                l_std_src, a_std_src, b_std_src = np.std(source_pixels, axis=0)

                l_mean_ref, a_mean_ref, b_mean_ref = np.mean(reference_pixels, axis=0)
                l_std_ref, a_std_ref, b_std_ref = np.std(reference_pixels, axis=0)

                l_std_ref = max(l_std_ref, 1e-6)
                a_std_ref = max(a_std_ref, 1e-6)
                b_std_ref = max(b_std_ref, 1e-6)

                # 3-4. 인페인트된 영역의 픽셀에 색상 변환 적용
                l_patch = l_inpainted[inpainted_area_mask > 0]
                a_patch = a_inpainted[inpainted_area_mask > 0]
                b_patch = b_inpainted[inpainted_area_mask > 0]

                l_new = (l_patch - l_mean_ref) * (l_std_src / l_std_ref) + l_mean_src
                a_new = (a_patch - a_mean_ref) * (a_std_src / a_std_ref) + a_mean_src
                b_new = (b_patch - b_mean_ref) * (b_std_src / b_std_ref) + b_mean_src

                l_inpainted[inpainted_area_mask > 0] = np.clip(l_new, 0, 255)
                a_inpainted[inpainted_area_mask > 0] = np.clip(a_new, 0, 255)
                b_inpainted[inpainted_area_mask > 0] = np.clip(b_new, 0, 255)

            # 4. 수정된 채널을 병합하고 BGR로 변환
            corrected_lab = cv2.merge([l_inpainted, a_inpainted, b_inpainted])
            corrected_bgr = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
            
            logger.info(f"[{request_id}] Local color correction applied successfully to text boxes.")
            return corrected_bgr

        except Exception as e:
            logger.error(f"[{request_id}] Local color correction failed: {e}. Returning original inpainted image.", exc_info=True)
            return inpainted_image

    def process_rendering_sync(self, task_data: dict):
        """렌더링 처리 (순수 동기 함수 - ThreadPool에서 실행)"""
        request_id = task_data.get("request_id", "unknown")
        
        try:
            logger.info(f"[{request_id}] Starting rendering in ThreadPool")
            
            image_id = task_data["image_id"]
            translate_data = task_data["translate_data"]
            inpainted_image = task_data["inpainted_image"]
            original_image = task_data["original_image_array"]
            is_long = task_data["is_long"]
            
            if original_image is None:
                raise ValueError("Failed to get original image array")
            
            if inpainted_image is None:
                raise ValueError("Failed to get inpainted image array")
            
            if translate_data is None:
                raise ValueError("Failed to get translate data")
            
            if inpainted_image.shape != original_image.shape:
                original_h, original_w = original_image.shape[:2]
                inpainted_image = cv2.resize(inpainted_image, (original_w, original_h), interpolation=cv2.INTER_AREA)

            # 1. 최종 출력 크기 결정
            original_h, original_w = original_image.shape[:2]
            if not is_long:
                target_h, target_w = RESIZE_TARGET_SIZE
                resized_original = cv2.resize(original_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
                resized_inpainted = cv2.resize(inpainted_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
            else:
                target_h, target_w = original_h, original_w
                resized_original = original_image
                resized_inpainted = inpainted_image
            
            # 3. 좌표 스케일링
            height_scale = target_h / original_h if original_h > 0 else 0
            width_scale = target_w / original_w if original_w > 0 else 0
            translate_data = self._scale_bounding_boxes(translate_data, width_scale, height_scale)

            # 4. 전역 색상 보정: 인페인팅된 이미지 전체의 색감을 원본의 배경 톤에 맞춤
            color_corrected_inpainted = self._correct_global_color(resized_original, resized_inpainted, translate_data, request_id)
            
            # 5. 고품질 배경 생성: 선명한 원본을 기반으로, 색상이 보정된 인페인팅 영역만 합성
            rendered_image = resized_original.copy()

            # 붙여넣을 전체 영역을 마스크로 계산
            paste_mask = np.zeros((target_h, target_w), dtype=np.uint8)
            if "translate_result" in translate_data:
                for item in translate_data["translate_result"]:
                    if "box" in item and item["box"]:
                        box = np.array(item["box"], dtype=np.int32)
                        if cv2.contourArea(box) > 0:
                            cv2.fillPoly(paste_mask, [box], 255)
            
            # 마스크 확장: 텍스트 박스 + MASK_PADDING_PIXELS + 1
            paste_padding = MASK_PADDING_PIXELS + 1
            kernel = np.ones((paste_padding * 2 + 1, paste_padding * 2 + 1), np.uint8)
            dilated_paste_mask = cv2.dilate(paste_mask, kernel)

            # 확장된 마스크를 사용해 색상 보정된 영역을 한 번에 합성
            rendered_image[dilated_paste_mask > 0] = color_corrected_inpainted[dilated_paste_mask > 0]

            # 6. 렌더링할 텍스트가 있는지 확인
            texts_to_render = [
                item for item in translate_data.get("translate_result", []) 
                if item.get("translated_text", "").strip()
            ]

            if texts_to_render:

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
                        original_image=resized_original,  # 원본 이미지를 전달하여 텍스트 색상 후보 추출
                        inpainted_image=rendered_image    # 최종 배경을 전달하여 대비 계산
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
        """PIL을 사용하여 모든 텍스트를 이미지에 효율적으로 렌더링합니다. (회전 지원)"""
        if not translate_data or "translate_result" not in translate_data:
            return image

        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # 텍스트 렌더링을 위한 배치 처리
            for item in translate_data["translate_result"]:
                text = item.get("translated_text")
                box = item.get("box")
                text_color = item.get("text_color")
                font_size = item.get("font_size_px", 20)
                angle = item.get("box_angle", 0)

                if not (text and box and text_color):
                    continue

                font = self._get_font(max(1, int(font_size)))
                if font is None:
                    continue
                
                # 박스 중앙점 계산
                box_np = np.array(box, dtype=np.float32)
                center_x, center_y = np.mean(box_np, axis=0)

                # RGB 색상 변환
                rgb_text_color = (text_color.get("r", 0), text_color.get("g", 0), text_color.get("b", 0))

                # --- 텍스트 블록 크기 계산 (멀티라인 지원) ---
                lines = text.split('\n')
                
                # Pillow 10.0.0 부터 textbbox가 draw에서 font로 이동 (getbbox)
                # getbbox는 (left, top, right, bottom)을 반환
                total_text_width = 0
                total_text_height = 0
                line_heights = []

                for line in lines:
                    try:
                        # font.getbbox 사용
                        line_bbox = font.getbbox(line)
                        line_width = line_bbox[2] - line_bbox[0]
                        line_height = line_bbox[3] - line_bbox[1]
                    except TypeError:
                        # 일부 오래된 Pillow 버전 호환성
                        line_width, line_height = font.getsize(line)

                    total_text_width = max(total_text_width, line_width)
                    total_text_height += line_height
                    line_heights.append(line_height)

                # --- 임시 투명 캔버스에 텍스트 렌더링 ---
                padding = 10  # 텍스트 잘림 방지용 패딩
                canvas_width = total_text_width + padding * 2
                canvas_height = total_text_height + padding * 2

                txt_canvas = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 0))
                txt_draw = ImageDraw.Draw(txt_canvas)

                current_y = padding
                for i, line in enumerate(lines):
                    try:
                        line_bbox = font.getbbox(line)
                        line_w = line_bbox[2] - line_bbox[0]
                    except TypeError:
                        line_w, _ = font.getsize(line)
                    
                    # 각 라인 중앙 정렬
                    line_x = (canvas_width - line_w) / 2
                    txt_draw.text((line_x, current_y - font.getbbox(line)[1]), line, font=font, fill=rgb_text_color)
                    current_y += line_heights[i]
                
                # --- 캔버스 회전 및 원본 이미지에 합성 ---
                # Pillow는 반시계 방향으로 회전하므로, cv2에서 얻은 각도에 -1을 곱하여 방향을 맞춤
                rotated_canvas = txt_canvas.rotate(-angle, expand=True, resample=Image.BICUBIC)
                
                paste_x = int(center_x - rotated_canvas.width / 2)
                paste_y = int(center_y - rotated_canvas.height / 2)
                
                pil_image.paste(rotated_canvas, (paste_x, paste_y), mask=rotated_canvas)
            
            # RGB -> BGR 변환을 한 번만 수행
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.error(f"Text rendering error: {e}", exc_info=True)
            return image

