import numpy as np
import cv2
from typing import Dict, Tuple, Any, List
import logging
import math
from PIL import ImageFont

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextSizeCalculator:
    """
    텍스트 박스에 맞는 적절한 텍스트 크기를 계산하는 클래스 (Pillow 사용)
    """
    
    def __init__(self, font_path: str, initial_size_ratio: float = 1.0, min_font_size: int = 1):
        """
        TextSizeCalculator 초기화
        
        Args:
            font_path: 사용할 TTF 폰트 파일 경로
            initial_size_ratio: 텍스트 박스 높이 대비 초기 글자 크기 비율 (기본값 1.0)
            min_font_size: 최소 폰트 크기 (픽셀 단위)
        """
        self.font_path = font_path
        self.initial_size_ratio = initial_size_ratio
        self.min_font_size = min_font_size
        self.font_cache: Dict[int, ImageFont.FreeTypeFont] = {} # 폰트 캐시 딕셔너리 추가
        try:
            # 폰트 존재 여부 확인 및 테스트 로드 (캐시에 저장)
            self._get_font(10)
            logger.info(f"Font loaded successfully for testing: {self.font_path}")
        except IOError:
            logger.error(f"Failed to load font: {self.font_path}. Please check the path.")
            raise # 폰트 로드 실패 시 에러 발생시켜 초기화 중단
        except Exception as e:
            logger.error(f"Unexpected error during font test loading: {e}", exc_info=True)
            raise
    
    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """
        지정된 크기의 폰트 객체를 캐시에서 가져오거나 로드하여 캐시에 저장합니다.

        Args:
            size: 폰트 크기 (픽셀)

        Returns:
            ImageFont.FreeTypeFont: 로드된 폰트 객체

        Raises:
            IOError: 폰트 파일 로드 실패 시
        """
        if size not in self.font_cache:
            try:
                self.font_cache[size] = ImageFont.truetype(self.font_path, size)
                # logger.debug(f"Font size {size} loaded and cached.") # 필요시 DEBUG 로그 활성화
            except IOError as e:
                logger.error(f"Error loading font '{self.font_path}' at size {size}: {e}")
                raise # 오류를 다시 발생시켜 호출한 쪽에서 처리하도록 함
            except Exception as e:
                logger.error(f"Unexpected error loading font at size {size}: {e}", exc_info=True)
                raise
        return self.font_cache[size]
    
    def _calculate_box_dimensions(self, box: List[List[float]]) -> Tuple[float, float]:
        """
        텍스트 박스의 너비와 높이 계산 (min/max 좌표 사용)
        
        Args:
            box: 텍스트 박스 좌표 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        """
        try:
            # 박스 좌표를 numpy 배열로 변환
            box_points = np.array(box, dtype=np.float32)
            
            # min/max 좌표로 너비/높이 계산
            x_coords = box_points[:, 0]
            y_coords = box_points[:, 1]
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            width = x_max - x_min
            height = y_max - y_min
            
            # 너비 또는 높이가 0 이하인 경우 로깅 및 기본값 반환 방지 (calculate_font_size에서 처리)
            if width <= 0 or height <= 0:
                logger.warning(f"Calculated zero or negative box dimension: width={width}, height={height} for box {box}. Calculation might fail.")
            
            return width, height
        except Exception as e:
            logger.error(f"Error calculating box dimensions: {e} for box {box}", exc_info=True)
            # 오류 발생 시 0, 0 반환하여 이후 단계에서 처리되도록 함
            return 0.0, 0.0
    
    def calculate_font_size(self, text: str, box: List[List[float]]) -> int:
        """
        텍스트가 박스 안에 맞도록 Pillow를 사용하여 폰트 크기(픽셀 단위) 계산
        
        Args:
            text: 텍스트 내용 (단일 라인 가정)
            box: 텍스트 박스 좌표
            
        Returns:
            int: 계산된 폰트 크기 (픽셀 단위)
        """
        try:
            # 박스 크기 계산
            box_width, box_height = self._calculate_box_dimensions(box)
            
            # 텍스트가 비어있거나 박스 크기가 유효하지 않으면 최소 크기 반환
            if not text or box_width <= 0 or box_height <= 0:
                logger.warning(f"Invalid input for font size calculation: text='{text}', box_width={box_width}, box_height={box_height}. Returning min font size.")
                return self.min_font_size

            # 초기 폰트 크기 설정 (박스 높이를 기준으로)
            # 정수형 픽셀 크기로 시작
            current_font_size = max(self.min_font_size, int(box_height * self.initial_size_ratio))
            
            logger.debug(f"Calculating font size for text: '{text[:20]}...', box_width: {box_width:.2f}, box_height: {box_height:.2f}, initial_font_size: {current_font_size}")

            while current_font_size >= self.min_font_size:
                try:
                    # 현재 크기로 폰트 로드 (캐싱된 함수 사용)
                    font = self._get_font(current_font_size)
                    # Pillow의 getbbox 사용 (left, top, right, bottom)
                    # textbbox는 (0,0) 기준이므로 bbox[2]가 너비, bbox[3]이 높이
                    text_bbox = font.getbbox(text)
                    text_width = text_bbox[2] - text_bbox[0] 
                    text_height = text_bbox[3] - text_bbox[1] # 실제 렌더링 높이

                    # 텍스트 너비가 박스 너비 안에 들어오는지 확인
                    # 높이는 초기 크기 설정 시 고려되었으므로 너비만 체크 (단일 라인 가정)
                    if text_width <= box_width:
                        logger.debug(f"Font size {current_font_size} fits.")
                        # 가장 큰 적합한 크기 반환
                        return current_font_size
                    
                except OSError as e:
                    logger.error(f"Error using font '{self.font_path}' at size {current_font_size}: {e}. Trying smaller size.")
                    # 폰트 관련 오류 시 크기 줄이기 시도
                except Exception as e:
                    logger.error(f"Unexpected error during font size calculation at size {current_font_size}: {e}", exc_info=True)
                    # 예기치 않은 오류 발생 시 안전하게 최소 크기 반환 또는 루프 중단
                    break

                # 폰트 크기 축소 (예: 1픽셀씩 또는 비율로)
                # current_font_size -= 1 
                new_size = int(current_font_size * 0.95) # 비율로 줄이기
                if new_size < current_font_size: # 최소 1씩은 줄어들도록 보장
                    current_font_size = new_size
                else:
                    current_font_size -= 1 # 비율 감소가 효과 없으면 1씩 줄임

                # current_font_size = max(self.min_font_size, current_font_size) # 최소 크기 보장

            # 루프를 빠져나왔지만 적합한 크기를 찾지 못한 경우 (매우 작은 박스 등)
            logger.warning(f"Could not find a fitting font size for text '{text[:20]}...' within box width {box_width:.2f}. Returning minimum size {self.min_font_size}.")
            return self.min_font_size
            
        except Exception as e:
            logger.error(f"Error calculating font size: {str(e)}", exc_info=True)
            # 오류 발생 시 안전하게 최소 폰트 크기 반환
            return self.min_font_size
    
    def calculate_font_sizes(self, translate_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        번역 결과 데이터의 각 항목에 대한 폰트 크기(픽셀) 계산
        
        Args:
            translate_data: 번역 결과 데이터
            
        Returns:
            Dict[str, Any]: 폰트 크기 정보가 추가된 번역 결과 데이터
        """
        try:
            # 번역 결과 항목 추출
            translate_result = translate_data.get("translate_result", [])
            
            for i, item in enumerate(translate_result):
                box = item.get("box")
                text = item.get("translated_text", "")
                
                if not box or not text:
                    # 박스나 텍스트가 없으면 기본 폰트 크기 사용
                    translate_result[i]["font_size_px"] = self.min_font_size
                    logger.warning(f"Missing box or text for item {i}. Using min font size.")
                    continue
                
                # 폰트 크기 계산 (픽셀 단위)
                font_size_px = self.calculate_font_size(text, box)
                
                # 결과 데이터에 폰트 크기 정보 추가 (픽셀 단위)
                translate_result[i]["font_size_px"] = font_size_px
            
            # 업데이트된 번역 결과 반환
            translate_data["translate_result"] = translate_result
            return translate_data
            
        except Exception as e:
            logger.error(f"텍스트 크기 계산 중 오류 발생: {str(e)}")
            return translate_data  # 오류 발생 시 원본 데이터 그대로 반환
