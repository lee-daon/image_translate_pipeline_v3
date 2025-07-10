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
    
    def _calculate_box_info(self, box: List[List[float]]) -> Tuple[float, float, float]:
        """
        텍스트 박스의 너비, 높이, 각도 계산 (minAreaRect 사용)
        
        Args:
            box: 텍스트 박스 좌표 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            
        Returns:
            Tuple[float, float, float]: (너비, 높이, 각도)
        """
        try:
            box_points = np.array(box, dtype=np.float32)
            _center, (width, height), angle = cv2.minAreaRect(box_points)

            # 너비와 높이를 정규화하고 그에 따라 각도 조정
            # angle은 [-90, 0) 범위의 값을 가지며, 너비(width)에 대한 각도임.
            # 너비가 높이보다 작으면, 박스가 세로 방향이라는 의미.
            # 이 경우, 너비와 높이를 바꾸고 각도에 90도를 더해 긴 쪽의 각도를 얻음.
            if width < height:
                width, height = height, width
                angle += 90
            
            # 각도를 [-90, 90] 범위로 정규화하여 뒤집힌 텍스트 처리
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180

            # 사용자의 요청: 정규화된 각도를 ±30도로 제한하여 안정성 확보
            if abs(angle) > 30:
                logger.warning(f"Normalized angle {angle:.2f} is outside the allowed range [-30, 30]. "
                               f"Resetting angle to 0 and using axis-aligned box for font size.")
                angle = 0
                # 각도를 0으로 리셋하는 경우, 폰트 크기 계산을 위해 축 정렬된 박스 크기를 사용
                x_coords = box_points[:, 0]
                y_coords = box_points[:, 1]
                width = np.max(x_coords) - np.min(x_coords)
                height = np.max(y_coords) - np.min(y_coords)
            
            # 너비 또는 높이가 0 이하인 경우 경고 로깅
            if width <= 0 or height <= 0:
                logger.warning(f"Calculated zero or negative box dimension: width={width}, height={height} for box {box}. Calculation might fail.")
            
            return width, height, angle
        except Exception as e:
            logger.error(f"Error calculating box dimensions: {e} for box {box}", exc_info=True)
            # 오류 발생 시 0, 0, 0 반환하여 이후 단계에서 처리되도록 함
            return 0.0, 0.0, 0.0
    
    def calculate_font_size(self, text: str, box_width: float, box_height: float) -> int:
        """
        텍스트가 박스 안에 맞도록 이진 검색을 사용하여 폰트 크기(픽셀 단위) 계산
        
        Args:
            text: 텍스트 내용 (단일 라인 가정)
            box_width: 텍스트 박스의 실제 너비
            box_height: 텍스트 박스의 실제 높이
            
        Returns:
            int: 계산된 폰트 크기 (픽셀 단위)
        """
        try:
            # 텍스트가 비어있거나 박스 크기가 유효하지 않으면 최소 크기 반환
            if not text or box_width <= 0 or box_height <= 0:
                logger.warning(f"Invalid input for font size calculation: text='{text}', box_width={box_width}, box_height={box_height}. Returning min font size.")
                return self.min_font_size

            # 이진 검색 범위 설정
            min_size = self.min_font_size
            # 박스 높이를 기준으로 최대 폰트 크기 초기화
            max_size = max(self.min_font_size, int(box_height * self.initial_size_ratio))
            best_size = min_size
            
            logger.debug(f"Calculating font size for text: '{text[:20]}...', box_width: {box_width:.2f}, box_height: {box_height:.2f}, search_range: {min_size}-{max_size}")

            # 이진 검색으로 최적 폰트 크기 찾기
            while min_size <= max_size:
                mid_size = (min_size + max_size) // 2
                
                try:
                    # 현재 크기로 폰트 로드 (캐싱된 함수 사용)
                    font = self._get_font(mid_size)
                    # Pillow의 getbbox 사용 (left, top, right, bottom)
                    text_bbox = font.getbbox(text)
                    text_width = text_bbox[2] - text_bbox[0] 
                    text_height = text_bbox[3] - text_bbox[1] # 실제 렌더링 높이

                    # 텍스트가 박스 안에 맞는지 확인 (너비와 높이 모두 체크)
                    if text_width <= box_width and text_height <= box_height:
                        best_size = mid_size  # 현재 크기 저장
                        min_size = mid_size + 1  # 더 큰 크기 시도
                        logger.debug(f"Font size {mid_size} fits, trying larger")
                    else:
                        max_size = mid_size - 1  # 더 작은 크기 시도
                        logger.debug(f"Font size {mid_size} too large, trying smaller")
                    
                except OSError as e:
                    logger.error(f"Error using font '{self.font_path}' at size {mid_size}: {e}. Trying smaller size.")
                    max_size = mid_size - 1  # 폰트 오류 시 더 작은 크기 시도
                except Exception as e:
                    logger.error(f"Unexpected error during font size calculation at size {mid_size}: {e}", exc_info=True)
                    break

            logger.debug(f"Final font size: {best_size}")
            return best_size
            
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
                    # 박스나 텍스트가 없으면 기본 폰트 크기 및 정보 사용
                    translate_result[i].update({
                        "font_size_px": self.min_font_size,
                        "box_width": 0,
                        "box_height": 0,
                        "box_angle": 0
                    })
                    logger.warning(f"Missing box or text for item {i}. Using min font size and default box info.")
                    continue
                
                # 박스 정보(너비, 높이, 각도) 계산
                box_width, box_height, box_angle = self._calculate_box_info(box)
                
                # 폰트 크기 계산 (픽셀 단위)
                font_size_px = self.calculate_font_size(text, box_width, box_height)
                
                # 결과 데이터에 폰트 크기 및 박스 정보 추가 (픽셀 단위)
                translate_result[i].update({
                    "font_size_px": font_size_px,
                    "box_width": box_width,
                    "box_height": box_height,
                    "box_angle": box_angle
                })
            
            # 업데이트된 번역 결과 반환
            translate_data["translate_result"] = translate_result
            return translate_data
            
        except Exception as e:
            logger.error(f"텍스트 크기 계산 중 오류 발생: {str(e)}")
            return translate_data  # 오류 발생 시 원본 데이터 그대로 반환
