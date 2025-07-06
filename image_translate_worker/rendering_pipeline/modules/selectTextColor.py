import numpy as np
import cv2
from typing import List, Dict, Tuple, Any, Optional
import logging
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
import json



# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextColorSelector:
    """
    원본 이미지와 인페인팅된 이미지를 비교하여 적절한 텍스트 색상을 선택하는 클래스
    """
    
    def __init__(self, num_clusters=3):
        self.num_clusters = num_clusters

        
        # ---> 추가: K-Means 클러스터 개수 설정 (원본 텍스트+배경 후보용) <---
        self.num_original_clusters = 2 # 텍스트 + 주 배경색 가정
        # ---> 추가 끝 <---
        

    
    # ---> 함수명 변경 및 로직 수정: _extract_dominant_color_from_roi <---
    def _extract_dominant_color_from_roi(self, image: np.ndarray, box: List[List[float]], 
                                         num_clusters: int = 1) -> Optional[Tuple[int, int, int]]:
        """
        주어진 박스 ROI 내 픽셀들에 대해 K-Means를 사용하여 주요 색상(들) 추출
        (수정: 픽셀 수가 많으면 10% 랜덤 샘플링 사용)
        
        Args:
            image: 분석할 이미지 (원본 또는 인페인팅된 이미지)
            box: 텍스트 박스 좌표
            num_clusters: 찾을 클러스터(색상) 개수
            
        Returns:
            Optional[Tuple[int, int, int]]: BGR 형식의 가장 지배적인 색상 (num_clusters=1일 때), 또는 None
            # num_clusters > 1인 경우 로직 수정 필요 (예: 모든 클러스터 중심 반환)
        """
        # 박스 좌표를 정수로 변환
        box = np.array(box, dtype=np.int32)
        
        # 바운딩 박스의 min/max 좌표 계산
        x_min = max(0, min(box[:, 0]))
        y_min = max(0, min(box[:, 1]))
        x_max = min(image.shape[1], max(box[:, 0]))
        y_max = min(image.shape[0], max(box[:, 1]))
        roi = image[y_min:y_max, x_min:x_max]
        
        # ROI가 비어있거나 너무 작으면 처리 불가
        if roi.size == 0 or roi.shape[0] * roi.shape[1] < num_clusters:
            logger.warning(f"ROI is empty or too small for clustering. Box: {box.tolist()}, ROI shape: {roi.shape}")
            return None
        
        # 픽셀 데이터를 (N, 3) 형태로 변환 (N: 픽셀 수)
        pixels = roi.reshape(-1, 3)
        num_pixels = pixels.shape[0]

        # ---> 추가: 픽셀 샘플링 로직 <---
        sample_size_ratio = 0.1 # 샘플링 비율 (10%)
        min_pixels_for_sampling = 100 # 샘플링을 위한 최소 픽셀 수

        if num_pixels > min_pixels_for_sampling:
            sample_size = max(num_clusters, int(num_pixels * sample_size_ratio)) # 최소 클러스터 수만큼은 샘플링

            perm = np.random.permutation(num_pixels)
            pixels_to_cluster = pixels[perm[:sample_size]]
        else:
            # 픽셀 수가 적으면 모든 픽셀 사용
            pixels_to_cluster = pixels
        # ---> 추가 끝 <---

        try:
            # MiniBatchKMeans 클러스터링 수행 (메모리 효율성 및 속도 향상)
            kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=256)
            # kmeans.fit(pixels) # 이전 코드
            kmeans.fit(pixels_to_cluster) # 수정된 코드
            
            # 클러스터 중심 (BGR 색상) 추출
            dominant_colors_bgr = kmeans.cluster_centers_.astype(int)
            
            # num_clusters=1인 경우, 해당 색상 반환
            if num_clusters == 1:
                # 반환 형식을 튜플로 통일
                return tuple(dominant_colors_bgr[0])
            else:
                # TODO: num_clusters > 1인 경우 처리 로직 추가 필요 (예: 모든 클러스터 중심 반환)
                logger.warning(f"_extract_dominant_color_from_roi called with num_clusters > 1. Returning only the first cluster center.")
                return tuple(dominant_colors_bgr[0])

        except Exception as e:
            logger.error(f"Error during KMeans clustering in ROI: {e}", exc_info=True)
            return None
    # ---> 함수 로직 수정 끝 <---

    # ---> 추가: 원본 ROI에서 색상 후보 추출 함수 <---
    def _get_original_color_candidates(self, original_image: np.ndarray, box: List[List[float]]) -> List[Tuple[int, int, int]]:
        """
        원본 이미지의 박스 ROI에서 K-Means를 사용하여 색상 후보 목록 추출
        (수정: 픽셀 수가 많으면 10% 랜덤 샘플링 사용)

        Args:
            original_image: 원본 이미지
            box: 텍스트 박스 좌표

        Returns:
            List[Tuple[int, int, int]]: BGR 형식의 색상 후보 목록
        """
        candidates = []
        try:
            # 박스 좌표 정수 변환 및 ROI 추출
            box_int = np.array(box, dtype=np.int32)
            x_min = max(0, min(box_int[:, 0]))
            y_min = max(0, min(box_int[:, 1]))
            x_max = min(original_image.shape[1], max(box_int[:, 0]))
            y_max = min(original_image.shape[0], max(box_int[:, 1]))
            roi = original_image[y_min:y_max, x_min:x_max]

            if roi.size == 0 or roi.shape[0] * roi.shape[1] < self.num_original_clusters:
                logger.warning(f"Original ROI is empty or too small for clustering. Box: {box}, ROI shape: {roi.shape}")
                return []

            pixels = roi.reshape(-1, 3)
            num_pixels = pixels.shape[0]

            # ---> 추가: 픽셀 샘플링 로직 (위 함수와 동일) <---
            sample_size_ratio = 0.1 # 샘플링 비율 (10%)
            min_pixels_for_sampling = 100 # 샘플링을 위한 최소 픽셀 수
            num_target_clusters = self.num_original_clusters

            if num_pixels > min_pixels_for_sampling:
                sample_size = max(num_target_clusters, int(num_pixels * sample_size_ratio))
                perm = np.random.permutation(num_pixels)
                pixels_to_cluster = pixels[perm[:sample_size]]
            else:
                pixels_to_cluster = pixels
            # ---> 추가 끝 <---

            # MiniBatchKMeans 실행 (메모리 효율성 및 속도 향상)
            kmeans = MiniBatchKMeans(n_clusters=self.num_original_clusters, random_state=0, batch_size=256)
            kmeans.fit(pixels_to_cluster) # 수정된 코드

            # 클러스터 중심(색상 후보) 추출 및 튜플 변환
            candidate_colors_bgr = kmeans.cluster_centers_.astype(int)
            candidates = [tuple(color) for color in candidate_colors_bgr]
            
        except Exception as e:
            logger.error(f"Error getting original color candidates: {e}", exc_info=True)
            
        return candidates
    # ---> 추가 끝 <---
    
    def _calculate_luminance(self, color: Tuple[int, int, int]) -> float:
        """
        RGB 색상의 상대 휘도 계산 (sRGB 공간)
        
        Args:
            color: BGR 색상 튜플
            
        Returns:
            float: 색상의 상대 휘도
        """
        # BGR에서 RGB로 변환
        r, g, b = color[2] / 255.0, color[1] / 255.0, color[0] / 255.0
        
        # sRGB 값을 선형 RGB로 변환
        r = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
        
        # 상대 휘도 계산
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    def _calculate_contrast_ratio(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """
        두 색상 간의 대비율 계산
        
        Args:
            color1, color2: BGR 색상 튜플
            
        Returns:
            float: 두 색상 간의 대비율
        """
        l1 = self._calculate_luminance(color1) + 0.05
        l2 = self._calculate_luminance(color2) + 0.05
        
        # 밝은 색상을 l1으로 설정
        if l1 < l2:
            l1, l2 = l2, l1
            
        # 대비율 계산: (L1 + 0.05) / (L2 + 0.05)
        return l1 / l2
    
    def _adjust_color_for_contrast(self, text_color: Tuple[int, int, int], 
                                  bg_color: Tuple[int, int, int], 
                                  target_ratio: float = 4.5) -> Tuple[int, int, int]:
        """
        배경색과의 대비율이 낮을 경우, 텍스트 색상을 강제로 조정 (검정/흰색)
        
        Args:
            text_color: 텍스트 색상
            bg_color: 배경색
            target_ratio: 목표 대비율 (WCAG 1.4.3 기준은 4.5:1)
            
        Returns:
            Tuple[int, int, int]: 조정된 (또는 강제 지정된) 텍스트 색상 (BGR)
        """
        current_ratio = self._calculate_contrast_ratio(text_color, bg_color)
        
        if current_ratio < target_ratio:
            # 배경 휘도 계산
            bg_lum = self._calculate_luminance(bg_color)
            
            # 배경이 밝으면 (휘도 > 0.5) -> 텍스트는 검정색
            if bg_lum > 0.5:
                return (0, 0, 0) # BGR: Black
            # 배경이 어두우면 (휘도 <= 0.5) -> 텍스트는 흰색
            else:
                return (255, 255, 255) # BGR: White
        else:
            # 이미 대비율이 충분하면 원래 텍스트 색상 반환
            return text_color
    
    def select_text_color(self, request_id: str, translate_data: Dict[str, Any],
                          # ---> 수정: 인자 설명 변경 <---
                          original_image: np.ndarray,
                          inpainted_image: np.ndarray
                          # ---> 수정 끝 <---
                          # original_shm_info: Dict[str, Any], # 삭제
                          # inpainted_shm_info: Dict[str, Any] # 삭제
                          ) -> Dict[str, Any]:
        """
        번역된 텍스트에 적합한 색상을 선택 (KMeans 기반)
        
        Args:
            request_id: 로깅 및 추적을 위한 요청 ID
            translate_data: 번역 결과 데이터
            # ---> 수정: 인자 설명 변경 <---
            original_image: 원본 이미지 (NumPy 배열)
            inpainted_image: 인페인팅된 이미지 (NumPy 배열)
            # ---> 수정 끝 <---

        Returns:
            Dict[str, Any]: 색상 정보가 추가된 번역 결과 데이터
        """
        try:

            # 번역 결과에 색상 정보 추가
            translate_result = translate_data.get("translate_result", [])
            
            # ---> 수정 시작: 새로운 로직 적용 <---
            for i, item in enumerate(translate_result): 
                box = item.get("box")
                
                if not box:
                    logger.warning(f"[{request_id}] No box found for item {i}. Skipping color selection.")
                    # 기본값 설정 또는 오류 처리 필요
                    item["text_color"] = {"r": 0, "g": 0, "b": 0} # Default: Black
                    item["bg_color"] = {"r": 255, "g": 255, "b": 255} # Default: White
                    item["contrast_ratio"] = 21.0
                    continue
                
                # 1. 인페인팅된 배경색 추출 (KMeans k=1)
                inpainted_bg_color = self._extract_dominant_color_from_roi(
                    inpainted_image, box, num_clusters=1
                ) 
                if inpainted_bg_color is None:
                    logger.warning(f"[{request_id}] Could not extract inpainted background color for item {i}. Using default white.")
                    inpainted_bg_color = (255, 255, 255) # 기본 흰색
                
                # 2. 원본 이미지에서 색상 후보 추출 (KMeans k=num_original_clusters)
                original_candidates = self._get_original_color_candidates(original_image, box)
                
                # 후보가 없으면 기본 검정 텍스트 사용
                if not original_candidates:
                    logger.warning(f"[{request_id}] No original color candidates found for item {i}. Using default black text.")
                    chosen_text_color = (0, 0, 0) # 기본 검정
                else:
                    # 3. 후보 중 배경과 대비가 가장 높은 색상 선택
                    best_contrast = -1
                    chosen_text_color = original_candidates[0] # 기본값으로 첫 번째 후보
                    
                    for candidate_color in original_candidates:
                        contrast = self._calculate_contrast_ratio(candidate_color, inpainted_bg_color)
                        # logger.debug(f"[{request_id}] Candidate {candidate_color} vs BG {inpainted_bg_color} -> Contrast: {contrast:.2f}") # 후보별 DEBUG 로그 제거
                        if contrast > best_contrast:
                            best_contrast = contrast
                            chosen_text_color = candidate_color
                
                # 4. 최종 대비율 확인 및 강제 조정
                final_contrast = self._calculate_contrast_ratio(chosen_text_color, inpainted_bg_color)
                adjusted_text_color = self._adjust_color_for_contrast(
                    chosen_text_color, inpainted_bg_color, target_ratio=4.5
                )

                # 조정이 발생했다면 최종 대비율 다시 계산
                if adjusted_text_color != chosen_text_color:
                    final_contrast = self._calculate_contrast_ratio(adjusted_text_color, inpainted_bg_color)
                    logger.info(f"[{request_id}] Text color adjusted to {adjusted_text_color}. Final contrast: {final_contrast:.2f}")
                else:
                    pass # 명시적으로 아무 작업도 안 함을 표시
                    
                # 결과 저장 (BGR -> RGB)
                item["text_color"] = {"r": adjusted_text_color[2], "g": adjusted_text_color[1], "b": adjusted_text_color[0]}
                item["bg_color"] = {"r": inpainted_bg_color[2], "g": inpainted_bg_color[1], "b": inpainted_bg_color[0]}
                item["contrast_ratio"] = round(final_contrast, 2)
            
            # ---> 수정 끝 <---
            
            # 업데이트된 번역 결과 반환
            translate_data["translate_result"] = translate_result
            
            return translate_data
            
        except Exception as e:
            logger.error(f"텍스트 색상 선택 중 오류 발생: {str(e)}")
            raise
