import os
import json
import logging
import asyncio
import numpy as np
from typing import Dict, Any, Optional, Tuple
import cv2
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ImageUtils:
    """이미지 처리 관련 유틸리티 클래스"""
    
    def __init__(self, output_dir: str = './output/translated'):
        """
        초기화
        
        Args:
            output_dir: 출력 디렉토리 경로
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 결과 저장을 위한 JSON 파일 경로
        self.results_file = self.output_dir / 'results.json'
        
        logger.info(f"ImageUtils 초기화 완료 - 출력 디렉토리: {self.output_dir}")
    
    async def download_image_from_url(self, image_url: str) -> Optional[np.ndarray]:
        """
        URL에서 이미지 다운로드 및 디코딩
        
        Args:
            image_url: 다운로드할 이미지 URL
            request_id: 요청 ID (로깅용)
            
        Returns:
            이미지 배열 또는 None (실패 시)
        """
        try:
            import aiohttp
            
            # URL이 //로 시작하는 경우 https: 추가
            if image_url.startswith('//'):
                image_url = 'https:' + image_url
            
            logger.debug(f"이미지 다운로드: {image_url}")
            
            # aiohttp로 이미지 다운로드
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    response.raise_for_status()
                    image_bytes = await response.read()
                    
                    # 이미지 디코딩
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image_array is None:
                        raise ValueError("이미지 디코딩 실패")
                    
                    logger.debug(f"이미지 다운로드 및 디코딩 성공: {image_array.shape}")
                    return image_array
                    
        except ImportError:
            logger.error(f"aiohttp가 설치되지 않아 이미지 다운로드 불가")
            return None
        except Exception as e:
            logger.error(f"URL에서 이미지 다운로드 실패: {e}", exc_info=True)
            return None
    
    async def save_image_to_file(self, image_array: np.ndarray, image_id: str) -> Tuple[bool, str]:
        """
        이미지를 로컬 파일로 저장
        
        Args:
            image_array: 저장할 이미지 배열
            image_id: 이미지 ID
            
        Returns:
            (성공 여부, 파일 경로 또는 오류 메시지) 튜플
        """
        try:
            # 파일명 생성 (timestamp 포함)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{image_id}_{timestamp}.jpg"
            file_path = self.output_dir / filename
            
            # 이미지 저장 (BGR 형식 유지)
            success = cv2.imwrite(
                str(file_path),
                image_array,
                [cv2.IMWRITE_JPEG_QUALITY, 100]
            )
            
            if success:
                logger.info(f"이미지 저장 완료: {file_path}")
                return True, str(file_path)
            else:
                error_msg = f"이미지 저장 실패: {file_path}"
                logger.error(error_msg)
                return False, error_msg
                
        except Exception as e:
            error_msg = f"이미지 저장 중 오류: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    async def save_result_to_json(self, image_id: str, file_path: str):
        """
        결과를 JSON 파일에 저장
        
        Args:
            image_id: 이미지 ID
            file_path: 저장된 파일 경로
        """
        try:
            # 기존 결과 로드
            results = []
            if self.results_file.exists():
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            
            # 새 결과 추가
            new_result = {
                "timestamp": datetime.now().isoformat(),
                "image_id": image_id,
                "file_path": file_path,
                "status": "completed"
            }
            results.append(new_result)
            
            # 결과 저장
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            logger.info(f"결과 JSON에 저장됨: {image_id}")
            
        except Exception as e:
            logger.error(f"결과 JSON 저장 중 오류: {str(e)}", exc_info=True) 