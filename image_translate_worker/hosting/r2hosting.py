import os
import io
import boto3
import logging
import numpy as np
import cv2
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class R2ImageHosting:
    """R2를 사용한 이미지 호스팅 클래스"""
    
    def __init__(self):
        """R2 클라이언트 초기화"""
        env_vars = {
            "R2_ENDPOINT": "endpoint_url",
            "CLOUDFLARE_ACCESS_KEY_ID": "access_key_id",
            "CLOUDFLARE_SECRET_KEY": "secret_access_key",
            "R2_BUCKET_NAME": "bucket_name",
            "R2_DOMAIN": "public_url_base",
        }

        missing_vars = []
        for env_key, attr_name in env_vars.items():
            value = os.getenv(env_key)
            if not value:
                missing_vars.append(env_key)
            setattr(self, attr_name, value)

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # S3 호환 클라이언트 생성
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name='auto'  # R2는 'auto' 사용
            )
            logger.info("R2 클라이언트 초기화 완료")
        except Exception as e:
            logger.error(f"R2 클라이언트 초기화 실패: {e}")
            raise
    
    def upload_image_from_array(self, 
                               image_array: np.ndarray, 
                               image_id: str, 
                               sub_path: str = 'translated',
                               file_ext: str = '.jpg',
                               quality: int = 90,
                               metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        NumPy 배열을 이미지로 변환하여 R2에 업로드
        
        Args:
            image_array: 업로드할 이미지 배열
            image_id: 이미지 ID (파일명으로 사용)
            sub_path: 버킷 내 하위 경로
            file_ext: 파일 확장자 (.jpg, .png 등)
            quality: JPEG 품질 (1-100)
            metadata: 추가 메타데이터
            
        Returns:
            업로드 결과 딕셔너리 (success, url, error)
        """
        try:
            # 이미지 인코딩
            ext = file_ext.lower()
            if ext in ['.jpg', '.jpeg']:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                success, encoded_img = cv2.imencode('.jpg', image_array, encode_param)
                content_type = 'image/jpeg'
            elif ext == '.png':
                success, encoded_img = cv2.imencode('.png', image_array)
                content_type = 'image/png'
            else:
                return {
                    "success": False,
                    "error": f"지원하지 않는 파일 형식: {file_ext}"
                }
            
            if not success:
                return {
                    "success": False,
                    "error": "이미지 인코딩 실패"
                }
            
            # 바이트 스트림 생성
            image_bytes = io.BytesIO(encoded_img.tobytes())
            
            # S3 키 생성
            s3_key = f"{sub_path}/{image_id}{file_ext}"
            
            # R2에 업로드
            self.s3_client.upload_fileobj(
                image_bytes,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'Metadata': metadata or {},
                    'ContentType': content_type,
                    'CacheControl': 'public, max-age=31536000, immutable'
                }
            )
            
            # 공개 URL 생성
            public_url = f"{self.public_url_base.rstrip('/')}/{s3_key}"
            
            logger.info(f"이미지 업로드 성공: {image_id} -> {public_url}")
            
            return {
                "success": True,
                "url": public_url,
                "s3_key": s3_key,
                "bucket": self.bucket_name
            }
            
        except ClientError as e:
            error_msg = f"R2 업로드 실패 (ClientError): {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"이미지 업로드 중 오류: {e}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg
            }
    
    def delete_image(self, s3_key: str) -> bool:
        """
        R2에서 이미지 삭제
        
        Args:
            s3_key: 삭제할 이미지의 S3 키
            
        Returns:
            삭제 성공 여부
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"이미지 삭제 성공: {s3_key}")
            return True
        except ClientError as e:
            logger.error(f"R2 삭제 실패: {e}")
            return False
        except Exception as e:
            logger.error(f"이미지 삭제 중 오류: {e}", exc_info=True)
            return False 