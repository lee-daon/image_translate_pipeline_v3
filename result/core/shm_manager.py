from multiprocessing import shared_memory
import numpy as np
import logging
import uuid
from typing import Tuple, Dict, Any

from .config import SHM_NAME_PREFIX

logger = logging.getLogger(__name__)

# 생성된 공유 메모리 블록 추적 (클린업 목적)
# 실제 프로덕션에서는 더 견고한 관리 메커니즘 필요 (예: Redis)
_managed_shms = set()

def create_shm_from_array(img_array: np.ndarray) -> Dict[str, Any]:
    """NumPy 배열을 공유 메모리에 쓰고, 접근 정보를 반환합니다."""
    try:
        # 고유한 공유 메모리 이름 생성
        shm_name = f"{SHM_NAME_PREFIX}{uuid.uuid4().hex}"
        size = img_array.nbytes
        shape = img_array.shape
        dtype = img_array.dtype

        # 공유 메모리 블록 생성
        shm = shared_memory.SharedMemory(name=shm_name, create=True, size=size)
        logger.debug(f"Created shared memory: {shm_name} with size {size} bytes")

        # 공유 메모리 블록에 매핑된 NumPy 배열 생성
        shm_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        # 원본 배열 데이터를 공유 메모리 배열에 복사
        shm_array[:] = img_array[:]

        # 관리 목록에 추가
        _managed_shms.add(shm_name)

        return {
            "shm_name": shm_name,
            "shape": shape,
            "dtype": str(dtype), # Redis 저장 위해 문자열 변환
            "size": size
        }
    except Exception as e:
        logger.error(f"Error creating shared memory: {e}", exc_info=True)
        raise

def cleanup_shm(shm_name: str):
    """지정된 이름의 공유 메모리 블록을 해제(unlink)합니다."""
    try:
        # 임시 SharedMemory 객체를 생성하여 unlink 호출 (존재하지 않으면 에러 발생)
        temp_shm = shared_memory.SharedMemory(name=shm_name, create=False)
        temp_shm.close() # 먼저 close 호출 (참조 카운트 감소)
        temp_shm.unlink() # 시스템에서 블록 제거
        logger.debug(f"Successfully unlinked shared memory: {shm_name}")
        if shm_name in _managed_shms:
            _managed_shms.remove(shm_name)
    except FileNotFoundError:
        logger.warning(f"Shared memory {shm_name} not found for cleanup, might have been already unlinked.")
    except Exception as e:
        logger.error(f"Error cleaning up shared memory {shm_name}: {e}", exc_info=True)

def cleanup_all_managed_shms():
    """관리 중인 모든 공유 메모리 블록을 정리합니다 (애플리케이션 종료 시)."""
    logger.info(f"Cleaning up all managed shared memory blocks ({len(_managed_shms)})... ")
    # set 복사 후 순회 (반복 중 제거 문제 방지)
    shm_names_to_clean = list(_managed_shms)
    for shm_name in shm_names_to_clean:
        cleanup_shm(shm_name)
    logger.info("Shared memory cleanup finished.")

# 워커에서 사용할 함수 (참고용)
def get_array_from_shm(shm_info: Dict[str, Any]) -> np.ndarray:
    """공유 메모리 정보로 NumPy 배열을 재구성하여 반환합니다."""
    shm_name = shm_info['shm_name']
    shape = tuple(shm_info['shape']) # 리스트를 튜플로 변환
    dtype = np.dtype(shm_info['dtype']) # 문자열을 dtype 객체로 변환
    try:
        # 기존 공유 메모리 블록에 연결
        existing_shm = shared_memory.SharedMemory(name=shm_name, create=False)
        # 공유 메모리 버퍼를 사용하여 NumPy 배열 생성 (읽기 전용으로도 가능)
        img_array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

        # 중요: 워커는 사용 후 반드시 existing_shm.close() 를 호출해야 함
        #       unlink()는 생성한 프로세스(API 서버) 또는 별도 관리자가 담당
        #       여기서는 배열만 반환하고 close는 워커의 책임으로 남겨둠

        return img_array, existing_shm # 배열과 shm 객체 함께 반환 (워커에서 close/unlink 관리 용이)

    except FileNotFoundError:
        logger.error(f"Shared memory {shm_name} not found.")
        raise
    except Exception as e:
        logger.error(f"Error accessing shared memory {shm_name}: {e}", exc_info=True)
        raise
