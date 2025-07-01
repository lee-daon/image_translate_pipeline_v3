import os
import sys
import json
import logging
import asyncio
import time
from typing import List

import aiohttp
import numpy as np

# 프로젝트 루트 설정
WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(WORKER_DIR))
sys.path.insert(0, ROOT_DIR)

from core.config import (
    TRANSLATE_TEXT_RESULT_HASH_PREFIX, HOSTING_TASKS_QUEUE, SUCCESS_QUEUE, ERROR_QUEUE,
    GEMINI_API_KEY, GEMINI_MODEL_NAME, TRANSLATION_RPS
)
from core.redis_client import get_redis_client, enqueue_error_result, enqueue_success_result
from hosting.r2hosting import R2ImageHosting
from dispatching_pipeline.mask import filter_chinese_ocr_result

logger = logging.getLogger(__name__)



# API 키 검증
if not GEMINI_API_KEY:
    logger.critical("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
    raise ValueError("API 키가 필요합니다.")

# Gemini API 엔드포인트
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

# Rate limit 관리용 변수
_last_request_time = 0
_rate_limit_lock = asyncio.Lock()

# JSON 응답 스키마 정의 (문자열 배열)
TRANSLATION_LIST_SCHEMA = {
    "type": "ARRAY",
    "description": "입력된 텍스트 배열에 대한 번역된 텍스트 문자열 배열. 순서는 원본 배열과 동일해야 합니다.",
    "items": {
        "type": "STRING"
    }
}

# R2 호스팅 인스턴스 생성
r2_hosting = R2ImageHosting()

async def wait_for_rate_limit(request_id: str = "N/A"):
    """
    초당 요청 수 제한에 따라 API 호출을 제어합니다.
    """
    async with _rate_limit_lock:
        global _last_request_time
        current_time = time.time()
        
        # 마지막 요청 이후 경과 시간 계산
        time_since_last = current_time - _last_request_time
        min_interval = 1.0 / TRANSLATION_RPS  # 요청 간 최소 간격
        
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            logger.info(f"[{request_id}] Rate limit: waiting {wait_time:.2f}s (RPS: {TRANSLATION_RPS})")
            await asyncio.sleep(wait_time)
        
        _last_request_time = time.time()
        logger.debug(f"[{request_id}] Rate limit check passed (RPS: {TRANSLATION_RPS})")

async def call_gemini_translate_list(texts_to_translate: List[str], request_id: str = "N/A") -> List[str]:
    """
    Gemini API를 JSON 모드로 호출하여 텍스트 리스트 전체를 번역합니다.

    Args:
        texts_to_translate: 번역할 텍스트 문자열 리스트.
        request_id: 로깅을 위한 요청 ID.

    Returns:
        번역된 텍스트 문자열 리스트.

    Raises:
        aiohttp.ClientError: 네트워크 또는 API 호출 중 에러 발생 시.
        ValueError: API 응답 구문 분석 실패, 스키마 불일치, 길이 불일치 또는 API 에러 메시지 포함 시.
        json.JSONDecodeError: 입력 리스트를 JSON으로 변환 실패 시.
    """
    if not texts_to_translate:
        logger.info(f"[{request_id}] 번역할 텍스트 리스트가 비어있습니다. 빈 리스트 반환.")
        return []

    # 시스템 지침 정의
    system_instruction = (
        "You are a helpful translation assistant for e-commerce. "
        "Translate the texts from product detail images into Korean. "
        "The translation should be natural, polite, and concise, suitable for marketing content."
    )

    # 입력 텍스트 리스트를 JSON 문자열로 변환
    try:
        prompt_content = json.dumps(texts_to_translate, ensure_ascii=False)
    except json.JSONDecodeError as e:
        logger.error(f"[{request_id}] 입력 텍스트 리스트를 JSON으로 변환 실패: {e}")
        raise e

    headers = {
        'Content-Type': 'application/json'
    }
    request_data = {
        "system_instruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt_content}]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": TRANSLATION_LIST_SCHEMA
        }
    }

    logger.debug(f"[{request_id}] Calling Gemini API (JSON List). URL: {API_URL.split('?')[0]}, Num Texts: {len(texts_to_translate)}, Prompt (partial): {prompt_content[:100]}...")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(API_URL, headers=headers, json=request_data) as response:
                response_text = await response.text()
                response.raise_for_status()

                try:
                    response_data = json.loads(response_text)
                except json.JSONDecodeError:
                    logger.error(f"[{request_id}] Gemini API JSON 응답 파싱 실패 (List). 응답: {response_text}")
                    raise ValueError("Failed to parse Gemini API JSON response (List)")

                # JSON 모드 응답 구조 파싱
                if response_data.get("candidates") and \
                   len(response_data["candidates"]) > 0 and \
                   response_data["candidates"][0].get("content") and \
                   response_data["candidates"][0]["content"].get("parts") and \
                   len(response_data["candidates"][0]["content"]["parts"]) > 0 and \
                   isinstance(response_data["candidates"][0]["content"]["parts"][0].get("text"), str):

                    translated_list_json = response_data["candidates"][0]["content"]["parts"][0]["text"]
                    try:
                        translated_list = json.loads(translated_list_json)
                    except json.JSONDecodeError:
                        logger.error(f"[{request_id}] Gemini API 반환 JSON 내부 파싱 실패 (List). 내부 JSON: {translated_list_json}")
                        raise ValueError("Failed to parse inner JSON from Gemini API response (List)")

                    # 반환된 것이 리스트인지 확인
                    if not isinstance(translated_list, list):
                        logger.error(f"[{request_id}] Gemini API가 JSON 배열을 반환하지 않음 (List). 반환값 타입: {type(translated_list)}")
                        raise ValueError("Gemini API did not return a JSON array as expected (List)")

                    # 입력과 출력 리스트 길이 비교
                    if len(translated_list) != len(texts_to_translate):
                         logger.error(f"[{request_id}] Gemini API 번역 결과 길이 불일치 (List). 입력: {len(texts_to_translate)}, 출력: {len(translated_list)}")
                         raise ValueError(f"Length mismatch between input ({len(texts_to_translate)}) and translated output ({len(translated_list)}) (List)")

                    logger.debug(f"[{request_id}] Gemini API 응답 수신 (List). 번역된 항목 수: {len(translated_list)}")
                    return translated_list
                else:
                    logger.error(f"[{request_id}] Gemini API 응답 구조가 예상과 다릅니다 (List): {response_data}")
                    if response_data.get("promptFeedback") and response_data["promptFeedback"].get("blockReason"):
                        reason = response_data['promptFeedback']['blockReason']
                        ratings = response_data['promptFeedback'].get('safetyRatings', 'N/A')
                        logger.error(f"[{request_id}] Gemini API 요청 차단됨 (List): {reason}, 이유: {ratings}")
                        raise ValueError(f"Gemini API request blocked (List): {reason}")
                    raise ValueError("Unexpected Gemini API response structure (List)")

        except aiohttp.ClientResponseError as e:
            logger.error(f"[{request_id}] Gemini API HTTP 에러 (List): {e.status} {e.message}. 응답: {response_text}")
            try:
                error_details = json.loads(response_text)
                if error_details.get("error") and error_details["error"].get("message"):
                    logger.error(f"[{request_id}] 상세 Gemini API 에러 (List): {error_details['error']['message']}")
                    raise ValueError(f"Gemini API Error (List): {error_details['error']['message']}") from e
            except json.JSONDecodeError:
                pass
            raise e
        except aiohttp.ClientError as e:
            logger.error(f"[{request_id}] Gemini API 호출 중 네트워크/클라이언트 에러 (List): {e}")
            raise e
        except Exception as e:
             logger.error(f"[{request_id}] Gemini API 호출 중 예상치 못한 에러 (List): {e}", exc_info=True)
             raise e

async def call_translation_api(texts: List[str], request_id: str) -> List[str]:
    """
    Gemini API (JSON 모드)를 호출하여 텍스트 리스트 전체를 한 번에 번역합니다.
    API 호출 실패 시 한 번 재시도하며, 최종 실패 시 빈 리스트를 반환합니다.

    Args:
        texts: 번역할 텍스트 문자열 리스트.
        request_id: 로깅을 위한 요청 ID.

    Returns:
        번역된 텍스트 문자열 리스트. 최종 실패 시 빈 리스트 `[]` 반환.
    """
    if not texts:
        logger.info(f"[{request_id}] No texts provided for translation.")
        return []

    logger.info(f"[{request_id}] Calling Gemini translation API (JSON List) for {len(texts)} texts...")

    max_retries = 1
    for attempt in range(max_retries + 1):
        try:
            # Rate limit 적용
            await wait_for_rate_limit(request_id)
            
            translated_texts = await call_gemini_translate_list(
                texts_to_translate=texts,
                request_id=request_id
            )
            logger.info(f"[{request_id}] Gemini translation (JSON List) finished on attempt {attempt + 1}. Processed {len(texts)} texts, received {len(translated_texts)} translations.")
            return translated_texts

        except Exception as e:
            logger.error(f"[{request_id}] Attempt {attempt + 1} failed to translate text list using Gemini API (JSON List): {e}", exc_info=False)
            if attempt < max_retries:
                logger.info(f"[{request_id}] Retrying translation...")
                await asyncio.sleep(1)
            else:
                logger.error(f"[{request_id}] All translation attempts failed.")
                return []

    return []

async def save_result_to_internal_storage(result_checker, request_id: str, result_data: dict):
    """번역 결과를 ResultChecker의 내부 메모리 저장소에 저장합니다."""
    try:
        await result_checker.save_translation_result(request_id, result_data)
        logger.debug(f"[{request_id}] Translation result saved to internal storage")
        return True
    except Exception as e:
        logger.error(f"[{request_id}] Failed to save result to internal storage: {e}", exc_info=True)
        return False

async def process_and_save_translation(task_data: dict, image_url: str, result_checker):
    """
    번역의 전체 과정을 처리하고, 결과를 ResultChecker의 내부 저장소에 저장합니다.
    result_checker: ResultChecker 인스턴스 (내부 저장소 접근용)
    """
    request_id = task_data.get("request_id")
    image_id = task_data.get("image_id")
    ocr_result = task_data.get("ocr_result")
    
    try:
        # 이미 필터링된 결과가 있으면 사용, 없으면 필터링 수행
        filtered_ocr_result = task_data.get("filtered_ocr_result")
        if filtered_ocr_result is None:
            # 중국어 필터링 (CPU 작업이지만 가벼움)
            filtered_ocr_result = filter_chinese_ocr_result(ocr_result or [], request_id)

        if filtered_ocr_result:
            texts_to_translate = []
            original_items_for_rendering = []

            for item in filtered_ocr_result:
                try:
                    if isinstance(item, list) and len(item) == 2 and \
                       isinstance(item[0], list) and isinstance(item[1], list) and len(item[1]) >= 1:
                        box_info = item[0]
                        text_score_pair = item[1]
                        text = text_score_pair[0]

                        if isinstance(text, str) and text.strip():
                            texts_to_translate.append(text)
                            original_items_for_rendering.append({
                                "box": box_info,
                                "original_text": text
                            })
                except Exception as e:
                    logger.warning(f"[{request_id}] Error extracting translation data: {e}")

            if texts_to_translate:
                # 번역 API 호출 (순수 async I/O)
                logger.info(f"[{request_id}] Calling translation API for {len(texts_to_translate)} texts")
                translated_texts = await call_translation_api(texts_to_translate, request_id)

                # 번역 결과 처리
                translate_result_for_rendering = []
                
                if texts_to_translate and not translated_texts:
                    logger.error(f"[{request_id}] Translation API call failed or returned empty list.")
                    # API 호출이 실패했으므로 에러 큐에 넣고 이 작업은 중단
                    await enqueue_error_result(request_id, image_id, "Translation API call failed")
                    return

                if len(translated_texts) != len(original_items_for_rendering):
                    logger.error(f"[{request_id}] Translation length mismatch")
                    translate_result_for_rendering = None
                else:
                    for original_info, translated_text in zip(original_items_for_rendering, translated_texts):
                        translate_result_for_rendering.append({
                            "box": original_info["box"],
                            "translated_text": translated_text,
                            "original_char_count": len(original_info["original_text"])
                        })

                # 결과 저장 (메인 루프에서 async Redis 호출)
                if translate_result_for_rendering is not None:
                    rendering_data = {
                        "image_id": image_id,
                        "image_url": image_url,  # 원본 URL 그대로 사용
                        "translate_result": translate_result_for_rendering
                    }
                    await save_result_to_internal_storage(result_checker, request_id, rendering_data)
                    logger.debug(f"[{request_id}] Translation saved to internal storage")
                    # 내부 저장소의 save_translation_result가 자동으로 렌더링 확인 및 트리거
            else:
                # 번역할 텍스트가 없는 경우 - 성공 큐로 바로 전송
                await enqueue_success_result(request_id, image_id, image_url)
                logger.info(f"[{request_id}] No texts to translate, forwarded to success queue")
        else:
            # 필터링된 결과가 없는 경우 - 성공 큐로 바로 전송
            await enqueue_success_result(request_id, image_id, image_url)
            logger.info(f"[{request_id}] No Chinese text found, forwarded to success queue")
            
    except Exception as e:
        logger.error(f"[{request_id}] Error in translation process: {e}", exc_info=True)
        # 번역 과정 전반의 에러 시 에러 큐로 전송
        await enqueue_error_result(request_id, image_id, f"Translation process error: {str(e)}") 