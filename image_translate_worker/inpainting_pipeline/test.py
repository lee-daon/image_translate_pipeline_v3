import os
import cv2
from tqdm import tqdm
import glob
from .image_inpainter import ImageInpainter

# --- 테스트 데이터 경로 설정 ---
# 스크립트의 위치를 기준으로 프로젝트 루트 디렉토리를 동적으로 찾습니다.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 테스트에 사용할 이미지와 결과를 저장할 디렉토리 경로
IMAGE_DIR = os.path.join(PROJECT_ROOT, "script/input/LaMa_test_images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/test_results")

def main():
    # --- 설정 ---
    MAX_IMAGES = 50
    BATCH_SIZE = 8
    MAX_WORKERS = 16
    UPSCALE_MODE = 'simple'  # 'simple' 또는 'ai'로 변경하여 테스트 가능

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 이미지 및 마스크 파일 목록 준비
    all_image_paths = sorted([p for p in glob.glob(os.path.join(IMAGE_DIR, '*.png')) if '_mask' not in p])
    
    image_paths = []
    mask_paths = []
    
    for img_path in all_image_paths:
        base, ext = os.path.splitext(img_path)
        mask_path = base + '_mask' + ext
        if os.path.exists(mask_path):
            image_paths.append(img_path)
            mask_paths.append(mask_path)
        if len(image_paths) >= MAX_IMAGES:
            break
            
    print(f"테스트를 위해 총 {len(image_paths)}개의 이미지-마스크 쌍을 찾았습니다.")

    # 2. 이미지와 마스크를 NumPy 배열로 로드
    print("이미지와 마스크를 메모리로 로딩합니다...")
    image_list = [cv2.imread(p) for p in tqdm(image_paths, desc="Loading Images")]
    # 마스크는 흑백(단일 채널)으로 로드
    mask_list = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in tqdm(mask_paths, desc="Loading Masks")]

    # 3. 파이프라인 초기화
    try:
        inpainter = ImageInpainter(max_workers=MAX_WORKERS)
    except ValueError as e:
        print(f"파이프라인 초기화 실패: {e}")
        return

    # 4. 파이프라인 실행 및 결과 저장
    print(f"\n{len(image_list)}개 이미지에 대한 인페인팅 파이프라인을 시작합니다 (배치 크기: {BATCH_SIZE})...")
    
    # process_images는 (원본 인덱스, 결과 이미지) 튜플을 반환
    progress_bar = tqdm(total=len(image_list), desc="Processing and Saving")
    for original_index, result_image in inpainter.process_images(
        image_list, mask_list, batch_size=BATCH_SIZE, upscale_mode=UPSCALE_MODE
    ):
        # 반환된 원본 인덱스를 사용하여 정확한 파일 이름을 찾음
        original_filename = os.path.basename(image_paths[original_index])
        output_path = os.path.join(OUTPUT_DIR, f"result_{original_filename}")
        
        cv2.imwrite(output_path, result_image)
        progress_bar.update(1)
        progress_bar.set_postfix_str(f"Saved to {output_path}")

    progress_bar.close()
    
    # 5. 파이프라인 종료 (스레드 풀 정리)
    inpainter.close()
    print(f"\n모든 작업이 완료되었습니다. 결과는 '{OUTPUT_DIR}' 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()
