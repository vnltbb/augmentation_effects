import os
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def preprocess_images(data_dir, input_shape, image_exts):
    """
    원본 데이터셋의 모든 이미지를 지정된 크기로 리사이징하여
    새로운 경로에 저장합니다.
    """
    # 설정값 불러오기
    source_dir = Path(data_dir)
    target_dir_name = source_dir.name + "-preprocessed"
    target_dir = source_dir.parent / target_dir_name
    target_size = (input_shape[1], input_shape[0]) # (width, height)
    image_exts = image_exts
    
    print("-" * 50)
    print(f"🖼️  데이터셋 전처리를 시작합니다.")
    print(f"원본 경로: {source_dir}")
    print(f"저장 경로: {target_dir}")
    print(f"리사이즈 크기: {target_size}")
    print("-" * 50)

    # 처리할 모든 이미지 파일 검색
    all_images = []
    for ext in image_exts:
        all_images.extend(source_dir.rglob(f"*{ext}"))
        # 대문자 확장자도 포함
        all_images.extend(source_dir.rglob(f"*{ext.upper()}"))

    # 중복 제거
    all_images = sorted(list(set(all_images)))
    
    if not all_images:
        print("오류: 원본 경로에서 이미지를 찾을 수 없습니다.")
        return

    # tqdm으로 진행률 표시
    for source_path in tqdm(all_images, desc="이미지 전처리 중"):
        try:
            # 원본 경로 구조를 유지하여 저장 경로 생성
            relative_path = source_path.relative_to(source_dir)
            target_path = target_dir / relative_path
            
            # 저장할 폴더가 없으면 생성
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 이미지 열기, RGB 변환, 리사이징 후 저장
            with Image.open(source_path) as img:
                img_rgb = img.convert("RGB")
                img_resized = img_rgb.resize(target_size)
                img_resized.save(target_path)
        except Exception as e:
            print(f"\n파일 처리 중 오류 발생: {source_path}, 오류: {e}")

    print("\n🎉 모든 이미지 전처리가 완료되었습니다!")

if __name__ == "__main__":

    # DecompressionBomb 경고 방지
    Image.MAX_IMAGE_PIXELS = None
    
    preprocess_images(
        data_dir="./data", 
        input_shape=(224, 224), 
        image_exts=[".jpg", ".png"]
        )