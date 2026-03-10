import os
import shutil
import random
import csv

# 허용할 이미지 확장자
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


def _collect_classes(src_dir: str):
    """src_dir 바로 아래의 클래스 폴더 목록을 정렬하여 반환."""
    return sorted(
        [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    )



def _collect_images(class_dir: str):
    """클래스 폴더 바로 아래의 이미지 파일 목록을 정렬하여 반환."""
    return sorted(
        [f for f in os.listdir(class_dir) if f.lower().endswith(VALID_EXTENSIONS)]
    )



def _validate_inputs(src_dir: str, test_count: int):
    """입력 경로 및 파라미터 유효성 검증."""
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"[Error] Source directory not found: {src_dir}")

    if not isinstance(test_count, int):
        raise TypeError(f"[Error] test_count must be int, got {type(test_count).__name__}")

    if test_count < 1:
        raise ValueError(f"[Error] test_count must be >= 1, got {test_count}")

    classes = _collect_classes(src_dir)
    if not classes:
        raise ValueError(f"[Error] No class folders found in source directory: {src_dir}")

    insufficient_classes = []
    empty_classes = []

    for cls in classes:
        cls_dir = os.path.join(src_dir, cls)
        images = _collect_images(cls_dir)

        if len(images) == 0:
            empty_classes.append(cls)
            continue

        # test 추출 후 최소 1장은 train에 남겨두는 정책
        if len(images) <= test_count:
            insufficient_classes.append((cls, len(images)))

    if empty_classes:
        raise ValueError(
            "[Error] The following classes do not contain any valid images: "
            + ", ".join(empty_classes)
        )

    if insufficient_classes:
        details = "\n".join(
            [f" - {cls}: {count} images (requires at least {test_count + 1})"
             for cls, count in insufficient_classes]
        )
        raise ValueError(
            "[Error] Some classes do not have enough images for the requested split.\n"
            f"Requested test_count per class: {test_count}\n"
            "Each class must contain at least test_count + 1 images so that train data remains.\n"
            f"{details}"
        )

    return classes



def _make_unique_test_name(rng: random.Random, used_names: set, ext: str) -> str:
    """seed 기반으로 재현 가능한 익명 파일명 생성."""
    while True:
        new_name = f"{rng.randint(100000, 999999)}_{rng.randint(1000, 9999)}{ext.lower()}"
        if new_name not in used_names:
            used_names.add(new_name)
            return new_name



def split_and_anonymize(src_dir, dest_root='.', test_count=10, seed=42):
    """
    원본 데이터를 읽어 클래스별로 정확히 test_count장씩 무작위 추출하여
    Train셋과 Test셋(익명화)으로 분리합니다.

    Parameters
    ----------
    src_dir : str
        입력 데이터셋 루트 경로. 구조는 src_dir/class_x/image.jpg 형태를 가정합니다.
    dest_root : str, default='.'
        출력 루트 경로.
    test_count : int, default=10
        각 클래스에서 테스트셋으로 추출할 이미지 수.
        각 클래스는 최소 test_count + 1장의 이미지를 가져야 합니다.
    seed : int, default=42
        재현 가능한 랜덤 추출을 위한 시드.
    """
    rng = random.Random(seed)

    classes = _validate_inputs(src_dir, test_count)

    # 경로 설정
    test_dir = os.path.join(dest_root, 'test-dataset')
    train_dir = os.path.join(dest_root, 'dataset')
    label_file = os.path.join(dest_root, 'test_labels.csv')

    # 초기화 (기존 폴더 삭제)
    for d in [test_dir, train_dir]:
        if os.path.exists(d):
            print(f"[Warning] Deleting existing folder: {d}")
            shutil.rmtree(d)
        os.makedirs(d)

    test_map = []  # (anonymized_filename, label, original_filename, original_path)
    used_test_names = set()

    print(
        f"[Info] Processing split... "
        f"(Classes: {len(classes)}, test_count per class: {test_count}, seed: {seed})"
    )

    for cls in classes:
        cls_src_path = os.path.join(src_dir, cls)
        cls_train_dest = os.path.join(train_dir, cls)
        os.makedirs(cls_train_dest, exist_ok=True)

        # 이미지 파일 수집 및 섞기
        images = _collect_images(cls_src_path)
        rng.shuffle(images)

        test_imgs = images[:test_count]
        train_imgs = images[test_count:]

        print(
            f"[Class] {cls} | total={len(images)} | test={len(test_imgs)} | train={len(train_imgs)}"
        )

        # 1. Test set 처리 (익명화 + 정답 기록)
        for img in test_imgs:
            src_f = os.path.join(cls_src_path, img)
            ext = os.path.splitext(img)[1]
            new_name = _make_unique_test_name(rng, used_test_names, ext)

            dst_f = os.path.join(test_dir, new_name)
            shutil.copy2(src_f, dst_f)

            # 정답 기록 (익명파일명, 실제클래스, 원본파일명, 원본경로)
            test_map.append([
                new_name,
                cls,
                img,
                os.path.abspath(src_f),
            ])

        # 2. Train set 처리 (폴더 구조 유지)
        for img in train_imgs:
            src_f = os.path.join(cls_src_path, img)
            dst_f = os.path.join(cls_train_dest, img)
            shutil.copy2(src_f, dst_f)

    # 정답지 CSV 저장
    with open(label_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label', 'original_filename', 'original_path'])
        writer.writerows(test_map)

    print(f"[Done] Train data: '{train_dir}'")
    print(f"[Done] Test data: '{test_dir}' (Labels saved to '{label_file}')")


if __name__ == "__main__":
    # 사용 시 원본 폴더 경로 및 test_count 수정 필요
    split_and_anonymize('./data-preprocessed', test_count=100, seed=42)
