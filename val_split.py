import os
import shutil
import random

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


def _collect_classes(src_dir: str):
    return sorted(
        [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    )


def _collect_images(class_dir: str):
    return sorted(
        [f for f in os.listdir(class_dir) if f.lower().endswith(VALID_EXTENSIONS)]
    )


def _validate_inputs(src_dir: str, val_count: int):
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"[Error] Source directory not found: {src_dir}")

    if not isinstance(val_count, int):
        raise TypeError(f"[Error] val_count must be int")

    if val_count < 1:
        raise ValueError(f"[Error] val_count must be >=1")

    classes = _collect_classes(src_dir)

    for cls in classes:
        cls_dir = os.path.join(src_dir, cls)
        images = _collect_images(cls_dir)

        if len(images) <= val_count:
            raise ValueError(
                f"[Error] Class '{cls}' has only {len(images)} images "
                f"(needs at least {val_count + 1})"
            )

    return classes


def split_val_dataset(src_dir, dest_root='.', val_count=50, seed=42):

    rng = random.Random(seed)
    classes = _validate_inputs(src_dir, val_count)

    train_dir = os.path.join(dest_root, 'dataset')
    val_dir = os.path.join(dest_root, 'val-dataset')

    if os.path.exists(train_dir):
        print(f"[Warning] Removing existing {train_dir}")
        shutil.rmtree(train_dir)

    if os.path.exists(val_dir):
        print(f"[Warning] Removing existing {val_dir}")
        shutil.rmtree(val_dir)

    os.makedirs(train_dir)
    os.makedirs(val_dir)

    print(
        f"[Info] Splitting validation set "
        f"(val_count per class: {val_count}, seed: {seed})"
    )

    for cls in classes:

        src_cls = os.path.join(src_dir, cls)
        train_cls = os.path.join(train_dir, cls)
        val_cls = os.path.join(val_dir, cls)

        os.makedirs(train_cls, exist_ok=True)
        os.makedirs(val_cls, exist_ok=True)

        images = _collect_images(src_cls)
        rng.shuffle(images)

        val_imgs = images[:val_count]
        train_imgs = images[val_count:]

        print(
            f"[Class] {cls} | total={len(images)} | val={len(val_imgs)} | train={len(train_imgs)}"
        )

        for img in val_imgs:
            shutil.copy2(
                os.path.join(src_cls, img),
                os.path.join(val_cls, img)
            )

        for img in train_imgs:
            shutil.copy2(
                os.path.join(src_cls, img),
                os.path.join(train_cls, img)
            )

    print(f"[Done] Train dataset → {train_dir}")
    print(f"[Done] Validation dataset → {val_dir}")


if __name__ == "__main__":

    split_val_dataset(
        src_dir="./data-preprocessed",
        val_count=96,
        seed=42
    )