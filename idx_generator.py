#!/usr/bin/env python3

import argparse
from pathlib import Path
import torch

IMG_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def is_image(p):
    return p.suffix.lower() in IMG_EXT


def build_index(dataset_root, n_per_class, out_dir, seed):

    dataset_root = Path(dataset_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    class_names = [p.name for p in class_dirs]

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    g = torch.Generator().manual_seed(seed)

    samples = []

    for class_dir in class_dirs:

        imgs = [p for p in class_dir.rglob("*") if p.is_file() and is_image(p)]

        if len(imgs) < n_per_class:
            raise ValueError(
                f"{class_dir.name} has only {len(imgs)} images (requested {n_per_class})"
            )

        perm = torch.randperm(len(imgs), generator=g)[:n_per_class]

        for i in perm:
            samples.append((str(imgs[i]), class_to_idx[class_dir.name]))

    # 전체 셔플
    perm = torch.randperm(len(samples), generator=g)
    samples = [samples[i] for i in perm]

    payload = {
        "samples": samples,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "seed": seed,
        "n_per_class": n_per_class,
    }

    out_path = out_dir / f"{seed}_{n_per_class}.pt"
    torch.save(payload, out_path)

    print("saved:", out_path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-per-class", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    build_index(
        args.dataset_root,
        args.n_per_class,
        args.out_dir,
        args.seed,
    )


if __name__ == "__main__":
    main()