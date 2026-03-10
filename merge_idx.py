# merge_index_pt.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch


IndexDict = Dict[str, Any]
Sample = Tuple[str, int]  # (image_path, class_idx)


# =========================
# User variables (as requested)
# =========================
INDEX1_PATH = "./data-idx/42_500.pt"
INDEX2_PATH = "./data-idx/77_500.pt"
OUT_PATH = "./data-idx/merged_test3.pt"  # <-- 원하는 출력 파일명으로 변경


def _load_index(pt_path: str) -> IndexDict:
    obj = torch.load(pt_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"[{pt_path}] expected dict, got {type(obj)}")

    required = ["samples", "class_to_idx", "idx_to_class"]
    missing = [k for k in required if k not in obj]
    if missing:
        raise KeyError(f"[{pt_path}] missing keys: {missing}")

    if not isinstance(obj["samples"], (list, tuple)):
        raise TypeError(f"[{pt_path}] 'samples' must be list/tuple")

    # Normalize idx_to_class keys (can be int or str depending on how it was saved)
    idx_to_class = obj["idx_to_class"]
    if isinstance(idx_to_class, dict):
        norm = {}
        for k, v in idx_to_class.items():
            try:
                kk = int(k)
            except Exception as e:
                raise TypeError(f"[{pt_path}] idx_to_class has non-int key: {k}") from e
            norm[kk] = v
        obj["idx_to_class"] = norm
    else:
        raise TypeError(f"[{pt_path}] 'idx_to_class' must be dict")

    class_to_idx = obj["class_to_idx"]
    if not isinstance(class_to_idx, dict):
        raise TypeError(f"[{pt_path}] 'class_to_idx' must be dict")

    return obj


def _build_samples_by_classname(index_obj: IndexDict) -> Dict[str, List[str]]:
    """
    Return: {class_name: [image_path, ...]}
    """
    idx_to_class: Dict[int, str] = index_obj["idx_to_class"]
    out: Dict[str, List[str]] = {}

    for item in index_obj["samples"]:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise ValueError(f"Invalid sample entry: {item} (expected (path, class_idx))")

        img_path, class_idx = item
        if not isinstance(img_path, str):
            raise TypeError(f"Invalid img_path type: {type(img_path)} in {item}")

        try:
            class_idx_int = int(class_idx)
        except Exception as e:
            raise TypeError(f"Invalid class_idx (not int-castable): {class_idx} in {item}") from e

        if class_idx_int not in idx_to_class:
            raise KeyError(f"class_idx {class_idx_int} not found in idx_to_class")

        cname = idx_to_class[class_idx_int]
        out.setdefault(cname, []).append(img_path)

    return out


def _make_canonical_mappings(class_names: List[str]) -> tuple[Dict[str, int], Dict[int, str]]:
    """
    Canonicalize class indices by sorted class name.
    """
    class_names_sorted = sorted(set(class_names))
    class_to_idx = {c: i for i, c in enumerate(class_names_sorted)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    return class_to_idx, idx_to_class


def merge_index_pts(
    index1_path: str,
    index2_path: str,
    out_path: str,
    *,
    dedup_by_path: bool = True,
    sort_paths_within_class: bool = False,
) -> None:
    """
    Merge two index .pt files into one .pt, grouping by the same class names.
    - dedup_by_path: 동일 경로 문자열 중복 제거 (True 권장)
    - sort_paths_within_class: 클래스 내 path 정렬 (재현성 목적)
    """
    idx1 = _load_index(index1_path)
    idx2 = _load_index(index2_path)

    by_class_1 = _build_samples_by_classname(idx1)
    by_class_2 = _build_samples_by_classname(idx2)

    classes_1 = set(by_class_1.keys())
    classes_2 = set(by_class_2.keys())

    only_1 = sorted(classes_1 - classes_2)
    only_2 = sorted(classes_2 - classes_1)

    if only_1 or only_2:
        # "예정"이라 했지만 실제로 다르면 여기서 바로 알려주는 게 맞음
        msg = []
        if only_1:
            msg.append(f"Only in index1: {only_1}")
        if only_2:
            msg.append(f"Only in index2: {only_2}")
        raise ValueError("Class name mismatch between pt files.\n" + "\n".join(msg))

    all_classes = sorted(classes_1)

    # Canonical mapping (stable)
    class_to_idx_out, idx_to_class_out = _make_canonical_mappings(all_classes)

    merged_samples: List[Sample] = []

    for cname in all_classes:
        paths = []
        paths.extend(by_class_1.get(cname, []))
        paths.extend(by_class_2.get(cname, []))

        if dedup_by_path:
            # preserve order (py3.7+)
            paths = list(dict.fromkeys(paths))

        if sort_paths_within_class:
            paths = sorted(paths)

        cidx = class_to_idx_out[cname]
        merged_samples.extend([(p, cidx) for p in paths])

    out_obj: IndexDict = {
        "samples": merged_samples,
        "class_to_idx": class_to_idx_out,
        "idx_to_class": idx_to_class_out,
        # seed / n_per_class는 "두 파일을 합쳤다"는 의미로 보수적으로 처리
        # - seed: 둘이 같으면 유지, 다르면 None
        # - n_per_class: 둘이 같으면 유지, 다르면 None
        "seed": idx1.get("seed") if idx1.get("seed") == idx2.get("seed") else None,
        "n_per_class": idx1.get("n_per_class") if idx1.get("n_per_class") == idx2.get("n_per_class") else None,
        # 참고용 메타
        "sources": [str(index1_path), str(index2_path)],
        "n_samples": len(merged_samples),
    }

    out_dir = Path(out_path).parent
    if str(out_dir) != "":
        os.makedirs(out_dir, exist_ok=True)

    torch.save(out_obj, out_path)

    # Minimal console report
    per_class_counts = {c: 0 for c in all_classes}
    for _, cidx in merged_samples:
        per_class_counts[idx_to_class_out[int(cidx)]] += 1

    print(f"[OK] Saved merged index to: {out_path}")
    print(f"  - classes: {len(all_classes)}")
    print(f"  - samples: {len(merged_samples)}")
    print("  - per-class counts:")
    for c in all_classes:
        print(f"    {c}: {per_class_counts[c]}")


if __name__ == "__main__":
    merge_index_pts(
        INDEX1_PATH,
        INDEX2_PATH,
        OUT_PATH,
        dedup_by_path=True,
        sort_paths_within_class=False,
    )