import argparse
import csv
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CheckIdxError(RuntimeError):
    """Raised when validation fails under fail-fast policy."""


def pil_to_tensor(img):
    byte_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    n_channels = len(img.getbands())
    byte_tensor = byte_tensor.view(img.size[1], img.size[0], n_channels)
    return byte_tensor.permute(2, 0, 1).contiguous().float().div(255.0)


class IndexedDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(label)


def validate_runtime_args(batch_size, num_workers, max_batches):
    if batch_size < 1:
        raise CheckIdxError(f"batch_size must be >= 1, got {batch_size}")
    if num_workers < 0:
        raise CheckIdxError(f"num_workers must be >= 0, got {num_workers}")
    if max_batches is not None and max_batches < 1:
        raise CheckIdxError(f"max_batches must be >= 1 when provided, got {max_batches}")


def load_index(index_file):
    index_path = Path(index_file)
    data = torch.load(index_path, map_location="cpu")
    if not isinstance(data, dict):
        raise CheckIdxError(f"Index file must contain a dict, got: {type(data)}")
    return index_path, data

def _count_by_class(samples, idx_to_class):
    """
    samples: [(path, label), ...]
    return: {class_name: count}
    """
    counter = Counter()

    for path, label in samples:
        label = int(label)
        cname = idx_to_class.get(label, f"class_{label}")
        counter[cname] += 1

    return counter


def inspect_merge_sources(index_file):
    """
    merged index 기준
    index1 / index2 / merged class 분포 비교
    """

    index_path, data = load_index(index_file)

    structure = validate_index_structure(data)

    samples = structure["samples"]
    idx_to_class = structure["idx_to_class"]

    # merged 분포
    merged_counter = _count_by_class(samples, idx_to_class)

    sources = data.get("sources")

    if not sources or len(sources) != 2:
        print("[INFO] This index does not contain merge source metadata.")
        print("[INFO] Only merged distribution will be shown.")
        src1_counter = {}
        src2_counter = {}
    else:
        src1_path, src2_path = sources

        _, data1 = load_index(src1_path)
        _, data2 = load_index(src2_path)

        s1 = validate_index_structure(data1)
        s2 = validate_index_structure(data2)

        src1_counter = _count_by_class(s1["samples"], s1["idx_to_class"])
        src2_counter = _count_by_class(s2["samples"], s2["idx_to_class"])

    classes = sorted(set(list(merged_counter.keys()) +
                         list(src1_counter.keys()) +
                         list(src2_counter.keys())))

    print("")
    print("[INFO] Merge source distribution")
    print("|class|label|index1_count|index2_count|merged_total|")
    print("|---|---:|---:|---:|---:|")

    for cname in classes:
        label = structure["class_to_idx"].get(cname, -1)

        c1 = src1_counter.get(cname, 0)
        c2 = src2_counter.get(cname, 0)
        cm = merged_counter.get(cname, 0)

        print(f"|{cname}|{label}|{c1}|{c2}|{cm}|")

    print("")

def normalize_idx_to_class(idx_to_class):
    if idx_to_class is None:
        return {}
    if not isinstance(idx_to_class, dict):
        raise CheckIdxError(f"idx_to_class must be dict, got: {type(idx_to_class)}")
    try:
        return {int(k): str(v) for k, v in idx_to_class.items()}
    except Exception as e:
        raise CheckIdxError(f"Failed to normalize idx_to_class: {type(e).__name__}: {e}")


def normalize_class_to_idx(class_to_idx):
    if class_to_idx is None:
        return {}
    if not isinstance(class_to_idx, dict):
        raise CheckIdxError(f"class_to_idx must be dict, got: {type(class_to_idx)}")
    try:
        return {str(k): int(v) for k, v in class_to_idx.items()}
    except Exception as e:
        raise CheckIdxError(f"Failed to normalize class_to_idx: {type(e).__name__}: {e}")


def validate_index_structure(data):
    if "samples" not in data:
        raise CheckIdxError("Missing required key: samples")

    samples = data.get("samples")
    if not isinstance(samples, (list, tuple)):
        raise CheckIdxError(f"'samples' must be list/tuple, got: {type(samples)}")

    class_to_idx = normalize_class_to_idx(data.get("class_to_idx"))
    idx_to_class = normalize_idx_to_class(data.get("idx_to_class"))

    if class_to_idx and idx_to_class:
        for class_name, idx in class_to_idx.items():
            mapped_name = idx_to_class.get(idx)
            if mapped_name != class_name:
                raise CheckIdxError(
                    f"Mapping mismatch: class_to_idx['{class_name}']={idx}, "
                    f"but idx_to_class[{idx}]={mapped_name!r}"
                )

    return {
        "samples": list(samples),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }


def inspect_samples(samples, idx_to_class):
    if len(samples) == 0:
        raise CheckIdxError("Index contains zero samples")

    warnings = []
    label_counter = Counter()
    valid_samples = []

    valid_labels = set(idx_to_class.keys()) if idx_to_class else None

    for i, item in enumerate(samples):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise CheckIdxError(f"Sample[{i}] must be (path, label), got: {item!r}")

        path, label = item
        path = Path(path)

        try:
            label = int(label)
        except Exception:
            raise CheckIdxError(f"Sample[{i}] label is not int-castable: {label!r}")

        if valid_labels is not None and label not in valid_labels:
            raise CheckIdxError(f"Sample[{i}] label {label} not found in idx_to_class")

        if not path.exists():
            raise CheckIdxError(f"Sample[{i}] image file does not exist: {path}")

        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            raise CheckIdxError(
                f"Sample[{i}] invalid/corrupted image: {path} :: {type(e).__name__}: {e}"
            )

        if idx_to_class:
            path_class_name = path.parent.name
            mapped_class_name = idx_to_class.get(label)
            if mapped_class_name is not None and path_class_name != mapped_class_name:
                warnings.append(
                    f"{path} :: parent='{path_class_name}' vs idx_to_class[{label}]='{mapped_class_name}'"
                )

        label_counter[label] += 1
        valid_samples.append((str(path), label))

    return {
        "warnings": warnings,
        "label_counter": label_counter,
        "valid_samples": valid_samples,
    }


def build_dataloader(samples, batch_size=32, num_workers=0):
    dataset = IndexedDataset(samples=samples, transform=pil_to_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataset, loader


def validate_dataloader(loader, max_batches=None):
    first_batch_summary = None
    reference_sample_shape = None
    batch_count = 0
    total_loaded = 0

    try:
        for batch_idx, batch in enumerate(loader):
            images, labels = batch
            batch_count += 1
            total_loaded += int(labels.shape[0])

            if not isinstance(images, torch.Tensor):
                raise CheckIdxError(f"Batch {batch_idx}: images is not torch.Tensor")
            if not isinstance(labels, torch.Tensor):
                raise CheckIdxError(f"Batch {batch_idx}: labels is not torch.Tensor")
            if images.ndim != 4:
                raise CheckIdxError(f"Batch {batch_idx}: images.ndim must be 4, got {images.ndim}")
            if labels.ndim != 1:
                raise CheckIdxError(f"Batch {batch_idx}: labels.ndim must be 1, got {labels.ndim}")

            sample_shape = tuple(images.shape[1:])
            if reference_sample_shape is None:
                reference_sample_shape = sample_shape
                first_batch_summary = {
                    "batch_idx": batch_idx,
                    "image_batch_shape": tuple(images.shape),
                    "label_batch_shape": tuple(labels.shape),
                    "sample_tensor_shape": sample_shape,
                }
            elif sample_shape != reference_sample_shape:
                raise CheckIdxError(
                    f"Batch {batch_idx}: sample tensor shape mismatch. "
                    f"expected {reference_sample_shape}, got {sample_shape}"
                )

            if max_batches is not None and batch_count >= max_batches:
                break
    except CheckIdxError:
        raise
    except Exception as e:
        raise CheckIdxError(f"DataLoader iteration failed: {type(e).__name__}: {e}")

    if batch_count == 0:
        raise CheckIdxError("No batch was produced by DataLoader")

    return {
        "first_batch_summary": first_batch_summary,
        "reference_sample_shape": reference_sample_shape,
        "batch_count": batch_count,
        "total_loaded": total_loaded,
    }


def distribution_rows(label_counter, idx_to_class):
    rows = []
    total = sum(label_counter.values())
    for label in sorted(label_counter.keys()):
        class_name = idx_to_class.get(label, f"class_{label}")
        rows.append({"class": class_name, "label": label, "count": int(label_counter[label]), "total": total})
    return rows


def save_distribution_csv(rows, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "label", "count", "total"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def save_warning_report(warnings, report_path):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# check_idx warning report", ""]
    lines.append("## path_class_mismatch")
    if warnings:
        lines.extend([f"- {item}" for item in warnings])
    else:
        lines.append("- none")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def print_stage_pass(title, extras=None, warnings=None):
    extras = extras or []
    warnings = warnings or []
    status = "PASS" if not warnings else "WARN"
    print(f"[{status}] {title}")
    for line in extras:
        print(f"  - {line}")
    for line in warnings:
        print(f"  - warning: {line}")
    print("")


def print_stage_fail(title, error, extras=None):
    extras = extras or []
    print(f"[FAIL] {title}")
    for line in extras:
        print(f"  - {line}")
    print(f"  - error: {error}")
    print("")


def run_check(index_file, batch_size=32, num_workers=0, output_csv=None, warning_report=None, max_batches=None):
    validate_runtime_args(batch_size=batch_size, num_workers=num_workers, max_batches=max_batches)

    try:
        index_path, data = load_index(index_file)
    except Exception as e:
        if isinstance(e, CheckIdxError):
            raise
        raise CheckIdxError(f"Failed to load index file: {type(e).__name__}: {e}")

    try:
        structure = validate_index_structure(data)
        print_stage_pass(
            "Index structure validation",
            extras=[
                f"index_file={index_path}",
                f"sample_count={len(structure['samples'])}",
                f"class_to_idx_count={len(structure['class_to_idx'])}",
                f"idx_to_class_count={len(structure['idx_to_class'])}",
            ],
        )
    except CheckIdxError as e:
        print_stage_fail(
            "Index structure validation",
            str(e),
            extras=[f"index_file={index_path}"],
        )
        raise

    try:
        sample_report = inspect_samples(structure["samples"], structure["idx_to_class"])
        print_stage_pass(
            "Sample metadata and image validation",
            extras=[
                f"validated_samples={sum(sample_report['label_counter'].values())}",
            ],
            warnings=sample_report["warnings"],
        )
    except CheckIdxError as e:
        print_stage_fail("Sample metadata and image validation", str(e))
        raise

    dataset, loader = build_dataloader(
        sample_report["valid_samples"],
        batch_size=batch_size,
        num_workers=num_workers,
    )

    try:
        dl_report = validate_dataloader(loader, max_batches=max_batches)
        extras = [
            f"usable_samples={len(dataset)}",
            f"batch_size={batch_size}",
            f"num_workers={num_workers}",
            f"checked_batches={dl_report['batch_count']}",
            f"loaded_samples={dl_report['total_loaded']}",
        ]
        if dl_report["first_batch_summary"] is not None:
            first = dl_report["first_batch_summary"]
            extras.extend([
                f"first_image_batch_shape={first['image_batch_shape']}",
                f"first_label_batch_shape={first['label_batch_shape']}",
                f"reference_sample_shape={first['sample_tensor_shape']}",
            ])
        print_stage_pass("DataLoader batch validation", extras=extras)
    except CheckIdxError as e:
        print_stage_fail(
            "DataLoader batch validation",
            str(e),
            extras=[
                f"usable_samples={len(dataset)}",
                f"batch_size={batch_size}",
                f"num_workers={num_workers}",
            ],
        )
        raise

    rows = distribution_rows(sample_report["label_counter"], structure["idx_to_class"])
    print("[INFO] Dataset distribution")
    inspect_merge_sources(index_file)

    output_csv = output_csv or (index_path.stem + "_distribution.csv")
    warning_report = warning_report or (index_path.stem + "_warnings.md")
    csv_path = save_distribution_csv(rows, output_csv)
    warn_path = save_warning_report(sample_report["warnings"], warning_report)

    print(f"[INFO] distribution_csv={csv_path}")
    print(f"[INFO] warning_report={warn_path}")
    print("")
    print("[SUMMARY]")
    print("  - errors=0")
    print(f"  - warnings={len(sample_report['warnings'])}")
    print(f"  - valid_counted_samples={sum(sample_report['label_counter'].values())}")

    return {
        "structure": structure,
        "sample_report": sample_report,
        "dataloader_report": dl_report,
        "distribution_rows": rows,
        "output_csv": str(csv_path),
        "warning_report": str(warn_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Validate index .pt file with fail-fast policy.")
    parser.add_argument("--index-file", type=str, default="./dataset_index.pt", help="Path to index .pt file")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to distribution csv output")
    parser.add_argument("--warning-report", type=str, default=None, help="Path to warning report markdown")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit number of batches to inspect")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_check(
            index_file=args.index_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_csv=args.output_csv,
            warning_report=args.warning_report,
            max_batches=args.max_batches,
        )
    except CheckIdxError:
        raise SystemExit(1)
