
import os
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
import optuna
from collections import defaultdict
from tabulate import tabulate

# =============================================================================
# [User Configuration]
# =============================================================================

SEED = 42

# 1. Online Augmentation 설정
ONLINE_AUG_CONFIG = {
    'use_resize': False,
    'use_squeeze': False,
    'use_color': False,
    'use_noise': False,
    'use_flip': False
}

# 2. 학습 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DATASET_IDX_PATH = "./data_idx/ratio_90.pt"
VAL_DATASET_IDX_PATH = "./data_idx/val_idx.pt"
BATCH_SIZE = 64
NUM_WORKERS = 4
FIXED_EPOCHS = 30
N_TRIALS = 20

# =============================================================================
# [Helper Class & Functions]
# =============================================================================

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() > self.p:
            return tensor
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


def build_transforms(is_train=True):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if not is_train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    tf_list = []

    if ONLINE_AUG_CONFIG['use_resize']:
        tf_list.append(transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)))
    else:
        tf_list.append(transforms.Resize((224, 224)))

    if ONLINE_AUG_CONFIG['use_squeeze']:
        tf_list.append(transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=(-5, 5)
        ))

    if ONLINE_AUG_CONFIG['use_flip']:
        tf_list.append(transforms.RandomHorizontalFlip())

    if ONLINE_AUG_CONFIG['use_color']:
        tf_list.append(transforms.ColorJitter(brightness=0.15, contrast=0.15))

    tf_list.append(transforms.ToTensor())

    if ONLINE_AUG_CONFIG['use_noise']:
        tf_list.append(AddGaussianNoise(std=0.05, p=0.3))

    tf_list.append(transforms.Normalize(mean, std))
    return transforms.Compose(tf_list)


class ApplyTransform(torch.utils.data.Dataset):
    """Dataset/Subset에 transform을 적용하기 위한 래퍼 클래스"""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

    @property
    def classes(self):
        if isinstance(self.subset, Subset):
            return self.subset.dataset.classes
        return self.subset.classes


# =============================================================================
# [Data Loading Logic]
# =============================================================================

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError("[Error] PIL(Pillow)이 필요합니다. `pip install pillow` 후 다시 실행하세요.") from e


class PathImageDataset(torch.utils.data.Dataset):
    """(path, label) 리스트 기반 Dataset.
    - ImageFolder처럼 classes / class_to_idx / targets 제공 (카운팅/로그 호환)
    - 반환은 (PIL.Image, label)
    """
    def __init__(self, samples, classes, class_to_idx):
        self.samples = list(samples)  # List[Tuple[str, int]]
        self.classes = list(classes)
        self.class_to_idx = dict(class_to_idx)
        self.targets = [y for _, y in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return img, y


def _seeded_shuffle(arr, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(arr))
    rng.shuffle(idx)
    return [arr[i] for i in idx]


def _normalize_idx_to_class(idx_to_class):
    normalized = {}
    for k, v in idx_to_class.items():
        try:
            normalized[int(k)] = v
        except (TypeError, ValueError) as e:
            raise ValueError(f"[Error] idx_to_class key must be int-castable, got key={k!r}") from e
    return normalized


def _validate_and_parse_dataset_idx(payload: dict):
    required_keys = {"samples", "class_to_idx", "idx_to_class"}
    missing = required_keys - set(payload.keys())
    if missing:
        raise KeyError(f"[Error] dataset idx payload is missing required keys: {sorted(missing)}")

    samples = payload["samples"]
    class_to_idx = payload["class_to_idx"]
    idx_to_class = _normalize_idx_to_class(payload["idx_to_class"])

    if not isinstance(samples, (list, tuple)):
        raise TypeError("[Error] payload['samples'] must be a list/tuple of (path, class_idx).")
    if not isinstance(class_to_idx, dict):
        raise TypeError("[Error] payload['class_to_idx'] must be a dict.")
    if not isinstance(payload["idx_to_class"], dict):
        raise TypeError("[Error] payload['idx_to_class'] must be a dict.")

    normalized_class_to_idx = {}
    for cls_name, cls_idx in class_to_idx.items():
        if not isinstance(cls_name, str):
            raise TypeError(f"[Error] class_to_idx key must be str, got {type(cls_name)}")
        try:
            normalized_class_to_idx[cls_name] = int(cls_idx)
        except (TypeError, ValueError) as e:
            raise TypeError(f"[Error] class_to_idx value must be int-castable, got {cls_idx!r}") from e

    class_indices = sorted(normalized_class_to_idx.values())
    idx_keys = sorted(idx_to_class.keys())
    if class_indices != idx_keys:
        raise ValueError(
            "[Error] class_to_idx and idx_to_class index sets do not match.\n"
            f" - class_to_idx values: {class_indices}\n"
            f" - idx_to_class keys: {idx_keys}"
        )

    expected_indices = list(range(len(normalized_class_to_idx)))
    if class_indices != expected_indices:
        raise ValueError(
            "[Error] class indices must be contiguous integers starting from 0.\n"
            f" - expected: {expected_indices}\n"
            f" - actual: {class_indices}"
        )

    mismatched_pairs = []
    for cls_name, cls_idx in normalized_class_to_idx.items():
        mapped_name = idx_to_class.get(cls_idx)
        if mapped_name != cls_name:
            mismatched_pairs.append((cls_name, cls_idx, mapped_name))

    if mismatched_pairs:
        preview = mismatched_pairs[:5]
        raise ValueError(
            "[Error] class_to_idx and idx_to_class mappings are inconsistent. "
            f"preview={preview}"
        )

    classes = [idx_to_class[i] for i in expected_indices]

    parsed_samples = []
    missing_paths = []
    invalid_labels = []

    for i, item in enumerate(samples):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"[Error] samples[{i}] must be (path, class_idx), got {item!r}")

        path, y = item
        if not isinstance(path, str):
            raise TypeError(f"[Error] samples[{i}][0] must be str path, got {type(path)}")

        try:
            y = int(y)
        except (TypeError, ValueError) as e:
            raise TypeError(f"[Error] samples[{i}][1] must be int-castable, got {y!r}") from e

        if y not in idx_to_class:
            invalid_labels.append((i, y))
            continue

        if not os.path.exists(path):
            missing_paths.append(path)

        parsed_samples.append((path, y))

    if invalid_labels:
        raise ValueError(f"[Error] some sample labels are not defined in idx_to_class: {invalid_labels[:5]}")

    if not parsed_samples:
        raise ValueError("[Error] dataset idx contains no valid samples.")

    if missing_paths:
        preview = missing_paths[:5]
        raise FileNotFoundError(
            "[Error] some sample image paths do not exist. "
            f"missing_count={len(missing_paths)}, preview={preview}"
        )

    return parsed_samples, classes, normalized_class_to_idx, idx_to_class


def load_dataset_idx(dataset_idx_path: str):
    if not os.path.exists(dataset_idx_path):
        raise FileNotFoundError(
            f"[Error] dataset idx file not found: {dataset_idx_path}\n"
            "Set DATASET_IDX_PATH to a valid merged .pt file."
        )

    try:
        payload = torch.load(dataset_idx_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(
            "[Error] failed to load dataset idx file with torch.load().\n"
            f" - path: {dataset_idx_path}\n"
            " - expected: a .pt file containing a dict with samples/class_to_idx/idx_to_class\n"
            " - check: file path, file integrity, and PyTorch save/load compatibility"
        ) from e

    if not isinstance(payload, dict):
        raise TypeError("[Error] dataset idx file must contain a dict payload.")

    parsed_samples, classes, class_to_idx, idx_to_class = _validate_and_parse_dataset_idx(payload)

    meta = {
        "seed": payload.get("seed"),
        "n_per_class": payload.get("n_per_class"),
        "sources": payload.get("sources"),
        "n_samples": payload.get("n_samples", len(parsed_samples)),
    }

    return parsed_samples, classes, class_to_idx, idx_to_class, meta


def _validate_train_val_idx_compatibility(
    train_classes,
    train_class_to_idx,
    train_idx_to_class,
    val_classes,
    val_class_to_idx,
    val_idx_to_class,
):
    if list(train_classes) != list(val_classes):
        raise ValueError(
            "[Error] train/val classes do not match.\n"
            f" - train classes: {list(train_classes)}\n"
            f" - val classes: {list(val_classes)}"
        )

    if dict(train_class_to_idx) != dict(val_class_to_idx):
        raise ValueError(
            "[Error] train/val class_to_idx do not match.\n"
            f" - train class_to_idx: {dict(train_class_to_idx)}\n"
            f" - val class_to_idx: {dict(val_class_to_idx)}"
        )

    if dict(train_idx_to_class) != dict(val_idx_to_class):
        raise ValueError(
            "[Error] train/val idx_to_class do not match.\n"
            f" - train idx_to_class: {dict(train_idx_to_class)}\n"
            f" - val idx_to_class: {dict(val_idx_to_class)}"
        )


def _assert_no_path_overlap(train_samples, val_samples):
    train_paths = {path for path, _ in train_samples}
    val_paths = {path for path, _ in val_samples}
    overlap = sorted(train_paths & val_paths)
    if overlap:
        preview = overlap[:5]
        raise ValueError(
            "[Error] train/val dataset idx files contain overlapping image paths. "
            f"overlap_count={len(overlap)}, preview={preview}"
        )


def get_dataloaders():
    print(f"[System] Loading train dataset idx from: '{TRAIN_DATASET_IDX_PATH}'")
    print(f"[System] Loading val dataset idx from:   '{VAL_DATASET_IDX_PATH}'")

    train_tf = build_transforms(is_train=True)
    val_tf = build_transforms(is_train=False)

    train_samples, classes, class_to_idx, idx_to_class, train_meta = load_dataset_idx(TRAIN_DATASET_IDX_PATH)
    val_samples, val_classes, val_class_to_idx, val_idx_to_class, val_meta = load_dataset_idx(VAL_DATASET_IDX_PATH)
    _validate_train_val_idx_compatibility(
        train_classes=classes,
        train_class_to_idx=class_to_idx,
        train_idx_to_class=idx_to_class,
        val_classes=val_classes,
        val_class_to_idx=val_class_to_idx,
        val_idx_to_class=val_idx_to_class,
    )
    _assert_no_path_overlap(train_samples, val_samples)
    num_classes = len(classes)

    train_base = PathImageDataset(train_samples, classes=classes, class_to_idx=class_to_idx)
    val_base = PathImageDataset(val_samples, classes=classes, class_to_idx=class_to_idx)

    train_ds = ApplyTransform(train_base, train_tf)
    val_ds = ApplyTransform(val_base, val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(
        "[System] train idx metadata | "
        f"seed={train_meta.get('seed')} | "
        f"n_per_class={train_meta.get('n_per_class')} | "
        f"n_samples={train_meta.get('n_samples')}"
    )
    if train_meta.get("sources") is not None:
        print(f"[System] train idx sources: {train_meta['sources']}")

    print(
        "[System] val idx metadata | "
        f"seed={val_meta.get('seed')} | "
        f"n_per_class={val_meta.get('n_per_class')} | "
        f"n_samples={val_meta.get('n_samples')}"
    )
    if val_meta.get("sources") is not None:
        print(f"[System] val idx sources: {val_meta['sources']}")
    print(f"[Result] Train: {len(train_ds)}, Val: {len(val_ds)} (Classes: {num_classes}, NUM_WORKERS: {NUM_WORKERS})")

    dataset_info = {
        "num_classes": num_classes,
        "classes": list(classes),
        "class_to_idx": dict(class_to_idx),
        "idx_to_class": dict(idx_to_class),
        "meta": {"train": dict(train_meta), "val": dict(val_meta)},
    }

    return train_loader, val_loader, dataset_info


def _unwrap(ds):
    while hasattr(ds, "subset"):
        ds = ds.subset
    return ds


def count_dataset_by_class(ds):
    """
    ImageFolder / Subset / ConcatDataset / ApplyTransform(래퍼) 대응
    return: dict {class_name: count}
    """
    ds = _unwrap(ds)
    counts = defaultdict(int)

    if isinstance(ds, ConcatDataset):
        for sub in ds.datasets:
            sub_counts = count_dataset_by_class(sub)
            for k, v in sub_counts.items():
                counts[k] += v
        return dict(counts)

    if isinstance(ds, Subset):
        base = ds.dataset
        if hasattr(base, "class_to_idx") and hasattr(base, "targets"):
            idx_to_class = {v: k for k, v in base.class_to_idx.items()}
            for i in ds.indices:
                counts[idx_to_class[base.targets[i]]] += 1
        return dict(counts)

    if hasattr(ds, "class_to_idx") and hasattr(ds, "targets"):
        idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
        for t in ds.targets:
            counts[idx_to_class[t]] += 1
        return dict(counts)


    raise TypeError(f"Unsupported dataset type for counting: {type(ds)}")


# =============================================================================
# [Model & Objective]
# =============================================================================

def build_head(trial, in_features: int, num_classes: int) -> nn.Sequential:
    """
    Head에 n_layers 적용:
    n_layers(1~3) + 각 레이어별 n_units_i, dropout_li
    """
    n_layers = trial.suggest_int("n_layers", 1, 3)

    layers = []
    cur_in = in_features

    for i in range(n_layers):
        out_features = trial.suggest_int(f"n_units_{i}", 64, 512)
        p = trial.suggest_float(f"dropout_l{i}", 0.2, 0.5)

        layers.append(nn.Linear(cur_in, out_features))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p))
        cur_in = out_features

    layers.append(nn.Linear(cur_in, num_classes))
    return nn.Sequential(*layers)


def apply_freeze_policy(trial, model: nn.Module) -> dict:
    """
    finetune 고정 freeze/unfreeze 정책 적용.
    - backbone 전체 freeze 후, fc는 항상 학습
    - uf_layers(0~4)만 탐색하여 뒤쪽 stage부터 일부 unfreeze
    return: dict (로그용 정보)
    """
    for p in model.parameters():
        p.requires_grad = False

    for p in model.fc.parameters():
        p.requires_grad = True

    uf_layers = trial.suggest_int("uf_layers", 0, 4)
    stages = [model.layer1, model.layer2, model.layer3, model.layer4]

    if uf_layers > 0:
        for stage in stages[-uf_layers:]:
            for p in stage.parameters():
                p.requires_grad = True

    info = {
        "freeze_policy": "finetune_fixed",
        "uf_layers": uf_layers,
    }
    return info


def define_model(trial, num_classes: int):
    if num_classes <= 0:
        raise ValueError(f"[Error] num_classes must be >= 1, got {num_classes}")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    in_features = model.fc.in_features
    model.fc = build_head(trial, in_features=in_features, num_classes=num_classes)

    _ = apply_freeze_policy(trial, model)

    return model.to(DEVICE)


def objective(trial, train_loader, val_loader, num_classes: int):
    model = define_model(trial, num_classes=num_classes)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FIXED_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(FIXED_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += (model(x).argmax(1) == y).sum().item()
                total += x.size(0)

        if total == 0:
            raise RuntimeError("[Error] validation loader produced zero samples. Check split logic and dataset contents.")

        val_acc = correct / total
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_acc


if __name__ == "__main__":
    try:
        train_loader, val_loader, dataset_info = get_dataloaders()
        num_classes = dataset_info["num_classes"]
        train_class_counts = count_dataset_by_class(train_loader.dataset)
        val_class_counts = count_dataset_by_class(val_loader.dataset)

        print(f"Start optimization on {DEVICE}...")
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial: objective(trial, train_loader, val_loader, num_classes), n_trials=N_TRIALS)

        print("\n" + "=" * 30)
        print("Optimization Finished")
        print(f"FIXED_EPOCHS={FIXED_EPOCHS} | BATCH_SIZE={BATCH_SIZE} | NUM_CLASSES={num_classes}")
        print(f"Best Accuracy: {study.best_value:.4f}")
        print("Best params:", study.best_params)
        for key, value in study.best_params.items():
            print(f"   {key}: {value}")

        print("\n" + "=" * 30)

        table_rows = []
        all_classes = sorted(set(train_class_counts) | set(val_class_counts))

        for cls in all_classes:
            train_cnt = train_class_counts.get(cls, 0)
            val_cnt = val_class_counts.get(cls, 0)
            total = train_cnt + val_cnt
            table_rows.append([cls, total, train_cnt, val_cnt])

        print(tabulate(
            table_rows,
            headers=["class", "total", "train", "val"],
            tablefmt="github"
        ))
        print("=" * 30 + "\n")

    except Exception as e:
        print(f"[Error] {e}")
        print("[Traceback]")
        print(traceback.format_exc())
