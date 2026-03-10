import os
import random
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")  # force non-interactive backend for training-script history plot saving

from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms

# =============================================================================
# [User Configuration]
# =============================================================================
"""
목표:
- resnet_hpo.py에서 찾은 Best params를 수동으로 입력 → 그대로 학습에 적용
- 이미 준비된 train dataset idx(.pt)와 val dataset idx(.pt)를 직접 불러와 학습
- 테스트(evaluation)는 별도: evaluation_model.py에서 수행

주의:
- 이 스크립트는 train/val/test 폴더를 만들지 않는다.
- train/val 분할은 외부에서 완료되어 있어야 하며, 이 스크립트 내부에서 CV/랜덤 분할을 수행하지 않는다.
"""

# -----------------------------
# (1) 경로/모드
# -----------------------------
TRAIN_DATASET_IDX_PATH = "./data_idx/ratio_90.pt"
VAL_DATASET_IDX_PATH = "./data_idx/val_idx.pt"
OUTPUT_DIR = "./results/ratio_90" # 예: ./resnet/best_model.pth, ./resnet/train_summary.json
ARCH = "resnet18"  # "resnet18" 고정(=resnet_hpo.py와 동일 설계). 필요시 확장 가능.
# -----------------------------
# (6) ★ Hyperparameters from HPO (수동 입력란)
# - resnet_hpo.py 결과(best_params)를 그대로 붙여넣기
# -----------------------------
HPO_BEST_PARAMS: Dict = {'n_layers': 3, 'n_units_0': 467, 'dropout_l0': 0.30844343466447566, 'n_units_1': 239, 'dropout_l1': 0.3824408145889585, 'n_units_2': 69, 'dropout_l2': 0.3982943760328624, 'uf_layers': 3, 'lr': 0.00010256656661556426}

# -----------------------------
# (2) Transfer / freeze policy
# - 전이학습 정책은 finetune으로 고정
# - head(n_layers) + uf_layers 적용(탐색 결과를 사용)
# -----------------------------

# -----------------------------
# (3) 고정 분할 설정
# -----------------------------
SEED = 42

# -----------------------------
# (4) 학습 설정
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 7              # EarlyStopping (val_loss 기준)
NUM_WORKERS = 4
PIN_MEMORY = True

# -----------------------------
# (5) Online Augmentation 설정 (on/off)
# - Geometric: squeezing/resize
# - Radiometric: Gaussian noise/brightness/contrast
# -----------------------------
ONLINE_AUG_CONFIG = {
    "use_resize": True,
    "use_squeeze": True,
    "use_color": True,
    "use_noise": True,
    "use_flip": True,
}


# -----------------------------
# (7) history plot options
# -----------------------------
SAVE_HISTORY_PLOTS = True  # True로 하면 학습 그래프 저장
HISTORY_PLOT_MAX_EPOCHS = 30

HISTORY_ACC_YLIM = (0.0, 1.0)  # None
HISTORY_LOSS_YLIM = (0.0, 2.5)  # (0.0, 2.5), None

HISTORY_PLOT_DIR = os.path.join(OUTPUT_DIR, "history_plots")

# =============================================================================
# [Utils]
# =============================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.05, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() > self.p:
            return tensor
        return tensor + torch.randn_like(tensor) * self.std + self.mean


def build_transforms(is_train: bool = True) -> transforms.Compose:
    """
    resnet_hpo.py의 build_transforms 로직을 CV 학습용으로 재사용.
    """
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if not is_train:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    tf_list = []

    if ONLINE_AUG_CONFIG["use_resize"]:
        tf_list.append(
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1))
        )
    else:
        tf_list.append(transforms.Resize((224, 224)))

    if ONLINE_AUG_CONFIG["use_squeeze"]:
        tf_list.append(
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=(-5, 5),
            )
        )

    if ONLINE_AUG_CONFIG["use_flip"]:
        tf_list.append(transforms.RandomHorizontalFlip())

    if ONLINE_AUG_CONFIG["use_color"]:
        tf_list.append(transforms.ColorJitter(brightness=0.15, contrast=0.15))

    tf_list.append(transforms.ToTensor())

    if ONLINE_AUG_CONFIG["use_noise"]:
        tf_list.append(AddGaussianNoise(std=0.05, p=0.3))

    tf_list.append(transforms.Normalize(mean, std))

    return transforms.Compose(tf_list)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# [Dataset: FileListDataset]
# =============================================================================
class FileListDataset(Dataset):
    """
    samples: List[(img_path, class_idx)]
    resnet_hpo_v6.py의 PathImageDataset contract를 최대한 반영:
    - classes
    - class_to_idx
    - targets
    """
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform=None,
        classes: Optional[List[str]] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.samples = list(samples)
        self.transform = transform
        self.classes = list(classes) if classes is not None else []
        self.class_to_idx = dict(class_to_idx) if class_to_idx is not None else {}
        self.targets = [y for _, y in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                "[Error] failed to load image sample.\n"
                f" - sample_index: {i}\n"
                f" - path: {path}\n"
                f" - class_idx: {y}\n"
                " - check: file integrity, file permission, mount/path consistency, and PIL-readable format\n"
                " - note: this is a sample loading failure during training, not direct evidence of train/val leakage"
            ) from e
        if self.transform:
            img = self.transform(img)
        return img, y


# =============================================================================
# [Data Loading: dataset idx]
# =============================================================================
def _normalize_idx_to_class(idx_to_class: Dict) -> Dict[int, str]:
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
        "n_samples": len(parsed_samples),
    }

    return parsed_samples, classes, class_to_idx, idx_to_class, meta


def validate_train_val_idx_compatibility(
    train_classes: List[str],
    train_class_to_idx: Dict[str, int],
    train_idx_to_class: Dict[int, str],
    val_classes: List[str],
    val_class_to_idx: Dict[str, int],
    val_idx_to_class: Dict[int, str],
):
    if train_classes != val_classes:
        raise ValueError(
            "[Error] train/val classes do not match.\n"
            f" - train_classes: {train_classes}\n"
            f" - val_classes: {val_classes}"
        )
    if train_class_to_idx != val_class_to_idx:
        raise ValueError(
            "[Error] train/val class_to_idx do not match.\n"
            f" - train_class_to_idx: {train_class_to_idx}\n"
            f" - val_class_to_idx: {val_class_to_idx}"
        )
    if train_idx_to_class != val_idx_to_class:
        raise ValueError(
            "[Error] train/val idx_to_class do not match.\n"
            f" - train_idx_to_class: {train_idx_to_class}\n"
            f" - val_idx_to_class: {val_idx_to_class}"
        )


def inspect_train_val_leakage_risk(
    train_samples: List[Tuple[str, int]],
    val_samples: List[Tuple[str, int]],
):
    train_paths = {path for path, _ in train_samples}
    val_paths = {path for path, _ in val_samples}
    overlap_paths = sorted(train_paths & val_paths)
    if overlap_paths:
        raise ValueError(
            "[Error] identical image paths appear in both train and val datasets.\n"
            f" - overlap_path_count: {len(overlap_paths)}\n"
            f" - preview: {overlap_paths[:10]}\n"
            " - check: train/val idx split, duplicated registration, or data leakage"
        )

# =============================================================================
# [Model: HPO params -> model]
# =============================================================================
def build_head_from_hp(in_features: int, num_classes: int, hp: Dict) -> nn.Sequential:
    """
    resnet_hpo.py의 head 설계를 그대로 재현 (n_layers 전용).
    - 필수 키: n_layers, n_units_i, dropout_li
    """
    if "n_layers" not in hp:
        raise KeyError("[Error] 'n_layers' is required in HPO_BEST_PARAMS")

    n_layers = int(hp["n_layers"])
    if n_layers < 1:
        raise ValueError("[Error] n_layers must be >= 1")

    layers: List[nn.Module] = []
    cur_in = in_features

    for i in range(n_layers):
        key_units = f"n_units_{i}"
        key_drop = f"dropout_l{i}"

        if key_units not in hp or key_drop not in hp:
            raise KeyError(
                f"[Error] missing head param(s): {key_units}, {key_drop}"
            )

        out_features = int(hp[key_units])
        p = float(hp[key_drop])

        layers.append(nn.Linear(cur_in, out_features))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p))
        cur_in = out_features

    layers.append(nn.Linear(cur_in, num_classes))
    return nn.Sequential(*layers)


def apply_finetune_freeze_policy_from_hp(model: nn.Module, hp: Dict) -> Dict:
    """
    finetune 전용 freeze 정책.
    - 기본: 전체 freeze
    - head(fc)는 항상 trainable
    - uf_layers(0~4) 적용하여 layer1~4 중 뒤에서 uf_layers개 unfreeze
    """
    for p in model.parameters():
        p.requires_grad = False

    for p in model.fc.parameters():
        p.requires_grad = True

    uf_layers = int(hp.get("uf_layers", 0))
    uf_layers = max(0, min(4, uf_layers))
    stages = [model.layer1, model.layer2, model.layer3, model.layer4]
    if uf_layers > 0:
        for stage in stages[-uf_layers:]:
            for p in stage.parameters():
                p.requires_grad = True

    return {"uf_layers": uf_layers}


def build_resnet18_from_hpo(num_classes: int, hp: Dict) -> Tuple[nn.Module, Dict]:
    """
    resnet_hpo.py의 define_model(trial)을
    - best_params(hp)로 고정
    - finetune 정책으로 고정 적용
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # head 교체
    in_features = model.fc.in_features
    model.fc = build_head_from_hp(in_features, num_classes, hp)

    # freeze/unfreeze
    freeze_info = apply_finetune_freeze_policy_from_hp(model, hp)

    return model, freeze_info


# =============================================================================
# [Train]
# =============================================================================
@dataclass
class TrainResult:
    best_epoch: int
    early_stopped: bool
    best_val_loss: float
    best_val_acc: float
    final_train_loss: float
    final_train_acc: float
    ckpt_path: str


def train_once(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
    patience: int,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = -1
    best_state = None
    patience_ctr = 0
    early_stopped = False

    model.to(device)

    for epoch in range(1, epochs + 1):
        # ---- train
        model.train()
        tloss_sum, tcorrect, ttotal = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            tloss_sum += loss.item() * x.size(0)
            tcorrect += (logits.argmax(1) == y).sum().item()
            ttotal += x.size(0)

        train_loss = tloss_sum / max(1, ttotal)
        train_acc = tcorrect / max(1, ttotal)

        # ---- val
        model.eval()
        vloss_sum, vcorrect, vtotal = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                vloss_sum += loss.item() * x.size(0)
                vcorrect += (logits.argmax(1) == y).sum().item()
                vtotal += x.size(0)

        val_loss = vloss_sum / max(1, vtotal)
        val_acc = vcorrect / max(1, vtotal)

        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(float(lr_now))

        # ---- early stopping (val_loss)
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = float(val_loss)
            best_val_acc = float(val_acc)
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                early_stopped = True
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    final_train_loss = float(history["train_loss"][-1]) if history["train_loss"] else 0.0
    final_train_acc = float(history["train_acc"][-1]) if history["train_acc"] else 0.0

    return model, history, TrainResult(
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        best_val_loss=float(best_val_loss),
        best_val_acc=float(best_val_acc),
        final_train_loss=final_train_loss,
        final_train_acc=final_train_acc,
        ckpt_path="",  # save 단계에서 채움
    )


# =============================================================================
# [Save]
# =============================================================================
def save_training_checkpoint(
    save_path: str,
    model: nn.Module,
    train_history: Dict,
    classes: List[str],
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[int, str],
    split_indices: Dict[str, List[int]],
    transform_config: Dict,
    train_dataset_idx_path: str,
    val_dataset_idx_path: str,
    train_idx_meta: Dict,
    val_idx_meta: Dict,
    arch: str,
    hpo_params: Dict,
    freeze_info: Dict,
    device_str: str,
):
    payload = {
        "model_state": model.state_dict(),
        "history": train_history,
        "classes": list(classes),
        "class_to_idx": class_to_idx,
        "idx_to_class": dict(idx_to_class),
        "split_indices": split_indices,
        "transform_config": transform_config,
        "train_dataset_idx_path": train_dataset_idx_path,
        "val_dataset_idx_path": val_dataset_idx_path,
        "train_idx_meta": dict(train_idx_meta),
        "val_idx_meta": dict(val_idx_meta),
        "arch": arch,
        "mode": "train_with_fixed_val",
        "num_classes": len(class_to_idx),
        "torch_version": torch.__version__,
        "device_str": device_str,
        "hpo_params": dict(hpo_params),
        "freeze_info": dict(freeze_info),
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(payload, save_path)

def save_history_csv(history: dict, save_dir: str, filename: str = "history.csv"):
    """
    training history를 CSV로 저장
    columns:
    epoch, train_loss, train_acc, val_loss, val_acc, lr
    """
    import csv
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, filename)

    keys = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]

    n = len(history["epoch"])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)

        for i in range(n):
            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i],
                history["train_acc"][i],
                history["val_loss"][i],
                history["val_acc"][i],
                history["lr"][i],
            ])

    print(f"[History] CSV saved: {path}")

def plot_training_history(
    history: dict,
    save_dir: str,
    tag: str = "train",
    *,
    max_epochs: int = 30,
    acc_ylim: tuple = (0.0, 1.0),
    loss_ylim: tuple | None = None,
):
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    tr_loss = history.get("train_loss", [])
    tr_acc = history.get("train_acc", [])
    va_loss = history.get("val_loss", [])
    va_acc = history.get("val_acc", [])

    n = min(len(tr_loss), len(tr_acc), len(va_loss), len(va_acc))
    if n == 0:
        print(f"[Plot] skip {tag}: history empty")
        return

    x = np.arange(1, n + 1)
    acc_path = os.path.join(save_dir, f"{tag}_acc.png")
    loss_path = os.path.join(save_dir, f"{tag}_loss.png")

    fig_acc, ax_acc = plt.subplots(figsize=(7, 4))
    ax_acc.plot(x, tr_acc[:n], label="train_acc")
    ax_acc.plot(x, va_acc[:n], label="val_acc")
    ax_acc.set_xlim(1, max_epochs)
    ax_acc.set_ylim(*acc_ylim)
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title(f"{tag} Accuracy")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()
    fig_acc.tight_layout()
    fig_acc.savefig(acc_path, dpi=200)
    plt.close(fig_acc)

    if loss_ylim is None:
        loss_vals = np.array(tr_loss[:n] + va_loss[:n], dtype=float)
        finite = loss_vals[np.isfinite(loss_vals)]
        if finite.size == 0:
            y0, y1 = 0.0, 1.0
        else:
            mn = float(finite.min())
            mx = float(finite.max())
            pad = (mx - mn) * 0.08 if mx > mn else 0.1
            y0, y1 = max(0.0, mn - pad), mx + pad
        loss_ylim = (y0, y1)

    fig_loss, ax_loss = plt.subplots(figsize=(7, 4))
    ax_loss.plot(x, tr_loss[:n], label="train_loss")
    ax_loss.plot(x, va_loss[:n], label="val_loss")
    ax_loss.set_xlim(1, max_epochs)
    ax_loss.set_ylim(*loss_ylim)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title(f"{tag} Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()
    fig_loss.tight_layout()
    fig_loss.savefig(loss_path, dpi=200)
    plt.close(fig_loss)
    plt.close("all")

    print(f"[Plot] backend=Agg | saved: {acc_path}")
    print(f"[Plot] backend=Agg | saved: {loss_path}")


# =============================================================================
# [Logging]
# =============================================================================
def print_train_val_distribution_table(
    class_to_idx: Dict[str, int],
    train_labels: List[int],
    val_labels: List[int],
):
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    train_counts = defaultdict(int)
    val_counts = defaultdict(int)

    for y in train_labels:
        train_counts[idx_to_class[y]] += 1
    for y in val_labels:
        val_counts[idx_to_class[y]] += 1

    headers = ["class", "train", "val", "total"]
    rows = []
    for cls_name in sorted(class_to_idx.keys(), key=lambda k: class_to_idx[k]):
        tr = train_counts.get(cls_name, 0)
        va = val_counts.get(cls_name, 0)
        rows.append([cls_name, tr, va, tr + va])

    rows.append(["TOTAL", len(train_labels), len(val_labels), len(train_labels) + len(val_labels)])

    col_widths = [
        max(len(str(x)) for x in [h] + [r[c] for r in rows])
        for c, h in enumerate(headers)
    ]

    def fmt_row(r):
        return "| " + " | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers))) + " |"

    print("\n" + "=" * 70)
    print("[Data] Class distribution (train / val)")
    print(fmt_row(headers))
    print("| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |")
    for r in rows:
        print(fmt_row(r))
    print("=" * 70 + "\n")


# =============================================================================
# [Main]
# =============================================================================
def main():
    set_seed(SEED)

    # (1) load train / val dataset idx
    train_samples, train_classes, train_class_to_idx, train_idx_to_class, train_idx_meta = load_dataset_idx(
        TRAIN_DATASET_IDX_PATH
    )
    val_samples, val_classes, val_class_to_idx, val_idx_to_class, val_idx_meta = load_dataset_idx(
        VAL_DATASET_IDX_PATH
    )

    validate_train_val_idx_compatibility(
        train_classes=train_classes,
        train_class_to_idx=train_class_to_idx,
        train_idx_to_class=train_idx_to_class,
        val_classes=val_classes,
        val_class_to_idx=val_class_to_idx,
        val_idx_to_class=val_idx_to_class,
    )
    inspect_train_val_leakage_risk(train_samples=train_samples, val_samples=val_samples)

    classes = train_classes
    class_to_idx = train_class_to_idx
    idx_to_class = train_idx_to_class
    train_labels = [y for _, y in train_samples]
    val_labels = [y for _, y in val_samples]
    num_classes = len(classes)

    # ---- sanity check: required hp keys
    if "lr" not in HPO_BEST_PARAMS:
        raise KeyError("[Error] HPO_BEST_PARAMS must include 'lr'")
    print(f"[System] DEVICE: {DEVICE}")
    print(f"[System] TRAIN_DATASET_IDX_PATH: {TRAIN_DATASET_IDX_PATH}")
    print(f"[System] VAL_DATASET_IDX_PATH: {VAL_DATASET_IDX_PATH}")
    print(
        "[System] train idx metadata | "
        f"seed={train_idx_meta.get('seed')} | "
        f"n_per_class={train_idx_meta.get('n_per_class')} | "
        f"n_samples={train_idx_meta.get('n_samples')}"
    )
    print(
        "[System] val idx metadata | "
        f"seed={val_idx_meta.get('seed')} | "
        f"n_per_class={val_idx_meta.get('n_per_class')} | "
        f"n_samples={val_idx_meta.get('n_samples')}"
    )
    if train_idx_meta.get("sources") is not None:
        print(f"[System] train idx sources: {train_idx_meta['sources']}")
    if val_idx_meta.get("sources") is not None:
        print(f"[System] val idx sources: {val_idx_meta['sources']}")
    print(
        f"[System] ARCH: {ARCH}  |  classes: {num_classes}  |  "
        f"train_samples: {len(train_samples)}  |  val_samples: {len(val_samples)}"
    )
    print(f"[HPO] Best params (manual): {dict(HPO_BEST_PARAMS)}")
    print(f"[Aug] Online aug config: {ONLINE_AUG_CONFIG}")

    print_train_val_distribution_table(class_to_idx, train_labels, val_labels)

    train_tf = build_transforms(is_train=True)
    val_tf = build_transforms(is_train=False)

    transform_config = {
        "input_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "train": {
            "online_aug_config": dict(ONLINE_AUG_CONFIG),
        },
        "val": {
            "center_crop": True,
        },
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_ds = FileListDataset(
        train_samples,
        transform=train_tf,
        classes=classes,
        class_to_idx=class_to_idx,
    )
    val_ds = FileListDataset(
        val_samples,
        transform=val_tf,
        classes=classes,
        class_to_idx=class_to_idx,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model, freeze_info = build_resnet18_from_hpo(
        num_classes=num_classes,
        hp=HPO_BEST_PARAMS,
    )
    model = model.to(DEVICE)

    trainable = count_trainable_params(model)
    lr = float(HPO_BEST_PARAMS["lr"])

    print(f"[Train] build model | trainable_params={trainable:,} | freeze_info={freeze_info} | lr={lr:.2e}")

    model, train_history, train_result = train_once(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        lr=lr,
        epochs=EPOCHS,
        patience=PATIENCE,
    )

    ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    split_indices = {
        "train": list(range(len(train_samples))),
        "val": list(range(len(val_samples))),
    }
    save_training_checkpoint(
        save_path=ckpt_path,
        model=model,
        train_history=train_history,
        classes=classes,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        split_indices=split_indices,
        transform_config=transform_config,
        train_dataset_idx_path=TRAIN_DATASET_IDX_PATH,
        val_dataset_idx_path=VAL_DATASET_IDX_PATH,
        train_idx_meta=train_idx_meta,
        val_idx_meta=val_idx_meta,
        arch=ARCH,
        hpo_params=HPO_BEST_PARAMS,
        freeze_info=freeze_info,
        device_str=str(DEVICE),
    )

    if SAVE_HISTORY_PLOTS:
        plot_training_history(
            history=train_history,
            save_dir=HISTORY_PLOT_DIR,
            tag="train",
            max_epochs=HISTORY_PLOT_MAX_EPOCHS,
            acc_ylim=HISTORY_ACC_YLIM,
            loss_ylim=HISTORY_LOSS_YLIM,
        )

    save_history_csv(
    history=train_history,
    save_dir=OUTPUT_DIR,
    filename="training_history.csv"
    )

    train_result.ckpt_path = ckpt_path

    summary = {
        "train_result": asdict(train_result),
        "best_model_path": ckpt_path,
        "classes": list(classes),
        "class_to_idx": class_to_idx,
        "idx_to_class": dict(idx_to_class),
        "transform_config": transform_config,
        "train_dataset_idx_path": TRAIN_DATASET_IDX_PATH,
        "val_dataset_idx_path": VAL_DATASET_IDX_PATH,
        "train_idx_meta": dict(train_idx_meta),
        "val_idx_meta": dict(val_idx_meta),
        "arch": ARCH,
        "mode": "train_with_fixed_val_summary",
        "hpo_params": dict(HPO_BEST_PARAMS),
        "torch_version": torch.__version__,
    }
    summary_path = os.path.join(OUTPUT_DIR, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    es_epoch = train_result.best_epoch
    print(
        f"[Train] train_acc(last): {train_result.final_train_acc:.4f}, "
        f"train_loss(last): {train_result.final_train_loss:.4f} | "
        f"val_acc(best@{es_epoch}): {train_result.best_val_acc:.4f}, "
        f"val_loss(best@{es_epoch}): {train_result.best_val_loss:.4f} | "
        f"early_stop: {train_result.early_stopped}"
    )

    print("\n" + "=" * 40)
    print(f"[Done] Saved best model: {ckpt_path}")
    print(f"[Done] Saved summary: {summary_path}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
