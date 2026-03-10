
import os
import csv
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18

import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

# GUI 없는 환경에서도 저장만 하도록
plt.switch_backend("Agg")


# =============================================================================
# [USER CONFIG] - 여기만 바꿔서 사용
# =============================================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) 테스트 데이터셋
TEST_DIR = "./test-dataset"
TEST_LABELS_CSV = os.path.join(TEST_DIR, "test_labels.csv")

# 2) 평가할 모델 경로 (K-fold checkpoints)
CKPT_PATH = "./20260108_resnet_finetune_min"
# ENSEMBLE_CKPT_PATHS: None이면 CKPT_PATH 내 .pth 파일 자동 탐색
#   직접 지정할 경우: ["./path/to/a.pth", "./path/to/b.pth", ...]
ENSEMBLE_CKPT_PATHS = None
ENSEMBLE_MODE = "logit"  # "prob" (확률 평균) or "logit" (로짓 평균)

# 3) 평가 모드
# - "single": 각 fold 단일 모델 평가 + fold mean/std 저장
# - "ensemble": 전체 ensemble 평가만 수행
EVAL_MODE = "single"  # "single" or "ensemble"
IS_FOLD = True
# True:  주어진 가중치를 fold 모델로 취급 → fold mean/std 계산 (EVAL_MODE="single"일 때만 적용)
# False: 주어진 가중치를 개별 모델로 취급 → 파일명 stem을 모델 태그로 사용

# 4) 결과 저장
RUN_NAME = "eval_run"
RESULT_DIR_NAME = None  # None이면 ECE_NUM_BINS에 맞춰 자동 생성
OUT_DIR = None  # main()에서 설정

# 5) 평가 배치/워커
BATCH_SIZE = 64
NUM_WORKERS = 4

# 6) Calibration / ECE 설정
ENABLE_CALIBRATION = True
ECE_NUM_BINS = 20

# 7) CAM 저장 설정
ENABLE_CAM = True
CAM_TOPK_HIGH = 5
CAM_TOPK_LOW = 5
CAM_SAVE_CORRECT_ALL = True
CAM_SAVE_INCORRECT_ALL = True

# 확장자: 데이터셋이 바뀌어도 최대한 많이 인식
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def resolve_out_dir() -> str:
    result_dir_name = RESULT_DIR_NAME or f"eval_results_bin{ECE_NUM_BINS}"
    return os.path.join(CKPT_PATH, result_dir_name, RUN_NAME)


def _discover_checkpoints(ckpt_dir: str) -> List[str]:
    """CKPT_PATH 내 .pth 파일을 자동 탐색하여 정렬된 경로 목록 반환.

    정렬 우선순위:
      1) 파일명에 fold(숫자) 패턴이 있으면 숫자 오름차순
      2) 없으면 파일명 알파벳 순
    """
    pth_files = list(Path(ckpt_dir).glob("*.pth"))

    def _sort_key(p: Path):
        m = re.search(r"fold(\d+)", p.stem)
        return (0, int(m.group(1)), p.stem) if m else (1, 0, p.stem)

    return [str(p) for p in sorted(pth_files, key=_sort_key)]



# =============================================================================
# 공통: 시드 고정
# =============================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_valid_image(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in ALLOWED_EXTS


# =============================================================================
# 체크포인트 로드 (PyTorch 2.6+ 대응)
# =============================================================================
def load_checkpoint(ckpt_path: str, map_location="cpu") -> Dict:
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    return ckpt


# =============================================================================
# 체크포인트 기반 모델 빌드
# =============================================================================
def build_model_from_checkpoint(ckpt: Dict, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Build model strictly from checkpoint weights.

    - torchvision `weights` is set to None to avoid any external pretrained weights.
    - FC/head structure is inferred from checkpoint state_dict keys.
    """
    arch = (ckpt.get("arch") or "resnet18").lower()
    class_to_idx: Dict[str, int] = ckpt["class_to_idx"]
    num_classes = int(ckpt.get("num_classes", len(class_to_idx)))
    hpo_params = ckpt.get("hpo_params") or ckpt.get("hpo") or {}
    state = ckpt["model_state"]

    if arch != "resnet18":
        raise ValueError(f"Unsupported architecture in checkpoint: {arch}")

    model = resnet18(weights=None)

    def _infer_and_set_fc(model: nn.Module):
        """
        Rebuild model.fc to exactly match checkpoint state_dict.
        Supports arbitrary-depth Sequential heads.
        """
        # fc.{idx}.weight 형태 수집
        fc_keys = sorted(
            k for k in state.keys()
            if k.startswith("fc.") and k.endswith(".weight")
        )

        # 단일 Linear head
        if fc_keys == ["fc.weight"]:
            in_features = state["fc.weight"].shape[1]
            out_features = state["fc.weight"].shape[0]
            model.fc = nn.Linear(in_features, out_features)
            return

        # Sequential head
        layers = []
        prev_out = None

        for k in fc_keys:
            idx = int(k.split(".")[1])
            w = state[k]
            out_f, in_f = w.shape

            layers.append(nn.Linear(in_f, out_f))

            # 마지막 Linear이 아니면 ReLU + Dropout
            if k != fc_keys[-1]:
                layers.append(nn.ReLU(inplace=True))
                p = float(hpo_params.get(f"dropout_l{idx//3}", hpo_params.get("dropout", 0.0)))
                layers.append(nn.Dropout(p))

        model.fc = nn.Sequential(*layers)


    _infer_and_set_fc(model)

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, class_to_idx


# =============================================================================
# TEST-DATASET: 폴더 하나에 이미지 + CSV 라벨
# =============================================================================
class CSVMappedImageDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        csv_path: str,
        class_to_idx: Dict[str, int],
        transform: transforms.Compose,
    ):
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform

        self.items: List[Tuple[str, int]] = []
        self._load_items()

    def _load_items(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"test_labels.csv not found: {self.csv_path}")

        existing_files = set()
        for fn in os.listdir(self.root_dir):
            p = os.path.join(self.root_dir, fn)
            if os.path.isfile(p) and is_valid_image(p):
                existing_files.add(fn)

        with open(self.csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = [c.strip().lower() for c in (reader.fieldnames or [])]
            if "filename" not in fieldnames:
                raise ValueError(f"CSV must have 'filename' column. got: {reader.fieldnames}")
            if "label" not in fieldnames and "class" not in fieldnames and "y" not in fieldnames:
                raise ValueError(f"CSV must have 'label' (or class/y) column. got: {reader.fieldnames}")

            fn_key = reader.fieldnames[fieldnames.index("filename")]
            if "label" in fieldnames:
                lb_key = reader.fieldnames[fieldnames.index("label")]
            elif "class" in fieldnames:
                lb_key = reader.fieldnames[fieldnames.index("class")]
            else:
                lb_key = reader.fieldnames[fieldnames.index("y")]

            missing = 0
            for row in reader:
                filename = str(row.get(fn_key, "")).strip()
                if not filename:
                    continue
                if filename not in existing_files:
                    missing += 1
                    continue

                raw_label = str(row.get(lb_key, "")).strip()
                if raw_label == "":
                    continue

                if raw_label.isdigit():
                    y = int(raw_label)
                else:
                    if raw_label not in self.class_to_idx:
                        raise ValueError(
                            f"Label '{raw_label}' not in class_to_idx. "
                            f"Known classes: {list(self.class_to_idx.keys())}"
                        )
                    y = int(self.class_to_idx[raw_label])

                self.items.append((os.path.join(self.root_dir, filename), y))

        if len(self.items) == 0:
            raise RuntimeError(f"No valid test items found. (root={self.root_dir}, csv={self.csv_path})")

        if missing > 0:
            print(f"[Warn] {missing} rows in CSV were skipped (file not found in folder).")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, y, os.path.basename(path), path


# =============================================================================
# Transform: 체크포인트의 transform_config를 최대한 존중
# =============================================================================
def build_test_transform_from_ckpt(ckpt: Dict) -> transforms.Compose:
    cfg = ckpt.get("transform_config", {}) or {}
    input_size = int(cfg.get("input_size", 224))
    mean = cfg.get("mean", [0.485, 0.456, 0.406])
    std = cfg.get("std", [0.229, 0.224, 0.225])

    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


# =============================================================================
# Soft-voting ensemble: 확률 평균
# =============================================================================
@torch.no_grad()
def predict_proba_single(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return torch.softmax(logits, dim=1)


@torch.no_grad()
def _predict_proba_ensemble_prob(models, x):
    probs_sum = None
    for m in models:
        p = predict_proba_single(m, x)
        probs_sum = p if probs_sum is None else (probs_sum + p)
    return probs_sum / float(len(models))

@torch.no_grad()
def _predict_proba_ensemble_logit(models, x):
    logits = [m(x) for m in models]
    logits_mean = torch.mean(torch.stack(logits), dim=0)
    return torch.softmax(logits_mean, dim=1)

@torch.no_grad()
def predict_proba_ensemble(models: Sequence[nn.Module], x: torch.Tensor) -> torch.Tensor:
    if ENSEMBLE_MODE == "prob":
        return _predict_proba_ensemble_prob(models, x)
    elif ENSEMBLE_MODE == "logit":
        return _predict_proba_ensemble_logit(models, x)
    else:
        raise ValueError(f"Unknown ENSEMBLE_MODE: {ENSEMBLE_MODE}")



# =============================================================================
# Overfitting Check #1: fold별 train/val score + K-fold val 평균/표준편차
# (checkpoint history 사용, 재학습 없음)
# =============================================================================
def _history_get_list(history: Any, key: str) -> Optional[List[float]]:
    if isinstance(history, dict) and key in history and isinstance(history[key], list):
        return history[key]
    return None


def extract_fold_scores_from_history(ckpt: Dict) -> Dict[str, Any]:
    history = ckpt.get("history", None)
    if history is None:
        return {"ok": False, "reason": "history not found in checkpoint"}

    train_acc = _history_get_list(history, "train_acc")
    val_acc = _history_get_list(history, "val_acc")
    train_loss = _history_get_list(history, "train_loss")
    val_loss = _history_get_list(history, "val_loss")

    if not (train_acc and val_acc):
        return {"ok": False, "reason": "history missing train_acc/val_acc"}

    best_epoch = int(np.argmax(np.array(val_acc)))  # 0-index

    out = {
        "ok": True,
        "fold": int(ckpt.get("fold", -1)),
        "best_epoch_1idx": best_epoch + 1,
        "best_val_acc": float(val_acc[best_epoch]),
        "best_train_acc_at_best": float(train_acc[best_epoch]) if len(train_acc) > best_epoch else float("nan"),
        "final_train_acc": float(train_acc[-1]),
        "final_val_acc": float(val_acc[-1]),
    }

    if train_loss and len(train_loss) > best_epoch:
        out["train_loss_at_best"] = float(train_loss[best_epoch])
    if val_loss and len(val_loss) > best_epoch:
        out["val_loss_at_best"] = float(val_loss[best_epoch])
    if train_loss:
        out["final_train_loss"] = float(train_loss[-1])
    if val_loss:
        out["final_val_loss"] = float(val_loss[-1])

    return out



def save_overfit_cv_history_summary(ckpt_paths: List[str], out_dir: str, model_tag: str) -> Tuple[List[str], str]:
    """Save fold train/val summary from checkpoint history and return printable lines + csv path."""
    os.makedirs(out_dir, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for p in ckpt_paths:
        ckpt = load_checkpoint(p, map_location="cpu")
        row = extract_fold_scores_from_history(ckpt)
        row["ckpt_path"] = p
        rows.append(row)

    save_path = os.path.join(out_dir, f"overfit_cv_history_{model_tag}.csv")

    ok_rows = [r for r in rows if r.get("ok")]
    lines: List[str] = []

    if len(ok_rows) == 0:
        # write minimal csv
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ckpt_path", "ok", "reason"])
            for r in rows:
                w.writerow([r.get("ckpt_path", ""), r.get("ok", False), r.get("reason", "")])
        lines.append("=" * 80)
        lines.append(f"[overfit:{model_tag}] Fold train/val (best epoch by max val_acc)")
        lines.append("  - (skipped) no valid history found in checkpoints.")
        lines.append("=" * 80)
        return lines, save_path

    val_list = [float(r["best_val_acc"]) for r in ok_rows]
    val_mean = float(np.mean(val_list))
    val_std = float(np.std(val_list, ddof=1)) if len(val_list) > 1 else 0.0

    # printable lines
    lines.append("=" * 80)
    lines.append(f"[overfit:{model_tag}] Fold train/val (best epoch by max val_acc)")
    for r in ok_rows:
        fold_disp = r.get("fold", -1)
        if fold_disp == -1:
            base = os.path.basename(r["ckpt_path"])
            m = re.search(r"fold(\d+)", base)
            fold_disp = int(m.group(1)) if m else -1
        lines.append(
            f"  - fold{fold_disp}: best_epoch={r['best_epoch_1idx']}, "
            f"train_acc@best={r['best_train_acc_at_best']:.4f}, val_acc@best={r['best_val_acc']:.4f}"
        )
    lines.append(f"  - ensemble: K-fold val_acc mean={val_mean:.4f}, std={val_std:.4f}")
    lines.append("=" * 80)

    # save csv
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "fold", "ckpt_path", "best_epoch",
            "train_acc_at_best", "val_acc_at_best",
            "train_loss_at_best", "val_loss_at_best",
            "final_train_acc", "final_val_acc",
            "final_train_loss", "final_val_loss",
            "ok", "reason"
        ])
        for r in rows:
            fold_disp = r.get("fold", -1)
            if fold_disp == -1:
                base = os.path.basename(r["ckpt_path"])
                m = re.search(r"fold(\d+)", base)
                fold_disp = int(m.group(1)) if m else -1

            w.writerow([
                fold_disp,
                r.get("ckpt_path", ""),
                r.get("best_epoch_1idx", ""),
                r.get("best_train_acc_at_best", ""),
                r.get("best_val_acc", ""),
                r.get("train_loss_at_best", ""),
                r.get("val_loss_at_best", ""),
                r.get("final_train_acc", ""),
                r.get("final_val_acc", ""),
                r.get("final_train_loss", ""),
                r.get("final_val_loss", ""),
                r.get("ok", False),
                r.get("reason", ""),
            ])
        w.writerow([])
        w.writerow(["Kfold_val_mean", val_mean])
        w.writerow(["Kfold_val_std", val_std])

    return lines, save_path



def compute_multiclass_ovr_roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """Compute one-vs-rest ROC/AUC for multiclass classification with safe fallbacks.

    A class is considered valid only when both positive and negative samples exist in y_true.
    """
    n_classes = len(class_names)
    result: Dict[str, Any] = {
        "macro_auc": float("nan"),
        "valid_class_count": 0,
        "per_class": [],
        "macro_curve": None,
    }

    if len(y_true) == 0 or y_prob.size == 0 or y_prob.shape[1] != n_classes:
        return result

    classes = np.arange(n_classes)
    y_true_bin = label_binarize(y_true, classes=classes)
    if y_true_bin.ndim == 1:
        y_true_bin = y_true_bin.reshape(-1, 1)

    per_class = []
    valid_aucs = []
    valid_curves = []

    for cls_idx, cls_name in enumerate(class_names):
        y_cls = y_true_bin[:, cls_idx]
        pos_count = int(np.sum(y_cls == 1))
        neg_count = int(np.sum(y_cls == 0))
        row: Dict[str, Any] = {
            "class_idx": cls_idx,
            "class_name": cls_name,
            "n_pos": pos_count,
            "n_neg": neg_count,
            "auc": float("nan"),
            "is_valid": False,
            "reason": "",
            "fpr": np.array([], dtype=float),
            "tpr": np.array([], dtype=float),
        }

        if pos_count == 0:
            row["reason"] = "no_positive_sample"
            per_class.append(row)
            continue
        if neg_count == 0:
            row["reason"] = "no_negative_sample"
            per_class.append(row)
            continue

        try:
            auc_value = float(roc_auc_score(y_cls, y_prob[:, cls_idx]))
            fpr, tpr, _ = roc_curve(y_cls, y_prob[:, cls_idx])
            row.update({
                "auc": auc_value,
                "is_valid": True,
                "reason": "ok",
                "fpr": fpr,
                "tpr": tpr,
            })
            valid_aucs.append(auc_value)
            valid_curves.append((fpr, tpr))
        except ValueError as e:
            row["reason"] = f"roc_failed:{e}"

        per_class.append(row)

    result["per_class"] = per_class
    result["valid_class_count"] = len(valid_aucs)

    if valid_aucs:
        result["macro_auc"] = float(np.mean(valid_aucs))
        grid = np.linspace(0.0, 1.0, 1001)
        mean_tpr = np.zeros_like(grid)
        for fpr, tpr in valid_curves:
            mean_tpr += np.interp(grid, fpr, tpr)
        mean_tpr /= float(len(valid_curves))
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        result["macro_curve"] = {
            "fpr": grid,
            "tpr": mean_tpr,
            "auc": float(auc(grid, mean_tpr)),
        }

    return result


def save_roc_artifacts(roc_info: Dict[str, Any], out_dir: str, tag: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "roc_auc_by_class.csv")
    fig_path = os.path.join(out_dir, "roc_curve_macro.png")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_idx", "class_name", "auc", "n_pos", "n_neg", "is_valid", "reason"])
        for row in roc_info.get("per_class", []):
            auc_v = row.get("auc", float("nan"))
            auc_str = f"{float(auc_v):.6f}" if np.isfinite(float(auc_v)) else "nan"
            w.writerow([
                row.get("class_idx", ""),
                row.get("class_name", ""),
                auc_str,
                row.get("n_pos", 0),
                row.get("n_neg", 0),
                bool(row.get("is_valid", False)),
                row.get("reason", ""),
            ])
        w.writerow([])
        macro_auc = roc_info.get("macro_auc", float("nan"))
        macro_auc_str = f"{float(macro_auc):.6f}" if np.isfinite(float(macro_auc)) else "nan"
        w.writerow(["macro_auc", "macro", macro_auc_str, "", "", "", f"valid_class_count={roc_info.get('valid_class_count', 0)}"])

    fig, ax = plt.subplots(figsize=(6, 6))
    macro_curve = roc_info.get("macro_curve")
    if macro_curve is not None:
        ax.plot(macro_curve["fpr"], macro_curve["tpr"], label=f"macro ROC (AUC={roc_info['macro_auc']:.4f})")
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "No valid class for ROC/AUC", ha="center", va="center", transform=ax.transAxes)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Macro ROC Curve ({tag})")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    return csv_path, fig_path

def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Dict[str, Any]:
    conf = y_prob.max(axis=1)
    correct = (y_pred == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_stats = []
    ece = 0.0
    n = len(conf)
    for b in range(n_bins):
        mask = bin_ids == b
        cnt = int(mask.sum())
        lower, upper = float(bins[b]), float(bins[b + 1])
        if cnt == 0:
            bin_stats.append({"bin": b, "lower": lower, "upper": upper, "count": 0, "conf": 0.0, "acc": 0.0})
            continue
        acc_b = float(correct[mask].mean())
        conf_b = float(conf[mask].mean())
        ece += (cnt / n) * abs(acc_b - conf_b)
        bin_stats.append({"bin": b, "lower": lower, "upper": upper, "count": cnt, "conf": conf_b, "acc": acc_b})

    return {"ece": float(ece), "bin_stats": bin_stats}



def save_reliability_diagram(ece_result: Dict[str, Any], out_dir: str, model_tag: str) -> Tuple[str, str]:
    """Save calibration bins CSV + reliability diagram PNG. Returns (csv_path, fig_path)."""
    os.makedirs(out_dir, exist_ok=True)
    bin_stats = ece_result["bin_stats"]
    ece = ece_result["ece"]

    csv_path = os.path.join(out_dir, f"calibration_bins_{model_tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin", "lower", "upper", "count", "avg_conf", "avg_acc"])
        for b in bin_stats:
            w.writerow([b["bin"], b["lower"], b["upper"], b["count"], b["conf"], b["acc"]])
        w.writerow([])
        w.writerow(["ECE", ece])

    lowers = np.array([b["lower"] for b in bin_stats], dtype=float)
    uppers = np.array([b["upper"] for b in bin_stats], dtype=float)
    widths = uppers - lowers
    accs = np.array([b["acc"] for b in bin_stats], dtype=float)
    confs = np.array([b["conf"] for b in bin_stats], dtype=float)
    counts = np.array([b["count"] for b in bin_stats], dtype=int)
    nonempty = counts > 0

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")
    ax.bar(lowers[nonempty], accs[nonempty], width=widths[nonempty], align="edge", alpha=0.6, edgecolor="black", label="avg accuracy")
    ax.bar(lowers[nonempty], np.clip(confs[nonempty] - accs[nonempty], 0.0, 1.0), bottom=accs[nonempty], width=widths[nonempty], align="edge", alpha=0.35, edgecolor="black", hatch="//", label="gap")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence bin")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram ({model_tag})\nECE={ece:.4f}")

    for lower, width, acc, cnt in zip(lowers[nonempty], widths[nonempty], accs[nonempty], counts[nonempty]):
        ax.annotate(str(int(cnt)), (float(lower + width / 2.0), float(acc)), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

    ax.legend(loc="lower right")
    fig.tight_layout()
    fig_path = os.path.join(out_dir, f"reliability_{model_tag}.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    return csv_path, fig_path


@torch.no_grad()
def evaluate(
    models: List[nn.Module],
    data_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    out_dir: str,
    tag: str,
) -> Dict[str, Any]:
    """
    Evaluate either a single model (len(models)==1) or an ensemble (len(models)>1).

    Saves:
      - metrics.csv
      - classification_report.csv
      - confusion_matrix.png
      - predictions.csv
      - roc_auc_by_class.csv
      - roc_curve_macro.png

    Returns eval results + saved_paths without printing.
    """
    os.makedirs(out_dir, exist_ok=True)

    if len(models) < 1:
        raise ValueError("[evaluate] models must contain at least 1 model.")

    for m in models:
        m.eval()

    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[np.ndarray] = []
    all_files: List[str] = []
    all_paths: List[str] = []

    is_ens = len(models) > 1

    for x, y, fname, fpath in data_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_ens:
            probs = predict_proba_ensemble(models, x)
        else:
            probs = predict_proba_single(models[0], x)

        pred = probs.argmax(dim=1)

        all_true.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(pred.detach().cpu().numpy().tolist())
        all_prob.extend(probs.detach().cpu().numpy())
        all_files.extend(list(fname))
        all_paths.extend(list(fpath))

    y_true = np.array(all_true, dtype=int)
    y_pred = np.array(all_pred, dtype=int)
    y_prob = np.stack(all_prob, axis=0) if len(all_prob) > 0 else np.zeros((0, len(class_names)), dtype=np.float32)

    acc = accuracy_score(y_true, y_pred) if len(y_true) else 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if len(y_true) else 0.0
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0) if len(y_true) else 0.0
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0) if len(y_true) else 0.0
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0) if len(y_true) else 0.0

    roc_info = compute_multiclass_ovr_roc_auc(y_true, y_prob, class_names)
    roc_auc_macro = float(roc_info.get("macro_auc", float("nan")))

    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"accuracy,{acc:.6f}\n")
        f.write(f"precision_macro,{precision_macro:.6f}\n")
        f.write(f"recall_macro,{recall_macro:.6f}\n")
        f.write(f"f1_macro,{macro_f1:.6f}\n")
        f.write(f"f1_weighted,{f1_weighted:.6f}\n")
        f.write(f"roc_auc_macro,{roc_auc_macro:.6f}\n" if np.isfinite(roc_auc_macro) else "roc_auc_macro,nan\n")

    labels = list(range(len(class_names)))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    ) if len(y_true) else {}

    report_path = os.path.join(out_dir, "classification_report.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1-score", "support"])
        if report_dict:
            for cls in class_names:
                row = report_dict.get(cls, {})
                w.writerow([cls, row.get("precision", 0.0), row.get("recall", 0.0), row.get("f1-score", 0.0), row.get("support", 0)])
            w.writerow([])
            macro_row = report_dict.get("macro avg", {})
            w.writerow(["macro avg", macro_row.get("precision", 0.0), macro_row.get("recall", 0.0), macro_row.get("f1-score", 0.0), macro_row.get("support", 0)])
            w.writerow(["accuracy", acc, "", "", ""])
        else:
            w.writerow(["(empty)", 0.0, 0.0, 0.0, 0])

    cm = confusion_matrix(y_true, y_pred, labels=labels) if len(y_true) else np.zeros((len(class_names), len(class_names)), dtype=int)

    fig, ax = plt.subplots(figsize=(6, 6))
    vmax = cm.max() if cm.size and cm.max() > 0 else 1
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=vmax)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Confusion Matrix ({tag})",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = vmax / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            color = "white" if value > thresh else "black"
            ax.text(j, i, value, ha="center", va="center", color=color)

    fig.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)

    pred_path = os.path.join(out_dir, "predictions.csv")
    idx_to_class = {i: n for i, n in enumerate(class_names)}
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "path", "true_idx", "true_class", "pred_idx", "pred_class", "pred_conf"])
        for i in range(len(y_true)):
            pred_idx = int(y_pred[i])
            pred_conf = float(y_prob[i, pred_idx]) if y_prob.shape[0] > 0 else 0.0
            w.writerow([
                all_files[i],
                all_paths[i],
                int(y_true[i]),
                idx_to_class.get(int(y_true[i]), str(int(y_true[i]))),
                pred_idx,
                idx_to_class.get(pred_idx, str(pred_idx)),
                f"{pred_conf:.6f}",
            ])

    roc_csv_path, roc_fig_path = save_roc_artifacts(roc_info, out_dir=out_dir, tag=tag)

    saved_paths = {
        "metrics": metrics_path,
        "report": report_path,
        "cm": cm_path,
        "preds": pred_path,
        "roc_by_class": roc_csv_path,
        "roc_curve_macro": roc_fig_path,
    }

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "filenames": all_files,
        "paths": all_paths,
        "metrics": {
            "accuracy": acc,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": macro_f1,
            "f1_weighted": f1_weighted,
            "roc_auc_macro": roc_auc_macro,
        },
        "roc_info": roc_info,
        "confusion_matrix": cm,
        "report": report_dict,
        "saved_paths": saved_paths,
        "is_ensemble": is_ens,
        "ensemble_mode": (globals().get("ENSEMBLE_MODE", None) if is_ens else None),
    }


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._hook_handles: List[Any] = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, _input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        try:
            self._hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))  # type: ignore[attr-defined]
        except Exception:
            self._hook_handles.append(self.target_layer.register_backward_hook(backward_hook))  # fallback

    def close(self):
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._hook_handles = []

    def generate(self, input_tensor: torch.Tensor, class_idx: int, device: torch.device):
        self.model.zero_grad()
        self.gradients = None
        self.activations = None
        input_tensor = input_tensor.to(device)

        output = self.model(input_tensor)
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations
        if gradients is None or activations is None:
            raise RuntimeError("GradCAM hooks did not capture gradients/activations.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam


def overlay_cam_on_image(img: Image.Image, cam: torch.Tensor, alpha: float = 0.5) -> Image.Image:
    cam_np = cam.squeeze().cpu().numpy()
    cam_np = np.uint8(255 * cam_np)

    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam_np / 255.0)[:, :, :3]
    heatmap = np.uint8(255 * heatmap)

    img = img.convert("RGB")
    img_np = np.array(img)

    if cam_np.shape != img_np.shape[:2]:
        heatmap_img = Image.fromarray(heatmap).resize((img_np.shape[1], img_np.shape[0]), resample=Image.BILINEAR)
        heatmap = np.array(heatmap_img)

    overlay = np.uint8(alpha * heatmap + (1 - alpha) * img_np)
    return Image.fromarray(overlay)


def build_cam_selection(eval_result: Dict[str, Any], topk_high: int, topk_low: int) -> Dict[str, List[int]]:
    y_true = np.asarray(eval_result["y_true"], dtype=int)
    y_pred = np.asarray(eval_result["y_pred"], dtype=int)
    y_prob = np.asarray(eval_result["y_prob"], dtype=float)

    if len(y_true) == 0 or y_prob.size == 0:
        return {"high_conf_topk": [], "low_conf_topk": [], "correct_all": [], "incorrect_all": []}

    pred_conf = y_prob[np.arange(len(y_pred)), y_pred]
    order_desc = np.argsort(-pred_conf)
    order_asc = np.argsort(pred_conf)

    return {
        "high_conf_topk": order_desc[: min(topk_high, len(order_desc))].tolist(),
        "low_conf_topk": order_asc[: min(topk_low, len(order_asc))].tolist(),
        "correct_all": np.where(y_true == y_pred)[0].tolist(),
        "incorrect_all": np.where(y_true != y_pred)[0].tolist(),
    }


def _build_dataset_path_index(dataset: CSVMappedImageDataset) -> Dict[str, int]:
    return {path: i for i, (path, _y) in enumerate(dataset.items)}


def _safe_class_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def _draw_bottom_text_box(
    img: Image.Image,
    info: str,
    font: Optional[ImageFont.ImageFont],
) -> Image.Image:
    """이미지 하단에 어두운 배경의 텍스트 박스를 붙여 반환한다."""
    lines = info.split("\n")
    # 라인 높이 추정 (font.size가 없으면 기본값 16 사용)
    if font is not None and hasattr(font, "size"):
        line_h = font.size + 4
    else:
        line_h = 16
    padding = 4
    box_h = line_h * len(lines) + padding * 2

    # 원본 이미지 아래에 텍스트 박스 영역을 합친 새 이미지 생성
    new_img = Image.new("RGB", (img.width, img.height + box_h), (0, 0, 0))
    new_img.paste(img, (0, 0))

    draw = ImageDraw.Draw(new_img)
    draw.rectangle([0, img.height, img.width, img.height + box_h], fill=(20, 20, 20))
    y = img.height + padding
    for line in lines:
        draw.text((padding, y), line, fill=(255, 255, 255), font=font)
        y += line_h

    return new_img


def _save_cam_overlay(
    model: nn.Module,
    dataset: CSVMappedImageDataset,
    class_names: List[str],
    device: torch.device,
    output_path: str,
    ds_i: int,
    true_idx: int,
    pred_idx: int,
    pred_conf: float,
    status: str,
    grad_cam: Any,
    font: Optional[ImageFont.ImageFont],
) -> bool:
    fullpath, _ = dataset.items[ds_i]
    pil_img = Image.open(fullpath).convert("RGB")
    x1 = dataset.transform(pil_img).unsqueeze(0)
    cam = grad_cam.generate(x1, class_idx=pred_idx, device=device)
    overlay_img = overlay_cam_on_image(pil_img, cam[0])

    true_class = class_names[true_idx] if 0 <= true_idx < len(class_names) else str(true_idx)
    pred_class = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx)
    info = f"true: {true_class}, pred: {pred_class}\nconf: {pred_conf:.3f}, status: {status}"
    overlay_img = _draw_bottom_text_box(overlay_img, info, font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    overlay_img.save(output_path)
    return True


def generate_cam_by_group(
    model: nn.Module,
    eval_result: Dict[str, Any],
    dataset: CSVMappedImageDataset,
    class_names: List[str],
    device: torch.device,
    output_dir: str,
    indices: List[int],
    group_name: str,
    split_by_true_class: bool = False,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    if len(indices) == 0:
        return {"output_dir": output_dir, "saved_count": 0, "group_name": group_name}

    y_true = np.asarray(eval_result["y_true"], dtype=int)
    y_pred = np.asarray(eval_result["y_pred"], dtype=int)
    y_prob = np.asarray(eval_result["y_prob"], dtype=float)
    paths = list(eval_result["paths"])

    path_to_i = _build_dataset_path_index(dataset)
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    saved = 0
    used_names = set()

    try:
        for rank, i in enumerate(indices, start=1):
            i = int(i)
            if not (0 <= i < len(paths)):
                continue
            p = paths[i]
            if p not in path_to_i:
                continue

            ds_i = path_to_i[p]
            true_idx = int(y_true[i])
            pred_idx = int(y_pred[i])
            pred_conf = float(y_prob[i, pred_idx])
            status = "correct" if true_idx == pred_idx else "incorrect"

            true_class = class_names[true_idx] if 0 <= true_idx < len(class_names) else str(true_idx)
            safe_true = _safe_class_name(true_class)
            base = os.path.splitext(os.path.basename(p))[0]

            if split_by_true_class:
                save_dir = os.path.join(output_dir, safe_true)
                save_name = f"{status}_{safe_true}_{base}.png"
            else:
                save_dir = output_dir
                save_name = f"{rank:03d}_{status}_{safe_true}_{base}.png"

            candidate = save_name
            stem, ext = os.path.splitext(candidate)
            suffix = 2
            while os.path.join(save_dir, candidate) in used_names:
                candidate = f"{stem}_{suffix}{ext}"
                suffix += 1
            used_names.add(os.path.join(save_dir, candidate))

            ok = _save_cam_overlay(
                model=model,
                dataset=dataset,
                class_names=class_names,
                device=device,
                output_path=os.path.join(save_dir, candidate),
                ds_i=ds_i,
                true_idx=true_idx,
                pred_idx=pred_idx,
                pred_conf=pred_conf,
                status=status,
                grad_cam=grad_cam,
                font=font,
            )
            if ok:
                saved += 1
    finally:
        grad_cam.close()

    return {"output_dir": output_dir, "saved_count": saved, "group_name": group_name}


def save_single_fold_eval_summary(fold_metric_rows: List[Dict[str, Any]], out_dir: str) -> Tuple[str, str]:
    """Save raw single-fold test metrics and mean/std summary."""
    os.makedirs(out_dir, exist_ok=True)

    raw_path = os.path.join(out_dir, "single_fold_metrics_by_fold.csv")
    summary_path = os.path.join(out_dir, "single_fold_metrics_summary.csv")

    metric_names = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "roc_auc_macro",
    ]

    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold", *metric_names])
        for row in fold_metric_rows:
            w.writerow([
                row.get("fold", ""),
                *[f"{float(row.get(m, float('nan'))):.6f}" for m in metric_names],
            ])

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std", "n_folds"])
        for metric in metric_names:
            values = [float(r[metric]) for r in fold_metric_rows if metric in r]
            if len(values) == 0:
                mean_v = float("nan")
                std_v = float("nan")
                n = 0
            else:
                mean_v = float(np.mean(values))
                std_v = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                n = len(values)
            w.writerow([metric, f"{mean_v:.6f}", f"{std_v:.6f}", n])

    return raw_path, summary_path


def _build_metric_summary_rows(metric_rows: List[Dict[str, Any]], metric_names: List[str]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for metric in metric_names:
        values = []
        for r in metric_rows:
            v = r.get(metric, float("nan"))
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = float("nan")
            if np.isfinite(fv):
                values.append(fv)
        if len(values) == 0:
            rows.append([metric, "nan", "nan", 0])
        else:
            mean_v = float(np.mean(values))
            std_v = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            rows.append([metric, f"{mean_v:.6f}", f"{std_v:.6f}", len(values)])
    return rows


def save_ensemble_comparison_csv(
    fold_metric_rows: List[Dict[str, Any]],
    ensemble_metrics: Dict[str, Any],
    out_dir: str,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    compare_path = os.path.join(out_dir, "ensemble_member_vs_final.csv")
    summary_path = os.path.join(out_dir, "ensemble_member_summary.csv")
    metric_names = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "roc_auc_macro",
    ]

    with open(compare_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag", "role", "eval_mode", "ensemble_mode", *metric_names])
        for row in fold_metric_rows:
            w.writerow([
                row.get("fold", ""),
                "member_fold",
                "single",
                "",
                *[f"{float(row.get(m, float('nan'))):.6f}" if np.isfinite(float(row.get(m, float('nan')))) else "nan" for m in metric_names],
            ])
        w.writerow([
            "ensemble",
            "final_ensemble",
            "ensemble",
            ENSEMBLE_MODE,
            *[f"{float(ensemble_metrics.get(m, float('nan'))):.6f}" if np.isfinite(float(ensemble_metrics.get(m, float('nan')))) else "nan" for m in metric_names],
        ])

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "member_mean", "member_std", "n_members", "ensemble", "ensemble_minus_member_mean", "ensemble_mode"])
        summary_rows = _build_metric_summary_rows(fold_metric_rows, metric_names)
        mean_map = {r[0]: r[1] for r in summary_rows}
        for metric, mean_str, std_str, n in summary_rows:
            ens_v = float(ensemble_metrics.get(metric, float("nan")))
            ens_str = f"{ens_v:.6f}" if np.isfinite(ens_v) else "nan"
            if mean_str == "nan" or not np.isfinite(ens_v):
                delta_str = "nan"
            else:
                delta_str = f"{ens_v - float(mean_str):.6f}"
            w.writerow([metric, mean_str, std_str, n, ens_str, delta_str, ENSEMBLE_MODE])

    return compare_path, summary_path


def _run_calibration_artifacts(eval_res: Dict[str, Any], out_dir: str, tag: str, saved_logs: List[str], ece_by_tag: Dict[str, float]):
    ece_res = compute_ece(eval_res["y_true"], eval_res["y_pred"], eval_res["y_prob"], n_bins=ECE_NUM_BINS)
    ece_by_tag[tag] = float(ece_res["ece"])
    cal_csv, cal_fig = save_reliability_diagram(ece_res, out_dir=out_dir, model_tag=tag)
    saved_logs.append(f"[Saved] {tag} calibration bins: {cal_csv}")
    saved_logs.append(f"[Saved] {tag} reliability diagram: {cal_fig}")


def _run_cam_artifacts(
    model: nn.Module,
    eval_res: Dict[str, Any],
    dataset: CSVMappedImageDataset,
    class_names: List[str],
    device: torch.device,
    cam_root: str,
    tag: str,
    saved_logs: List[str],
):
    selections = build_cam_selection(eval_res, topk_high=CAM_TOPK_HIGH, topk_low=CAM_TOPK_LOW)

    high_group_name = f"high_conf_top{CAM_TOPK_HIGH}"
    low_group_name = f"low_conf_top{CAM_TOPK_LOW}"

    high = generate_cam_by_group(
        model=model, eval_result=eval_res, dataset=dataset, class_names=class_names, device=device,
        output_dir=os.path.join(cam_root, high_group_name), indices=selections["high_conf_topk"],
        group_name=high_group_name, split_by_true_class=False,
    )
    saved_logs.append(f"[Saved] {tag} CAM({high_group_name}): {high['output_dir']}  (n={high['saved_count']})")

    low = generate_cam_by_group(
        model=model, eval_result=eval_res, dataset=dataset, class_names=class_names, device=device,
        output_dir=os.path.join(cam_root, low_group_name), indices=selections["low_conf_topk"],
        group_name=low_group_name, split_by_true_class=False,
    )
    saved_logs.append(f"[Saved] {tag} CAM({low_group_name}): {low['output_dir']}  (n={low['saved_count']})")

    if CAM_SAVE_CORRECT_ALL:
        correct = generate_cam_by_group(
            model=model, eval_result=eval_res, dataset=dataset, class_names=class_names, device=device,
            output_dir=os.path.join(cam_root, "correct"), indices=selections["correct_all"],
            group_name="correct_all", split_by_true_class=True,
        )
        saved_logs.append(f"[Saved] {tag} CAM(correct_all): {correct['output_dir']}  (n={correct['saved_count']})")

    if CAM_SAVE_INCORRECT_ALL:
        incorrect = generate_cam_by_group(
            model=model, eval_result=eval_res, dataset=dataset, class_names=class_names, device=device,
            output_dir=os.path.join(cam_root, "incorrect"), indices=selections["incorrect_all"],
            group_name="incorrect_all", split_by_true_class=True,
        )
        saved_logs.append(f"[Saved] {tag} CAM(incorrect_all): {incorrect['output_dir']}  (n={incorrect['saved_count']})")


def _print_final_summary(
    saved_logs: List[str],
    overfit_score_lines: List[str],
    ece_by_tag: Dict[str, float],
    eval_by_tag: Dict[str, Dict[str, float]],
    model_tags: List[str],
    eval_mode: str,
):
    print("=" * 50)
    for s in saved_logs:
        print(s)
    print("=" * 30)
    print()

    print("=" * 30)
    print(f"[EvalMode] {eval_mode}")
    if eval_mode == "ensemble":
        print(f"[Ensemble] mode={ENSEMBLE_MODE}")
    elif eval_mode == "single" and not IS_FOLD:
        print("[SingleMode] IS_FOLD=False → filename stem used as model tag")
    print("=" * 30)

    for ln in overfit_score_lines:
        print(ln)
    print()

    print("=" * 30)
    if ENABLE_CALIBRATION:
        print(f"[overfit:cal] ECE (top-label multiclass, bins={ECE_NUM_BINS})")
        for tag in model_tags:
            if tag in ece_by_tag:
                print(f"  - {tag}: ECE={ece_by_tag[tag]:.4f}")
        if "ensemble" in ece_by_tag:
            print(f"  - ensemble: ECE={ece_by_tag['ensemble']:.4f}")
    else:
        print("[overfit:cal] skipped (ENABLE_CALIBRATION=False)")
    print("=" * 30)
    print()

    print("=" * 30)
    print("[Eval] acc macro_f1 roc_auc_macro")
    for tag in model_tags:
        if tag in eval_by_tag:
            print(f"[Eval] {tag}: acc={eval_by_tag[tag]['acc']:.4f}  macro_f1={eval_by_tag[tag]['macro_f1']:.4f}  roc_auc_macro={eval_by_tag[tag].get('roc_auc_macro', float('nan')):.4f}")
    if "ensemble" in eval_by_tag:
        print(f"[Eval] ensemble: acc={eval_by_tag['ensemble']['acc']:.4f}  macro_f1={eval_by_tag['ensemble']['macro_f1']:.4f}  roc_auc_macro={eval_by_tag['ensemble'].get('roc_auc_macro', float('nan')):.4f}")
    print("=" * 30)



def run_all_evaluations(ckpt_paths: List[str]):
    """Run evaluation according to EVAL_MODE.

    - single: fold-wise evaluation + optional calibration + optional CAM + fold mean/std summary
    - ensemble: ensemble evaluation + optional calibration + optional CAM
    """
    if EVAL_MODE not in {"single", "ensemble"}:
        raise ValueError(f"Unknown EVAL_MODE: {EVAL_MODE}. Expected 'single' or 'ensemble'.")

    out_root = OUT_DIR
    os.makedirs(out_root, exist_ok=True)

    saved_logs: List[str] = []
    ece_by_tag: Dict[str, float] = {}
    eval_by_tag: Dict[str, Dict[str, float]] = {}
    fold_metric_rows: List[Dict[str, Any]] = []

    ckpt0 = load_checkpoint(ckpt_paths[0], map_location=DEVICE)
    test_tf = build_test_transform_from_ckpt(ckpt0)

    class_to_idx = ckpt0["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    test_ds = CSVMappedImageDataset(TEST_DIR, TEST_LABELS_CSV, class_to_idx, test_tf)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    fold_models: List[nn.Module] = []
    for p in ckpt_paths:
        ckpt = load_checkpoint(p, map_location=DEVICE)
        model, _ = build_model_from_checkpoint(ckpt, device=DEVICE)
        fold_models.append(model)

    overfit_score_lines, overfit_csv = save_overfit_cv_history_summary(
        ckpt_paths=ckpt_paths, out_dir=out_root, model_tag="score"
    )
    saved_logs.append(f"[Saved] overfit(history) summary: {overfit_csv}")
    if ENABLE_CALIBRATION:
        saved_logs.append(f"[Info] calibration/reliability uses top-label multiclass confidence (ECE_NUM_BINS={ECE_NUM_BINS})")
    else:
        saved_logs.append("[Info] calibration/reliability skipped by ENABLE_CALIBRATION=False")
    if EVAL_MODE == "single" and not IS_FOLD:
        saved_logs.append("[Info] single mode: IS_FOLD=False → filename stem used as model tag")
    if ENABLE_CAM and EVAL_MODE == "ensemble":
        saved_logs.append("[Info] ensemble CAM uses representative fold model (fold1) for visualization")
    elif not ENABLE_CAM:
        saved_logs.append("[Info] CAM generation skipped by ENABLE_CAM=False")

    if EVAL_MODE == "single":
        for i, (model, ckpt_p) in enumerate(zip(fold_models, ckpt_paths), start=1):
            tag = f"fold{i}" if IS_FOLD else Path(ckpt_p).stem
            fold_dir = os.path.join(out_root, tag)
            os.makedirs(fold_dir, exist_ok=True)

            eval_res = evaluate(
                models=[model],
                data_loader=test_loader,
                device=DEVICE,
                class_names=class_names,
                out_dir=fold_dir,
                tag=tag,
            )

            sp = eval_res.get("saved_paths", {})
            saved_logs.append(f"[Saved] {tag} cm    : {sp.get('cm','')}")
            saved_logs.append(f"[Saved] {tag} preds : {sp.get('preds','')}")
            saved_logs.append(f"[Saved] {tag} metrics: {sp.get('metrics','')}")
            saved_logs.append(f"[Saved] {tag} report : {sp.get('report','')}")
            saved_logs.append(f"[Saved] {tag} roc(csv): {sp.get('roc_by_class','')}")
            saved_logs.append(f"[Saved] {tag} roc(fig): {sp.get('roc_curve_macro','')}")

            if ENABLE_CALIBRATION:
                _run_calibration_artifacts(eval_res, out_dir=fold_dir, tag=tag, saved_logs=saved_logs, ece_by_tag=ece_by_tag)

            metric_row = {"fold": tag, **eval_res["metrics"]}
            fold_metric_rows.append(metric_row)
            eval_by_tag[tag] = {
                "acc": float(eval_res["metrics"]["accuracy"]),
                "macro_f1": float(eval_res["metrics"]["f1_macro"]),
                "roc_auc_macro": float(eval_res["metrics"].get("roc_auc_macro", float("nan"))),
            }

            if ENABLE_CAM:
                cam_root = os.path.join(fold_dir, "cam_results")
                _run_cam_artifacts(
                    model=model,
                    eval_res=eval_res,
                    dataset=test_ds,
                    class_names=class_names,
                    device=DEVICE,
                    cam_root=cam_root,
                    tag=tag,
                    saved_logs=saved_logs,
                )

        raw_csv, summary_csv = save_single_fold_eval_summary(fold_metric_rows, out_root)
        saved_logs.append(f"[Saved] single fold raw metrics: {raw_csv}")
        saved_logs.append(f"[Saved] single fold summary : {summary_csv}")

    elif EVAL_MODE == "ensemble":
        ens_tag = "ensemble"
        ens_dir = os.path.join(out_root, ens_tag)
        os.makedirs(ens_dir, exist_ok=True)

        eval_res_ens = evaluate(
            models=fold_models,
            data_loader=test_loader,
            device=DEVICE,
            class_names=class_names,
            out_dir=ens_dir,
            tag=ens_tag,
        )

        sp = eval_res_ens.get("saved_paths", {})
        saved_logs.append(f"[Saved] {ens_tag} cm    : {sp.get('cm','')}")
        saved_logs.append(f"[Saved] {ens_tag} preds : {sp.get('preds','')}")
        saved_logs.append(f"[Saved] {ens_tag} metrics: {sp.get('metrics','')}")
        saved_logs.append(f"[Saved] {ens_tag} report : {sp.get('report','')}")
        saved_logs.append(f"[Saved] {ens_tag} roc(csv): {sp.get('roc_by_class','')}")
        saved_logs.append(f"[Saved] {ens_tag} roc(fig): {sp.get('roc_curve_macro','')}")

        if ENABLE_CALIBRATION:
            _run_calibration_artifacts(eval_res_ens, out_dir=ens_dir, tag=ens_tag, saved_logs=saved_logs, ece_by_tag=ece_by_tag)

        eval_by_tag[ens_tag] = {
            "acc": float(eval_res_ens["metrics"]["accuracy"]),
            "macro_f1": float(eval_res_ens["metrics"]["f1_macro"]),
            "roc_auc_macro": float(eval_res_ens["metrics"].get("roc_auc_macro", float("nan"))),
        }

        member_dir = os.path.join(ens_dir, "member_folds")
        os.makedirs(member_dir, exist_ok=True)
        for i, (model, ckpt_p) in enumerate(zip(fold_models, ckpt_paths), start=1):
            member_tag = f"fold{i}" if IS_FOLD else Path(ckpt_p).stem
            member_out_dir = os.path.join(member_dir, member_tag)
            member_eval = evaluate(
                models=[model],
                data_loader=test_loader,
                device=DEVICE,
                class_names=class_names,
                out_dir=member_out_dir,
                tag=member_tag,
            )
            fold_metric_rows.append({"fold": member_tag, **member_eval["metrics"]})
            eval_by_tag[member_tag] = {
                "acc": float(member_eval["metrics"]["accuracy"]),
                "macro_f1": float(member_eval["metrics"]["f1_macro"]),
                "roc_auc_macro": float(member_eval["metrics"].get("roc_auc_macro", float("nan"))),
            }
            msp = member_eval.get("saved_paths", {})
            saved_logs.append(f"[Saved] ensemble member {member_tag} cm    : {msp.get('cm','')}")
            saved_logs.append(f"[Saved] ensemble member {member_tag} preds : {msp.get('preds','')}")
            saved_logs.append(f"[Saved] ensemble member {member_tag} metrics: {msp.get('metrics','')}")
            saved_logs.append(f"[Saved] ensemble member {member_tag} report : {msp.get('report','')}")
            saved_logs.append(f"[Saved] ensemble member {member_tag} roc(csv): {msp.get('roc_by_class','')}")
            saved_logs.append(f"[Saved] ensemble member {member_tag} roc(fig): {msp.get('roc_curve_macro','')}")

        comp_csv, comp_summary_csv = save_ensemble_comparison_csv(
            fold_metric_rows=fold_metric_rows,
            ensemble_metrics=eval_res_ens["metrics"],
            out_dir=ens_dir,
        )
        saved_logs.append(f"[Saved] ensemble compare raw : {comp_csv}")
        saved_logs.append(f"[Saved] ensemble compare summary: {comp_summary_csv}")

        if ENABLE_CAM:
            cam_root = os.path.join(ens_dir, "cam_results")
            _run_cam_artifacts(
                model=fold_models[0],
                eval_res=eval_res_ens,
                dataset=test_ds,
                class_names=class_names,
                device=DEVICE,
                cam_root=cam_root,
                tag=ens_tag,
                saved_logs=saved_logs,
            )

    if EVAL_MODE == "single":
        model_tags = [f"fold{i+1}" for i in range(len(ckpt_paths))] if IS_FOLD \
                     else [Path(p).stem for p in ckpt_paths]
    else:  # ensemble
        model_tags = [f"fold{i+1}" for i in range(len(ckpt_paths))]

    _print_final_summary(
        saved_logs=saved_logs,
        overfit_score_lines=overfit_score_lines,
        ece_by_tag=ece_by_tag,
        eval_by_tag=eval_by_tag,
        model_tags=model_tags,
        eval_mode=EVAL_MODE,
    )


def main():
    global OUT_DIR
    set_seed(SEED)
    OUT_DIR = resolve_out_dir()
    os.makedirs(OUT_DIR, exist_ok=True)

    # 체크포인트 경로 결정: 명시적 지정 우선, 없으면 CKPT_PATH 자동 탐색
    if ENSEMBLE_CKPT_PATHS is not None:
        ckpt_paths = ENSEMBLE_CKPT_PATHS
        missing = [p for p in ckpt_paths if not os.path.exists(p)]
        if missing:
            print(f"[Error] missing checkpoints: {missing}")
            return
    else:
        ckpt_paths = _discover_checkpoints(CKPT_PATH)
        if not ckpt_paths:
            print(f"[Error] No .pth files found in: {CKPT_PATH}")
            return
        print(f"[Info] Auto-discovered {len(ckpt_paths)} checkpoint(s) in '{CKPT_PATH}':")
        for p in ckpt_paths:
            print(f"  {p}")

    run_all_evaluations(ckpt_paths)


if __name__ == "__main__":
    main()