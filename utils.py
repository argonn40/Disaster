"""Shared helpers: seeding, device selection, metrics, plots and checkpoints."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn

CLASS_COLORS = [
    "#2E86AB",
    "#E4572E",
    "#17B890",
    "#B07AA1",
    "#F4A259",
    "#5B5F97",
]


def set_seed(seed: int = 42) -> None:
    """Make runs as reproducible as PyTorch/CUDA allow."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN slows training down noticeably; benchmark=False is
    # only needed if bit-exact reproducibility is required.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve 'auto' / 'cuda' / 'cpu' into a usable ``torch.device``."""
    requested = (requested or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(
            "[warn] CUDA was requested but is not available "
            "(no GPU, or a CPU-only PyTorch build). Falling back to CPU."
        )
        return torch.device("cpu")
    return torch.device(requested)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]
) -> Dict[str, Any]:
    """Accuracy, macro/weighted precision-recall-F1 and a confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    labels = list(range(len(class_names)))

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0
    )
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0
    )
    p_cls, r_cls, f_cls, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0
    )

    per_class = {
        name: {
            "precision": float(p_cls[i]),
            "recall": float(r_cls[i]),
            "f1": float(f_cls[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_w),
        "weighted_recall": float(recall_w),
        "weighted_f1": float(f1_w),
        "num_samples": int(len(y_true_arr)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    out_path: str | Path,
    title: str = "Confusion matrix",
) -> Path:
    """Save a confusion-matrix heatmap with counts and row percentages."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(matrix, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = np.divide(cm, np.maximum(row_sums, 1)) * 100.0

    fig, ax = plt.subplots(figsize=(1.15 * len(class_names) + 3.2, 0.95 * len(class_names) + 2.6))
    im = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}\n{row_pct[i, j]:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if row_pct[i, j] > 55 else "#1b1b1b",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% of true class")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_training_history(history: Iterable[Dict[str, Any]], out_path: str | Path) -> Path:
    """Loss / accuracy / macro-F1 curves for the training run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(history)
    if not rows:
        raise ValueError("Empty training history")
    epochs = [row["epoch"] for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(epochs, [r["train_loss"] for r in rows], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in rows], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")

    axes[1].plot(epochs, [r["train_acc"] for r in rows], label="train")
    axes[1].plot(epochs, [r["val_acc"] for r in rows], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")

    axes[2].plot(epochs, [r["val_macro_f1"] for r in rows], color="#E4572E")
    axes[2].set_title("Validation macro-F1")
    axes[2].set_xlabel("epoch")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="best") if ax is not axes[2] else None

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    arch: str,
    class_names: Sequence[str],
    img_size: int,
    dropout: float,
    epoch: int,
    metrics: Dict[str, Any] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Path:
    """Save everything needed to rebuild the model for inference."""
    payload: Dict[str, Any] = {
        "format_version": 1,
        "arch": arch,
        "class_names": list(class_names),
        "img_size": int(img_size),
        "dropout": float(dropout),
        "epoch": int(epoch),
        "metrics": metrics or {},
        "model_state": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def format_metrics(metrics: Dict[str, Any], class_names: Sequence[str]) -> str:
    """Human-readable summary used by evaluate.py and train.py."""
    lines = [
        f"  accuracy        : {metrics['accuracy'] * 100:6.2f} %",
        f"  macro precision : {metrics['macro_precision'] * 100:6.2f} %",
        f"  macro recall    : {metrics['macro_recall'] * 100:6.2f} %",
        f"  macro F1        : {metrics['macro_f1'] * 100:6.2f} %",
        f"  weighted F1     : {metrics['weighted_f1'] * 100:6.2f} %",
        f"  samples         : {metrics['num_samples']}",
        "",
        f"  {'class':<12}{'precision':>10}{'recall':>9}{'f1':>9}{'support':>9}",
    ]
    for name in class_names:
        row = metrics["per_class"][name]
        lines.append(
            f"  {name:<12}{row['precision'] * 100:9.2f}%"
            f"{row['recall'] * 100:8.2f}%{row['f1'] * 100:8.2f}%{row['support']:9d}"
        )
    return "\n".join(lines)
