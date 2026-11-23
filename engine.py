"""Training and evaluation loops (AMP-aware, Windows friendly)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import compute_metrics


def make_scaler(device: torch.device, amp: bool):
    """GradScaler for mixed precision, or ``None`` when AMP is off."""
    if not amp or device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):  # older PyTorch
        return torch.cuda.amp.GradScaler()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    amp: bool = True,
    grad_clip: float = 1.0,
    limit_batches: int = 0,
) -> Dict[str, float]:
    """One full pass over the training set."""
    model.train()
    n_batches = len(loader) if limit_batches <= 0 else min(limit_batches, len(loader))

    running_loss, correct, seen = 0.0, 0, 0
    bar = tqdm(
        total=n_batches,
        desc=f"train {epoch:02d}/{total_epochs:02d}",
        leave=False,
        dynamic_ncols=True,
    )

    for step, (images, targets) in enumerate(loader, start=1):
        if step > n_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"
        ):
            logits = model(images)
            loss = criterion(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = targets.size(0)
        running_loss += float(loss.detach()) * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        seen += batch_size

        bar.update(1)
        bar.set_postfix(
            loss=f"{running_loss / max(1, seen):.4f}",
            acc=f"{correct / max(1, seen):.4f}",
        )

    bar.close()
    return {
        "loss": running_loss / max(1, seen),
        "accuracy": correct / max(1, seen),
        "samples": float(seen),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str],
    tta: bool = False,
    limit_batches: int = 0,
    desc: str = "val",
    predictions_csv: str | Path | None = None,
) -> Dict[str, Any]:
    """Evaluate the model.

    Args:
        tta: average softmax probabilities of the image and its horizontal
            flip.  Costs one extra forward pass, typically worth a few tenths
            of a percent of accuracy -- use it for the final test report.
        predictions_csv: optional path to write per-image predictions.
    """
    model.eval()
    n_batches = len(loader) if limit_batches <= 0 else min(limit_batches, len(loader))

    dataset = getattr(loader, "dataset", None)
    samples = getattr(dataset, "samples", None)

    y_true: List[int] = []
    y_pred: List[int] = []
    all_probs: List[List[float]] = []
    rows: List[Tuple[str, str, str, float]] = []
    running_loss, seen = 0.0, 0
    offset = 0

    bar = tqdm(total=n_batches, desc=desc, leave=False, dynamic_ncols=True)
    for step, (images, targets) in enumerate(loader, start=1):
        if step > n_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        probs = logits.float().softmax(dim=1)
        if tta:
            flipped = model(torch.flip(images, dims=[3])).float().softmax(dim=1)
            probs = (probs + flipped) / 2.0

        confidences, predictions = probs.max(dim=1)
        batch_size = targets.size(0)

        y_true.extend(targets.tolist())
        y_pred.extend(predictions.tolist())
        all_probs.extend(probs.cpu().tolist())

        if samples is not None:
            for i in range(batch_size):
                index = offset + i
                if index < len(samples):
                    sample_path = Path(samples[index][0])
                    rows.append(
                        (
                            f"{sample_path.parent.name}/{sample_path.name}",
                            class_names[int(targets[i])],
                            class_names[int(predictions[i])],
                            float(confidences[i]),
                        )
                    )

        offset += batch_size
        running_loss += float(loss) * batch_size
        seen += batch_size
        bar.update(1)

    bar.close()

    if not y_true:
        raise RuntimeError(
            "No samples were evaluated. Check the split directory and --limit-batches."
        )

    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["loss"] = running_loss / max(1, seen)

    if predictions_csv is not None:
        path = Path(predictions_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("file,true_label,predicted_label,confidence,correct\n")
            for name, true_label, predicted_label, confidence in rows:
                handle.write(
                    f"{name},{true_label},{predicted_label},{confidence:.6f},"
                    f"{int(true_label == predicted_label)}\n"
                )

    return metrics
