"""Evaluate a trained checkpoint on the Val or Test split.

Example:
    python evaluate.py --checkpoint runs/resnet18/best.pt --split Test

Writes into ``<checkpoint folder>/eval_<split>/``:
    metrics.json              accuracy, macro/weighted precision-recall-F1
    per_class_metrics.csv     precision / recall / F1 per disaster type
    classification_report.txt sklearn report
    confusion_matrix.png      counts + row percentages
    predictions.csv           one row per image (easy to inspect mistakes)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from dataset import create_eval_loader
from engine import evaluate
from model import build_resnet
from utils import (
    compute_metrics,
    format_metrics,
    plot_confusion_matrix,
    resolve_device,
    save_json,
    set_seed,
    timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained disaster-classification checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="runs/resnet18/best.pt")
    parser.add_argument("--data-dir", default="data/aider")
    parser.add_argument("--split", default="Test", help="Train | Val | Test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--img-size",
        type=int,
        default=0,
        help="0 = use the size the model was trained with (recommended)",
    )
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="average with a horizontally flipped copy (a little more accurate)",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit-batches", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state" not in checkpoint or "class_names" not in checkpoint:
        raise ValueError(
            f"'{path}' is not a checkpoint written by train.py "
            "(expected keys: 'model_state', 'class_names')."
        )
    return checkpoint


def main() -> None:
    args = parse_args()
    set_seed(42)
    device = resolve_device(args.device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' not found. Train first:\n"
            "  python train.py --data-dir data/aider"
        )

    checkpoint = load_checkpoint(checkpoint_path, device)
    arch = checkpoint.get("arch", "resnet18")
    dropout = float(checkpoint.get("dropout", 0.0))
    class_names = list(checkpoint["class_names"])
    img_size = args.img_size or int(checkpoint.get("img_size", 224))

    model = build_resnet(arch, num_classes=len(class_names), dropout=dropout)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    loader, dataset_classes, _ = create_eval_loader(
        data_dir=args.data_dir,
        split=args.split,
        img_size=img_size,
        batch_size=args.batch_size,
        workers=args.workers,
        expected_classes=class_names,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else checkpoint_path.parent / f"eval_{args.split.lower()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(f"Evaluating {checkpoint_path} on the {args.split} split")
    print(f"  device: {device} | arch: {arch} | img_size: {img_size} | tta: {args.tta}")
    print(f"  images: {len(loader.dataset)} | classes: {dataset_classes}")
    print("=" * 68)

    # Reporting loss: plain cross-entropy (no class weights / smoothing) so the
    # number is comparable across runs.
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        tta=args.tta,
        limit_batches=args.limit_batches,
        desc=f"eval {args.split.lower()}",
        predictions_csv=output_dir / "predictions.csv",
    )

    save_json(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "split": args.split,
            "arch": arch,
            "img_size": img_size,
            "tta": args.tta,
            "evaluated_at": timestamp(),
            "metrics": metrics,
        },
        output_dir / "metrics.json",
    )

    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for name in class_names:
            row = metrics["per_class"][name]
            writer.writerow(
                [
                    name,
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['f1']:.4f}",
                    row["support"],
                ]
            )

    # Recompute the per-image arrays cheaply from the metrics (sklearn needs them).
    y_true, y_pred = _collect_labels(output_dir / "predictions.csv", class_names)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        output_dir / "confusion_matrix.png",
        title=f"{args.split} confusion matrix ({arch}, TTA={'on' if args.tta else 'off'})",
    )

    print("\nResults")
    print(format_metrics(metrics, class_names))
    print("\n" + report)
    print(f"Saved: {output_dir.resolve()}")
    print(
        "  metrics.json, per_class_metrics.csv, classification_report.txt,\n"
        "  confusion_matrix.png, predictions.csv"
    )


def _collect_labels(predictions_csv: Path, class_names: list[str]) -> tuple[list[int], list[int]]:
    """Read back the predictions written by ``evaluate`` for sklearn's report."""
    index = {name: i for i, name in enumerate(class_names)}
    y_true: list[int] = []
    y_pred: list[int] = []
    with predictions_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            y_true.append(index[row["true_label"]])
            y_pred.append(index[row["predicted_label"]])
    return y_true, y_pred


if __name__ == "__main__":
    main()
