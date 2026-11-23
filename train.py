"""Train a from-scratch ResNet to classify satellite/aerial disaster images.

Usage (Windows, RTX 3050 laptop):

    python train.py --data-dir data/aider --arch resnet18 --epochs 30 --batch-size 64

What it does
------------
1. Loads Train/Val splits (ImageFolder) with augmentation.
2. Builds a randomly-initialised ResNet-18 and trains it with SGD + Nesterov
   momentum, warm-up, cosine decay, label smoothing and class weights.
3. Uses mixed precision (AMP) on the GPU -- roughly 2x faster, ~40% less VRAM.
4. Tracks validation macro-F1 (fair when classes are imbalanced) and keeps
   ``best.pt`` (best F1) plus ``last.pt`` (resumable) in ``--output-dir``.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from dataset import compute_class_weights, create_train_val_loaders
from engine import evaluate, make_scaler, train_one_epoch
from model import SUPPORTED_ARCHS, build_resnet, count_parameters
from utils import (
    format_metrics,
    plot_confusion_matrix,
    plot_training_history,
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
    timestamp,
)

HISTORY_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_macro_f1",
    "val_macro_precision",
    "val_macro_recall",
    "seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a from-scratch ResNet on the AIDERv2 disaster dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data/aider", help="folder with Train/Val/Test")
    parser.add_argument("--output-dir", default="runs/resnet18", help="where checkpoints/logs go")
    parser.add_argument("--arch", default="resnet18", choices=sorted(SUPPORTED_ARCHS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64, help="lower to 32 if CUDA runs out of memory")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout before the classifier")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="stop after N epochs without validation macro-F1 improvement (0 = never)",
    )
    parser.add_argument("--workers", type=int, default=4, help="DataLoader processes (0 on Windows if it hangs)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="weight the loss by inverse class frequency",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mixed-precision training on CUDA",
    )
    parser.add_argument("--limit-train-batches", type=int, default=0, help="debug: cap batches per epoch")
    parser.add_argument("--limit-val-batches", type=int, default=0, help="debug: cap validation batches")
    parser.add_argument("--resume", default=None, help="path to a last.pt to continue from")
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    return parser.parse_args()


def build_scheduler(
    optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int, base_lr: float
):
    """Linear warm-up followed by cosine decay to 1% of the peak LR."""
    cosine_epochs = max(1, epochs - max(0, warmup_epochs))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=base_lr * 0.01
    )
    if warmup_epochs <= 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def write_history(history: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field) for field in HISTORY_FIELDS})


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("Disaster image classification - training")
    print("=" * 68)
    print(f"  device      : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  started     : {timestamp()}")
    print(f"  data        : {Path(args.data_dir).resolve()}")
    print(f"  output      : {output_dir.resolve()}")

    train_loader, val_loader, class_names, counts = create_train_val_loaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
    )
    print(f"  classes     : {class_names}")
    print(f"  train/val   : {sum(counts.values())} / {len(val_loader.dataset)} images")
    for name in class_names:
        print(f"      {name:<12} {counts[name]:>6} train images")

    model = build_resnet(args.arch, num_classes=len(class_names), dropout=args.dropout)
    model.to(device)
    print(f"  model       : {args.arch} ({count_parameters(model) / 1e6:.2f}M parameters, from scratch)")

    weights = None
    if args.class_weights:
        weights = compute_class_weights(counts, class_names).to(device)
        print("  class weights:", {name: round(float(w), 3) for name, w in zip(class_names, weights)})

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs, args.lr)
    scaler = make_scaler(device, args.amp)
    print(f"  amp         : {'on' if scaler is not None else 'off'}")
    print("=" * 68)

    start_epoch = 1
    best_f1 = -1.0
    best_metrics: dict = {}
    history: list[dict] = []

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_f1 = float(checkpoint.get("best_macro_f1", -1.0))
        history = list(checkpoint.get("history", []))
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best macro-F1 {best_f1:.4f})")

    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        learning_rate = optimizer.param_groups[0]["lr"]

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            amp=args.amp,
            grad_clip=args.grad_clip,
            limit_batches=args.limit_train_batches,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
            tta=False,
            limit_batches=args.limit_val_batches,
            desc=f"val   {epoch:02d}/{args.epochs:02d}",
        )

        scheduler.step()
        elapsed = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "lr": round(learning_rate, 6),
            "train_loss": round(train_stats["loss"], 5),
            "train_acc": round(train_stats["accuracy"], 5),
            "val_loss": round(val_metrics["loss"], 5),
            "val_acc": round(val_metrics["accuracy"], 5),
            "val_macro_f1": round(val_metrics["macro_f1"], 5),
            "val_macro_precision": round(val_metrics["macro_precision"], 5),
            "val_macro_recall": round(val_metrics["macro_recall"], 5),
            "seconds": round(elapsed, 1),
        }
        history.append(row)
        write_history(history, output_dir / "history.csv")

        print(
            f"[{epoch:02d}/{args.epochs:02d}] lr {learning_rate:.4f} | "
            f"train loss {train_stats['loss']:.4f} acc {train_stats['accuracy'] * 100:.2f}% | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy'] * 100:.2f}% "
            f"F1 {val_metrics['macro_f1'] * 100:.2f}% | {elapsed:.1f}s"
        )

        improved = val_metrics["macro_f1"] > best_f1 + 1e-4
        if improved:
            best_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                arch=args.arch,
                class_names=class_names,
                img_size=args.img_size,
                dropout=args.dropout,
                epoch=epoch,
                metrics=val_metrics,
                extra={"args": vars(args)},
            )
            print(f"        -> new best macro-F1 {best_f1 * 100:.2f}% (saved best.pt)")
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            output_dir / "last.pt",
            model,
            arch=args.arch,
            class_names=class_names,
            img_size=args.img_size,
            dropout=args.dropout,
            epoch=epoch,
            metrics=val_metrics,
            extra={
                "args": vars(args),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_macro_f1": best_f1,
                "history": history,
            },
        )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"Early stopping: no macro-F1 improvement for {args.patience} epochs."
            )
            break

    if history:
        write_history(history, output_dir / "history.csv")
        plot_training_history(history, output_dir / "training_curves.png")
        plot_confusion_matrix(
            best_metrics["confusion_matrix"],
            class_names,
            output_dir / "val_confusion_matrix.png",
            title="Validation confusion matrix (best epoch)",
        )
        save_json(
            {
                "arch": args.arch,
                "class_names": class_names,
                "img_size": args.img_size,
                "best_val_metrics": best_metrics,
                "best_macro_f1": best_f1,
                "epochs_trained": len(history),
                "args": vars(args),
                "finished": timestamp(),
            },
            output_dir / "training_summary.json",
        )

    print("\nTraining finished.")
    if best_metrics:
        print("Best validation metrics:")
        print(format_metrics(best_metrics, class_names))
    print(f"\nCheckpoints, curves and logs are in: {output_dir.resolve()}")
    print(
        "Next step:\n"
        f"  python evaluate.py --checkpoint {output_dir / 'best.pt'} --split Test\n"
        "  streamlit run app.py"
    )


if __name__ == "__main__":
    # The guard matters on Windows: DataLoader workers are started by spawn,
    # so training code must never run at import time.
    main()
