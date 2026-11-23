"""Dataset / DataLoader helpers for the disaster classifier.

Expected on-disk layout (created by ``download_data.py``)::

    data/aider/
        Train/{Earthquake,Flood,Normal,Wildfire}/*.jpg
        Val/  {Earthquake,Flood,Normal,Wildfire}/*.jpg
        Test/ {Earthquake,Flood,Normal,Wildfire}/*.jpg
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

SPLITS = ("Train", "Val", "Test")
# AIDERv2 folder names are Earthquake / Fire / Flood / Normal.  Class order is
# read from the folders themselves (ImageFolder sorts them alphabetically), so
# this project works unchanged if you rename "Fire" to "Wildfire".


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_transforms(img_size: int = 224, train: bool = False) -> transforms.Compose:
    """Return the augmentation / preprocessing pipeline.

    Training uses strong-but-safe augmentation (the dataset is a few thousand
    images per class, so regularisation matters).  Evaluation is deterministic:
    resize the short side to 256 then centre-crop.
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    img_size, scale=(0.55, 1.0), ratio=(0.8, 1.25)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                # Light occlusion regulariser: helps avoid relying on a single
                # corner of the frame.  value=0 is the "mean" pixel after
                # normalisation, so it behaves like a grey patch.
                transforms.RandomErasing(p=0.10, scale=(0.02, 0.15), value=0.0),
            ]
        )

    resize_to = int(round(img_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize_to),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# --------------------------------------------------------------------------- #
# Locating the splits on disk
# --------------------------------------------------------------------------- #
def _has_subdirectories(path: Path) -> bool:
    try:
        return any(child.is_dir() for child in path.iterdir())
    except OSError:
        return False


def resolve_split_dir(data_dir: str | os.PathLike, split: str) -> Path:
    """Find the directory that holds ``split`` (e.g. ``Train``).

    The AIDERv2 zips contain an extra nesting level (``Dataset_Images/Train``),
    so we search the tree instead of assuming a fixed depth.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Data directory '{root}' does not exist. "
            "Run `python download_data.py` first (see README.md)."
        )

    candidates: List[Path] = []
    for dirpath, _dirnames, _filenames in os.walk(root):
        current = Path(dirpath)
        if current.name.lower() == split.lower() and _has_subdirectories(current):
            candidates.append(current)

    if not candidates:
        found = sorted({p.name for p in root.rglob("*") if p.is_dir()})[:15]
        raise FileNotFoundError(
            f"No '{split}' split found under '{root}'.\n"
            f"Directories that do exist: {found or 'none'}.\n"
            "Expected layout: data/aider/Train/<ClassName>/*.jpg"
        )

    # Prefer the shallowest match (closest to the data root).
    return min(candidates, key=lambda p: len(p.parts))


# --------------------------------------------------------------------------- #
# DataLoaders
# --------------------------------------------------------------------------- #
def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _make_loader(
    dataset: datasets.ImageFolder,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=workers > 0,
        worker_init_fn=_seed_worker if workers > 0 else None,
        generator=generator if shuffle else None,
    )


def class_counts(dataset: datasets.ImageFolder) -> Dict[str, int]:
    """Number of images per class in an ``ImageFolder`` dataset."""
    counts = {name: 0 for name in dataset.classes}
    for _, label in dataset.samples:
        counts[dataset.classes[label]] += 1
    return counts


def compute_class_weights(
    counts: Dict[str, int], class_names: List[str]
) -> torch.Tensor:
    """Inverse-frequency weights so rare disaster classes are not ignored."""
    total = sum(counts.values())
    n_classes = len(class_names)
    weights = [total / (n_classes * max(1, counts[name])) for name in class_names]
    # Normalise so the average weight is 1.0 (keeps the loss scale stable).
    mean_weight = sum(weights) / len(weights)
    return torch.tensor([w / mean_weight for w in weights], dtype=torch.float32)


def create_train_val_loaders(
    data_dir: str | os.PathLike,
    img_size: int = 224,
    batch_size: int = 64,
    workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, List[str], Dict[str, int]]:
    """Build the training and validation loaders plus class metadata."""
    train_dir = resolve_split_dir(data_dir, "Train")
    val_dir = resolve_split_dir(data_dir, "Val")

    train_ds = datasets.ImageFolder(
        str(train_dir), transform=build_transforms(img_size, train=True)
    )
    val_ds = datasets.ImageFolder(
        str(val_dir), transform=build_transforms(img_size, train=False)
    )

    if train_ds.classes != val_ds.classes:
        raise RuntimeError(
            "Train and Val contain different class folders:\n"
            f"  Train: {train_ds.classes}\n  Val:   {val_ds.classes}"
        )

    class_names = list(train_ds.classes)
    train_loader = _make_loader(train_ds, batch_size, True, workers, seed)
    val_loader = _make_loader(val_ds, batch_size, False, workers, seed)
    return train_loader, val_loader, class_names, class_counts(train_ds)


def create_eval_loader(
    data_dir: str | os.PathLike,
    split: str = "Test",
    img_size: int = 224,
    batch_size: int = 64,
    workers: int = 4,
    expected_classes: List[str] | None = None,
) -> Tuple[DataLoader, List[str], datasets.ImageFolder]:
    """Build a deterministic loader for one split (``Val`` or ``Test``)."""
    split_dir = resolve_split_dir(data_dir, split)
    dataset = datasets.ImageFolder(
        str(split_dir), transform=build_transforms(img_size, train=False)
    )
    if expected_classes is not None and list(dataset.classes) != list(
        expected_classes
    ):
        raise RuntimeError(
            f"Class folders in '{split}' do not match the checkpoint:\n"
            f"  checkpoint: {list(expected_classes)}\n"
            f"  {split}: {list(dataset.classes)}"
        )
    loader = _make_loader(dataset, batch_size, False, workers, 0)
    return loader, list(dataset.classes), dataset
