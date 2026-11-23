"""Checkpoint loading and single-image inference (used by app.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torchvision import transforms

from dataset import IMAGENET_MEAN, IMAGENET_STD
from model import build_resnet
from utils import resolve_device


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    """Load a checkpoint written by train.py."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" not in checkpoint or "class_names" not in checkpoint:
        raise ValueError(
            f"'{path}' is not a checkpoint written by train.py "
            "(expected keys: 'model_state', 'class_names')."
        )
    return checkpoint


def load_model_for_inference(
    checkpoint_path: str | Path, device: str | torch.device = "auto"
) -> Tuple[torch.nn.Module, List[str], int, Dict[str, Any]]:
    """Rebuild the trained network for prediction.

    Returns ``(model, class_names, img_size, metadata)``.
    """
    device = resolve_device(str(device)) if isinstance(device, str) else device
    checkpoint = load_checkpoint(checkpoint_path)

    arch = str(checkpoint.get("arch", "resnet18"))
    class_names = [str(name) for name in checkpoint["class_names"]]
    img_size = int(checkpoint.get("img_size", 224))
    dropout = float(checkpoint.get("dropout", 0.0))

    model = build_resnet(arch, num_classes=len(class_names), dropout=dropout)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    metadata = {
        "arch": arch,
        "img_size": img_size,
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics") or {},
        "path": str(checkpoint_path),
    }
    return model, class_names, img_size, metadata


def build_inference_transform(img_size: int = 224) -> transforms.Compose:
    """Deterministic preprocessing -- must match the validation transform."""
    resize_to = int(round(img_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize_to),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@torch.no_grad()
def predict_pil_image(
    model: torch.nn.Module,
    image: Image.Image,
    class_names: List[str],
    img_size: int = 224,
    device: str | torch.device | None = None,
    tta: bool = True,
) -> Dict[str, Any]:
    """Classify one PIL image.

    Returns a dict with ``label``, ``confidence`` (0-1) and ``probabilities``
    (class name -> probability).
    """
    if device is None:
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)

    model.eval()
    tensor = build_inference_transform(img_size)(image.convert("RGB")).unsqueeze(0)
    tensor = tensor.to(device)

    probabilities = model(tensor).float().softmax(dim=1)
    if tta:
        flipped = model(torch.flip(tensor, dims=[3])).float().softmax(dim=1)
        probabilities = (probabilities + flipped) / 2.0
    probabilities = probabilities.squeeze(0).cpu()

    confidence, index = probabilities.max(dim=0)
    return {
        "label": class_names[int(index)],
        "confidence": float(confidence),
        "probabilities": {
            name: float(probabilities[i]) for i, name in enumerate(class_names)
        },
    }


@torch.no_grad()
def predict_path(
    model: torch.nn.Module,
    image_path: str | Path,
    class_names: List[str],
    img_size: int = 224,
    device: str | torch.device | None = None,
    tta: bool = True,
) -> Dict[str, Any]:
    """Classify an image on disk."""
    with Image.open(image_path) as image:
        return predict_pil_image(model, image, class_names, img_size, device, tta)
