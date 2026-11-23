"""ResNet-18/34/50 implemented from scratch (no pretrained weights).

Architecture follows "Deep Residual Learning for Image Recognition"
(He et al., 2015):

    stem   : 7x7/2 conv (64) -> BN -> ReLU -> 3x3/2 max-pool
    stage1 : BasicBlock/Bottleneck x N   (64)
    stage2 : ...                          (128, stride 2)
    stage3 : ...                          (256, stride 2)
    stage4 : ...                          (512, stride 2)
    head   : global average pool -> dropout -> fully connected

ResNet-18 is the default: ~11.7M parameters, ~1.8 GFLOPs per 224x224 image.
That trains comfortably in mixed precision on a 4 GB RTX 3050 laptop GPU,
while ResNet-50 (25.6M params, ~3x the compute) gives no benefit on a
~13k-image dataset and is far more prone to overfitting when trained
from scratch.
"""

from __future__ import annotations

from typing import List, Tuple, Type

import torch
import torch.nn as nn

__all__ = [
    "BasicBlock",
    "Bottleneck",
    "ResNet",
    "build_resnet",
    "count_parameters",
    "SUPPORTED_ARCHS",
]


class BasicBlock(nn.Module):
    """Two 3x3 convolutions plus a (possibly projected) identity shortcut."""

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class Bottleneck(nn.Module):
    """1x1 -> 3x3 -> 1x1 block with a 4x channel expansion (ResNet-50/101)."""

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        out_planes = planes * self.expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes, out_planes, kernel_size=1, stride=stride, bias=False
                ),
                norm_layer(out_planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + identity
        return self.relu(out)


class ResNet(nn.Module):
    """A ResNet built from ``block`` and per-stage block counts ``layers``."""

    def __init__(
        self,
        block: Type[nn.Module],
        layers: List[int],
        num_classes: int = 4,
        dropout: float = 0.2,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        self._norm_layer = nn.BatchNorm2d
        self.in_planes = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()
        if zero_init_residual:
            # Zero-initialise the last BN gamma of every residual branch so the
            # network starts as a set of identity mappings (He et al.).
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.zeros_(module.bn3.weight)
                elif isinstance(module, BasicBlock):
                    nn.init.zeros_(module.bn2.weight)

    def _make_layer(
        self, block: Type[nn.Module], planes: int, blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(block(self.in_planes, planes, block_stride, self._norm_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.forward_features(x)))


SUPPORTED_ARCHS: dict[str, Tuple[Type[nn.Module], List[int]]] = {
    "resnet18": (BasicBlock, [2, 2, 2, 2]),
    "resnet34": (BasicBlock, [3, 4, 6, 3]),
    "resnet50": (Bottleneck, [3, 4, 6, 3]),
}


def build_resnet(
    arch: str = "resnet18", num_classes: int = 4, dropout: float = 0.2
) -> ResNet:
    """Create a randomly-initialised ResNet for ``num_classes`` classes."""
    key = arch.lower().replace("-", "").replace("_", "")
    if key not in SUPPORTED_ARCHS:
        raise ValueError(
            f"Unsupported architecture '{arch}'. Choose one of: "
            f"{', '.join(sorted(SUPPORTED_ARCHS))}"
        )
    block, layers = SUPPORTED_ARCHS[key]
    return ResNet(block, layers, num_classes=num_classes, dropout=dropout)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of (trainable) parameters."""
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )
