"""Siamese CNN backbone and embedding head."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Basic Conv2D -> ReLU -> MaxPool block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SignatureBackbone(nn.Module):
    """CNN feature extractor that outputs an L2-normalized embedding."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.proj(x)
        # L2 normalization makes Euclidean distance scale stable across runs.
        x = F.normalize(x, p=2, dim=1)
        return x


class SiameseNetwork(nn.Module):
    """Siamese network with shared CNN backbone."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.backbone = SignatureBackbone(embedding_dim=embedding_dim)

    def forward_once(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding_a = self.forward_once(image_a)
        embedding_b = self.forward_once(image_b)
        return embedding_a, embedding_b
