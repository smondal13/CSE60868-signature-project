"""Contrastive loss for Siamese metric learning."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveLoss(nn.Module):
    """Classic contrastive loss with a margin on negative pairs.

    Labels must be 1 for positive (same-writer genuine pair) and 0 for negative.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        if margin <= 0:
            raise ValueError("margin must be > 0")
        self.margin = margin

    def forward(self, distances: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.float()
        positive_term = labels * distances.pow(2)
        negative_term = (1.0 - labels) * F.relu(self.margin - distances).pow(2)
        return (positive_term + negative_term).mean()
