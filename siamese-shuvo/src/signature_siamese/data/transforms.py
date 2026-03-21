"""Image preprocessing used for both training and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


class SignaturePreprocessor:
    """Load, invert, resize, and convert signature image to a torch tensor.

    The output tensor is float32 in [0, 1] with shape [1, H, W].
    """

    def __init__(self, image_height: int = 155, image_width: int = 220) -> None:
        self.image_height = image_height
        self.image_width = image_width

    def __call__(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("L")
        image = ImageOps.invert(image)
        image = image.resize((self.image_width, self.image_height), resample=Image.BILINEAR)

        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor
