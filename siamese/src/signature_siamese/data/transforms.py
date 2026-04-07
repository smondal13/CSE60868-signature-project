"""Image preprocessing used for both training and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps


class SignaturePreprocessor:
    """Load, invert, resize, and convert signature image to a torch tensor.

    The output tensor is float32 in [0, 1] with shape [1, H, W].
    """

    def __init__(self, image_height: int = 155, image_width: int = 220) -> None:
        self.image_height = image_height
        self.image_width = image_width

    def load_image(self, image_path: Path) -> Image.Image:
        image = Image.open(image_path).convert("L")
        image = ImageOps.invert(image)
        image = image.resize((self.image_width, self.image_height), resample=Image.BILINEAR)
        return image

    def to_tensor(self, image: Image.Image) -> torch.Tensor:
        if image.size != (self.image_width, self.image_height):
            image = image.resize((self.image_width, self.image_height), resample=Image.BILINEAR)

        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor

    def __call__(self, image_path: Path) -> torch.Tensor:
        image = self.load_image(image_path)
        return self.to_tensor(image)


class QueryPerturbation:
    """Base callable for deterministic query-image robustness transforms."""

    def __call__(self, image: Image.Image) -> Image.Image:
        return image


class RotationPerturbation(QueryPerturbation):
    def __init__(self, degrees: float) -> None:
        self.degrees = degrees

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.rotate(
            angle=self.degrees,
            resample=Image.BILINEAR,
            fillcolor=0,
        )


class ResolutionPerturbation(QueryPerturbation):
    def __init__(self, scale: float) -> None:
        if not (0.0 < scale <= 1.0):
            raise ValueError("Resolution scale must be in (0, 1].")
        self.scale = scale

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.scale == 1.0:
            return image
        width, height = image.size
        reduced_size = (
            max(1, int(round(width * self.scale))),
            max(1, int(round(height * self.scale))),
        )
        reduced = image.resize(reduced_size, resample=Image.BILINEAR)
        return reduced.resize((width, height), resample=Image.BILINEAR)


class StrokeThicknessPerturbation(QueryPerturbation):
    def __init__(self, delta_pixels: int) -> None:
        self.delta_pixels = delta_pixels

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.delta_pixels == 0:
            return image
        kernel_size = 2 * abs(self.delta_pixels) + 1
        if self.delta_pixels > 0:
            return image.filter(ImageFilter.MaxFilter(size=kernel_size))
        return image.filter(ImageFilter.MinFilter(size=kernel_size))


def build_query_perturbation(
    kind: str,
    *,
    degrees: float = 0.0,
    scale: float = 1.0,
    delta_pixels: int = 0,
) -> QueryPerturbation:
    """Build a deterministic query perturbation for robustness evaluation."""
    normalized = kind.strip().lower()
    if normalized == "none":
        return QueryPerturbation()
    if normalized == "rotate":
        return RotationPerturbation(degrees=degrees)
    if normalized == "resolution":
        return ResolutionPerturbation(scale=scale)
    if normalized == "thickness":
        return StrokeThicknessPerturbation(delta_pixels=delta_pixels)
    raise ValueError(
        "Unknown perturbation kind. "
        "Expected one of {'none', 'rotate', 'resolution', 'thickness'}."
    )
