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


class TrainImageTransform:
    """Base callable for stochastic train-time augmentation."""

    def __call__(self, image: Image.Image) -> Image.Image:
        return image


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


class MildTrainAugmentation(TrainImageTransform):
    """Apply at most one mild perturbation to improve robustness.

    The transform intentionally uses conservative ranges so clean validation
    performance is less likely to regress while still teaching invariance to
    scan angle, mild blur/downsampling, and small stroke-width shifts.
    """

    def __init__(
        self,
        apply_probability: float = 0.6,
        max_rotation_degrees: float = 3.0,
        min_resolution_scale: float = 0.75,
        max_thickness_delta: int = 1,
    ) -> None:
        if not (0.0 <= apply_probability <= 1.0):
            raise ValueError("apply_probability must be in [0, 1].")
        if not (0.0 < min_resolution_scale <= 1.0):
            raise ValueError("min_resolution_scale must be in (0, 1].")
        if max_thickness_delta < 0:
            raise ValueError("max_thickness_delta must be non-negative.")

        self.apply_probability = apply_probability
        self.max_rotation_degrees = max_rotation_degrees
        self.min_resolution_scale = min_resolution_scale
        self.max_thickness_delta = max_thickness_delta

    def __call__(self, image: Image.Image) -> Image.Image:
        if torch.rand(()).item() >= self.apply_probability:
            return image

        transform_draw = torch.rand(()).item()
        if transform_draw < 0.45:
            degrees = float(
                torch.empty(1).uniform_(
                    -self.max_rotation_degrees,
                    self.max_rotation_degrees,
                ).item()
            )
            return RotationPerturbation(degrees=degrees)(image)

        if transform_draw < 0.8:
            scale = float(
                torch.empty(1).uniform_(
                    self.min_resolution_scale,
                    1.0,
                ).item()
            )
            return ResolutionPerturbation(scale=scale)(image)

        if self.max_thickness_delta == 0:
            return image

        delta = int(torch.randint(0, 2, (1,)).item()) * 2 - 1
        delta *= self.max_thickness_delta
        return StrokeThicknessPerturbation(delta_pixels=delta)(image)


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


def build_train_image_transform(profile: str) -> TrainImageTransform | None:
    """Build the stochastic train-time augmentation callable for a profile."""
    normalized = profile.strip().lower()
    if normalized in {"none", "off"}:
        return None
    if normalized == "mild_v1":
        return MildTrainAugmentation()
    raise ValueError(
        "Unknown train augmentation profile. "
        "Expected one of {'none', 'mild_v1'}."
    )
