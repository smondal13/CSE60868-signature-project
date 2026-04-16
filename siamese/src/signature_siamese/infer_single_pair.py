"""Run single-pair signature verification inference with a trained checkpoint.

This entry point is intended for demo/validation usage where we want one
command that loads a fixed pair of images, computes the embedding distance, and
reports the predicted verification decision against the checkpoint threshold.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .checkpoints import (
    infer_embedding_dim,
    load_checkpoint,
    resolve_checkpoint_path,
)
from .data.transforms import SignaturePreprocessor
from .device import resolve_device
from .model.siamese import SiameseNetwork
from .utils import dump_json, ensure_dir

# -----------------------------------------------------------------------------
# Top-level configuration
# -----------------------------------------------------------------------------
RUN_NAME_PREFIX = os.getenv("SIGNATURE_RUN_NAME_PREFIX", "siamese_full_hpo_lr5e4_m075")
checkpoint_override = os.getenv("SIGNATURE_CHECKPOINT_PATH")
CHECKPOINT_PATH: Path | None = (
    Path(checkpoint_override) if checkpoint_override else None
)

IMAGE_A_PATH = Path(os.getenv("SIGNATURE_PAIR_IMAGE_A", "validation/B-S-83-G-04.tif"))
IMAGE_B_PATH = Path(os.getenv("SIGNATURE_PAIR_IMAGE_B", "validation/B-S-83-F-05.tif"))
EXPECTED_LABEL = int(os.getenv("SIGNATURE_PAIR_EXPECTED_LABEL", "0"))
OUTPUT_DIR = Path(os.getenv("SIGNATURE_SINGLE_PAIR_OUTPUT_DIR", "validation/results"))

IMAGE_HEIGHT = 155
IMAGE_WIDTH = 220
EMBEDDING_DIM = 128
DEVICE = os.getenv("SIGNATURE_DEVICE", "auto")  # auto | cuda | mps | cpu
threshold_override = os.getenv("SIGNATURE_LOCKED_THRESHOLD")
LOCKED_THRESHOLD: float | None = (
    float(threshold_override) if threshold_override else None
)
# -----------------------------------------------------------------------------


def _validate_label(label: int) -> None:
    if label not in {0, 1}:
        raise ValueError("SIGNATURE_PAIR_EXPECTED_LABEL must be 0 or 1.")


def _load_image_tensor(preprocessor: SignaturePreprocessor, path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Could not find pair image at {path}.")
    return preprocessor(path).unsqueeze(0)


def main() -> None:
    _validate_label(EXPECTED_LABEL)

    preferred_device = None if DEVICE == "auto" else DEVICE
    device = resolve_device(preferred=preferred_device)

    checkpoint_path = resolve_checkpoint_path(CHECKPOINT_PATH, RUN_NAME_PREFIX)
    checkpoint = load_checkpoint(checkpoint_path)
    embedding_dim = infer_embedding_dim(checkpoint, fallback=EMBEDDING_DIM)

    locked_threshold = LOCKED_THRESHOLD
    if locked_threshold is None:
        checkpoint_threshold = checkpoint.get("best_threshold")
        if checkpoint_threshold is None:
            raise RuntimeError(
                "No threshold found in checkpoint. "
                "Set SIGNATURE_LOCKED_THRESHOLD explicitly."
            )
        locked_threshold = float(checkpoint_threshold)

    model = SiameseNetwork(embedding_dim=embedding_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    preprocessor = SignaturePreprocessor(
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
    )

    image_a = _load_image_tensor(preprocessor, IMAGE_A_PATH).to(device)
    image_b = _load_image_tensor(preprocessor, IMAGE_B_PATH).to(device)

    with torch.no_grad():
        embedding_a = model.forward_once(image_a)
        embedding_b = model.forward_once(image_b)
        distance = float(torch.norm(embedding_a - embedding_b, p=2, dim=1).item())

    predicted_label = 1 if distance <= locked_threshold else 0
    predicted_text = "match" if predicted_label == 1 else "non-match"
    expected_text = "match" if EXPECTED_LABEL == 1 else "non-match"

    output_dir = ensure_dir(OUTPUT_DIR)
    output_path = output_dir / "single_pair_result.json"
    payload = {
        "checkpoint": str(checkpoint_path),
        "image_a": str(IMAGE_A_PATH),
        "image_b": str(IMAGE_B_PATH),
        "distance": distance,
        "threshold": locked_threshold,
        "predicted_label": predicted_label,
        "predicted_text": predicted_text,
        "expected_label": EXPECTED_LABEL,
        "expected_text": expected_text,
        "correct": predicted_label == EXPECTED_LABEL,
        "device": device.type,
    }
    dump_json(output_path, payload)

    print("Single-pair inference complete.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image A: {IMAGE_A_PATH}")
    print(f"Image B: {IMAGE_B_PATH}")
    print(f"Distance: {distance:.6f}")
    print(f"Threshold: {locked_threshold:.6f}")
    print(f"Predicted: {predicted_text} ({predicted_label})")
    print(f"Expected: {expected_text} ({EXPECTED_LABEL})")
    print(f"Correct: {predicted_label == EXPECTED_LABEL}")
    print(f"Saved result: {output_path}")


if __name__ == "__main__":
    main()
