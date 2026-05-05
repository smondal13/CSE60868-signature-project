"""Single-sample inference for the writer-dependent signature classifier.

Loads a trained checkpoint (either the custom CNN or the fine-tuned ResNet-18)
and predicts the writer ID for one signature image. The image and the
checkpoint are both command-line arguments.

Usage:
    python infer_single_sample.py --image ./validation_sample.tif
    python infer_single_sample.py --image ./validation_sample.tif \
            --checkpoint "./BHsig Results/custom_allclasses_best.pt"

The script prints the top-1 predicted writer ID, a top-5 candidate list with
softmax probabilities, and the inference time. The attached
`validation_sample.tif` is a genuine signature from BHSig260-Hindi writer
011; the expected top-1 prediction is writer 11.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import transforms, models


# ---------------------------------------------------------------------------
# Model definitions (must match the training notebook exactly so checkpoints
# load cleanly).
# ---------------------------------------------------------------------------


class SignatureCNN(nn.Module):
    """Custom from-scratch CNN used by the 'custom' backbone.

    Mirrors the architecture in signature_cnn_final.ipynb: four conv blocks
    (Conv3x3 -> GroupNorm -> ReLU -> MaxPool2) followed by a GAP head.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        def block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_resnet18(num_classes: int) -> nn.Module:
    """Build ResNet-18 with a fresh num_classes-dim head. Weights are loaded
    from the checkpoint, so ImageNet init is irrelevant here."""
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


# ---------------------------------------------------------------------------
# Preprocessing (must match the training notebook).
# ---------------------------------------------------------------------------


class InvertTransform:
    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.invert(img)


class ToThreeChannel:
    """Repeat a single grayscale channel 3 times for 3-channel backbones."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] == 1:
            return tensor.repeat(3, 1, 1)
        return tensor


def build_transform(
    backbone: str, image_height: int, image_width: int,
    norm_mean: list[float], norm_std: list[float],
) -> transforms.Compose:
    steps: list = [
        transforms.Grayscale(num_output_channels=1),
        InvertTransform(),
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
    ]
    if backbone == "resnet18":
        steps.append(ToThreeChannel())
    steps.append(transforms.Normalize(mean=norm_mean, std=norm_std))
    return transforms.Compose(steps)


# ---------------------------------------------------------------------------
# Inference pipeline.
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: Path, device: torch.device):
    """Load the checkpoint and build the matching model. Returns the model,
    the preprocessing transform, and the list of class names."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    backbone = ckpt["backbone"]
    num_classes = ckpt["num_classes"]
    class_names = ckpt["class_names"]
    image_height = ckpt["image_height"]
    image_width = ckpt["image_width"]
    norm_mean = ckpt["norm_mean"]
    norm_std = ckpt["norm_std"]

    if backbone == "custom":
        model = SignatureCNN(num_classes)
    elif backbone == "resnet18":
        model = build_resnet18(num_classes)
    else:
        raise ValueError(f"Unknown backbone in checkpoint: {backbone!r}")

    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    transform = build_transform(
        backbone, image_height, image_width, norm_mean, norm_std
    )

    return model, transform, class_names, backbone


def predict(
    model: nn.Module,
    transform: transforms.Compose,
    class_names: list[str],
    image_path: Path,
    device: torch.device,
    top_k: int = 5,
):
    img = Image.open(image_path)
    x = transform(img).unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    top_idx = probs.argsort()[::-1][:top_k]
    top_candidates = [(class_names[i], float(probs[i])) for i in top_idx]

    return {
        "predicted_class": class_names[int(top_idx[0])],
        "predicted_prob": float(probs[int(top_idx[0])]),
        "top_candidates": top_candidates,
        "inference_time_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path, required=True,
        help="Path to the signature image to classify.",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("./BHsig Results/resnet18_allclasses_best.pt"),
        help="Path to the trained checkpoint (.pt file). "
             "Default is the ResNet-18 best checkpoint.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top candidates to print (default: 5).",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            f"Make sure you have run the training notebook at least once "
            f"with the matching backbone/classes configuration."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device:     {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image:      {args.image}")

    model, transform, class_names, backbone = load_model(args.checkpoint, device)
    print(f"Backbone:   {backbone}")
    print(f"Num classes: {len(class_names)}")
    print()

    result = predict(
        model=model,
        transform=transform,
        class_names=class_names,
        image_path=args.image,
        device=device,
        top_k=args.top_k,
    )

    print(f"Predicted writer: {result['predicted_class']} "
          f"(probability {result['predicted_prob']:.4f})")
    print(f"Inference time:   {result['inference_time_ms']:.1f} ms")
    print()
    print(f"Top-{args.top_k} candidates:")
    for rank, (name, prob) in enumerate(result["top_candidates"], start=1):
        print(f"  {rank}. writer {name:>4s}   p={prob:.4f}")


if __name__ == "__main__":
    main()
