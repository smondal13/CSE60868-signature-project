"""Robustness evaluation for query-side domain shifts.

This module keeps the trained model fixed, perturbs only the query image in each
verification pair, and measures how far the held-out metrics drift from the
unperturbed baseline.

Edit the configuration block below, then run:

    SIGNATURE_DATA_ROOT=/path/to/cedar-bhsig260 \
    PYTHONPATH=siamese/src \
    conda run -n machine-learning python -m signature_siamese.evaluate_robustness
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.datasets import SignatureDataset
from .data.pairs import VerificationPairDataset, build_verification_pairs, iter_pair_types
from .data.transforms import build_query_perturbation
from .device import resolve_device
from .eval_utils import eval_result_to_dict, evaluate_pair_loader
from .model.siamese import SiameseNetwork
from .utils import dump_json, ensure_dir

# -----------------------------------------------------------------------------
# Top-level configuration (edit these values directly)
# -----------------------------------------------------------------------------
# Environment variables can override these defaults so the same script works for
# quick local smoke tests and full CRC runs without repeated code edits.
RUN_PROFILE = os.getenv("SIGNATURE_RUN_PROFILE", "full")  # "small" or "full"
SPLIT = os.getenv("SIGNATURE_EVAL_SPLIT", "test")  # "val" or "test"

if RUN_PROFILE == "small":
    MANIFEST_CSV = Path("siamese/manifests/bhsig260_small_manifest.csv")
    RUN_NAME_PREFIX = "small_debug"
    DEFAULT_OUTPUT_DIR = Path("siamese/results/robustness_small")
    EVAL_BATCH_SIZE = 256
    THRESHOLD_POINTS = 400
    MAX_SKILLED_FORGERIES_PER_WRITER = 20
    MAX_RANDOM_IMPOSTORS_PER_WRITER = 10
    NUM_WORKERS = 0
elif RUN_PROFILE == "full":
    MANIFEST_CSV = Path("siamese/manifests/bhsig260_manifest.csv")
    RUN_NAME_PREFIX = "siamese_full"
    DEFAULT_OUTPUT_DIR = Path("siamese/results/robustness_full")
    EVAL_BATCH_SIZE = 128
    THRESHOLD_POINTS = 2000
    MAX_SKILLED_FORGERIES_PER_WRITER = 720
    MAX_RANDOM_IMPOSTORS_PER_WRITER = 200
    NUM_WORKERS = 2
else:
    raise ValueError("RUN_PROFILE must be either 'small' or 'full'.")

checkpoint_override = os.getenv("SIGNATURE_CHECKPOINT_PATH")
output_dir_override = os.getenv("SIGNATURE_OUTPUT_DIR")
CHECKPOINT_PATH: Path | None = (
    Path(checkpoint_override) if checkpoint_override else None
)
OUTPUT_DIR = Path(output_dir_override) if output_dir_override else DEFAULT_OUTPUT_DIR
IMAGE_HEIGHT = 155
IMAGE_WIDTH = 220
EMBEDDING_DIM = 128
INCLUDE_CROSS_WRITER_IMPOSTORS = True
LOCKED_THRESHOLD: float | None = None
DEVICE = "auto"  # auto | cuda | mps | cpu
SEED = 42

# The baseline plus a small set of query-only robustness checks.
ROBUSTNESS_SWEEP = [
    {"name": "baseline", "kind": "none"},
    {"name": "rotate_p3", "kind": "rotate", "degrees": 3.0},
    {"name": "rotate_m3", "kind": "rotate", "degrees": -3.0},
    {"name": "resolution_50", "kind": "resolution", "scale": 0.5},
    {"name": "thickness_plus1", "kind": "thickness", "delta_pixels": 1},
    {"name": "thickness_minus1", "kind": "thickness", "delta_pixels": -1},
]
# -----------------------------------------------------------------------------


def _load_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, object]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected checkpoint format at {path}.")
    return payload


def _infer_embedding_dim(checkpoint: dict[str, object], fallback: int) -> int:
    config = checkpoint.get("config", {})
    if isinstance(config, dict) and "embedding_dim" in config:
        return int(config["embedding_dim"])
    return fallback


def _resolve_latest_checkpoint_by_prefix(
    prefix: str, runs_root: Path = Path("siamese/runs")
) -> Path:
    candidates = sorted(
        [
            path
            for path in runs_root.iterdir()
            if path.is_dir() and path.name.startswith(f"{prefix}_")
        ],
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No run directories found under {runs_root} with prefix '{prefix}_'."
        )

    latest = candidates[-1]
    checkpoint = latest / "checkpoints" / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint at expected path: {checkpoint}")
    return checkpoint


def _resolve_checkpoint_path() -> Path:
    if CHECKPOINT_PATH is not None:
        return CHECKPOINT_PATH
    return _resolve_latest_checkpoint_by_prefix(RUN_NAME_PREFIX)


def _save_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "kind",
        "auc",
        "eer",
        "far_at_locked_threshold",
        "frr_at_locked_threshold",
        "delta_auc",
        "delta_eer",
        "delta_far_locked",
        "delta_frr_locked",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    preferred_device = None if DEVICE == "auto" else DEVICE
    device = resolve_device(preferred=preferred_device)

    checkpoint_path = _resolve_checkpoint_path()
    checkpoint = _load_checkpoint(checkpoint_path)
    embedding_dim = _infer_embedding_dim(checkpoint, fallback=EMBEDDING_DIM)

    model = SiameseNetwork(embedding_dim=embedding_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    dataset = SignatureDataset(
        manifest_csv=MANIFEST_CSV,
        split=SPLIT,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
    )
    pairs = build_verification_pairs(
        dataset=dataset,
        seed=SEED,
        max_skilled_forgeries_per_writer=MAX_SKILLED_FORGERIES_PER_WRITER,
        max_random_impostors_per_writer=MAX_RANDOM_IMPOSTORS_PER_WRITER,
        include_cross_writer_impostors=INCLUDE_CROSS_WRITER_IMPOSTORS,
    )
    pair_counts = dict(Counter(iter_pair_types(pairs)))

    locked_threshold = LOCKED_THRESHOLD
    if locked_threshold is None:
        checkpoint_threshold = checkpoint.get("best_threshold")
        if checkpoint_threshold is not None:
            locked_threshold = float(checkpoint_threshold)

    output_dir = ensure_dir(OUTPUT_DIR)
    metrics_path = output_dir / f"robustness_metrics_{SPLIT}.json"
    summary_csv = output_dir / f"robustness_summary_{SPLIT}.csv"

    sweep_rows: list[dict[str, object]] = []
    sweep_payload: list[dict[str, object]] = []
    baseline_result = None

    for config in ROBUSTNESS_SWEEP:
        perturbation = build_query_perturbation(
            config["kind"],
            degrees=float(config.get("degrees", 0.0)),
            scale=float(config.get("scale", 1.0)),
            delta_pixels=int(config.get("delta_pixels", 0)),
        )

        pair_dataset = VerificationPairDataset(
            base_dataset=dataset,
            pairs=pairs,
            query_transform=perturbation,
        )
        pair_loader = DataLoader(
            dataset=pair_dataset,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
            persistent_workers=NUM_WORKERS > 0,
        )
        result = evaluate_pair_loader(
            model=model,
            pair_loader=pair_loader,
            device=device,
            threshold_points=THRESHOLD_POINTS,
            locked_threshold=locked_threshold,
        )
        if baseline_result is None:
            baseline_result = result

        delta_auc = result.auc - baseline_result.auc
        delta_eer = result.eer - baseline_result.eer
        delta_far_locked = None
        delta_frr_locked = None
        if (
            result.far_at_locked_threshold is not None
            and baseline_result.far_at_locked_threshold is not None
        ):
            delta_far_locked = (
                result.far_at_locked_threshold
                - baseline_result.far_at_locked_threshold
            )
        if (
            result.frr_at_locked_threshold is not None
            and baseline_result.frr_at_locked_threshold is not None
        ):
            delta_frr_locked = (
                result.frr_at_locked_threshold
                - baseline_result.frr_at_locked_threshold
            )

        sweep_rows.append(
            {
                "name": config["name"],
                "kind": config["kind"],
                "auc": result.auc,
                "eer": result.eer,
                "far_at_locked_threshold": result.far_at_locked_threshold,
                "frr_at_locked_threshold": result.frr_at_locked_threshold,
                "delta_auc": delta_auc,
                "delta_eer": delta_eer,
                "delta_far_locked": delta_far_locked,
                "delta_frr_locked": delta_frr_locked,
            }
        )
        sweep_payload.append(
            {
                "config": config,
                "metrics": eval_result_to_dict(result),
                "delta_from_baseline": {
                    "auc": delta_auc,
                    "eer": delta_eer,
                    "far_at_locked_threshold": delta_far_locked,
                    "frr_at_locked_threshold": delta_frr_locked,
                },
            }
        )

        print(
            f"{config['name']}: "
            f"AUC={result.auc:.4f}, "
            f"EER={result.eer:.4f}, "
            f"delta_eer={delta_eer:+.4f}"
        )

    dump_json(
        metrics_path,
        {
            "run_profile": RUN_PROFILE,
            "split": SPLIT,
            "device": device.type,
            "checkpoint": str(checkpoint_path),
            "locked_threshold": locked_threshold,
            "pair_counts": pair_counts,
            "sweep": sweep_payload,
        },
    )
    _save_summary_csv(summary_csv, sweep_rows)

    print(f"Saved robustness metrics: {metrics_path}")
    print(f"Saved robustness summary: {summary_csv}")


if __name__ == "__main__":
    main()
