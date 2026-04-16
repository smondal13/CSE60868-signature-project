"""Evaluation entry point for validation/test FAR-FRR-ROC-EER reporting.

This module intentionally uses top-level editable variables instead of argparse.
Edit the configuration block below, then run:

    SIGNATURE_DATA_ROOT=/path/to/cedar-bhsig260 \
    PYTHONPATH=siamese/src \
    conda run -n machine-learning python -m signature_siamese.evaluate
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path
from torch.utils.data import DataLoader

from .checkpoints import (
    infer_embedding_dim,
    load_checkpoint,
    resolve_checkpoint_path,
)
from .data.datasets import SignatureDataset
from .data.pairs import (
    VerificationPairDataset,
    build_verification_pairs,
    iter_pair_types,
)
from .device import resolve_device
from .eval_utils import eval_result_to_dict, evaluate_pair_loader
from .model.siamese import SiameseNetwork
from .utils import dump_json, ensure_dir

# -----------------------------------------------------------------------------
# Top-level configuration (edit these values directly)
# -----------------------------------------------------------------------------
# Select whether evaluation should read the small debug manifest or full manifest.
RUN_PROFILE = os.getenv("SIGNATURE_RUN_PROFILE", "small")  # "small" or "full"
# Choose evaluation subset. "test" should be used for final held-out reporting.
SPLIT = os.getenv("SIGNATURE_EVAL_SPLIT", "val")  # "train" | "val" | "test"

if RUN_PROFILE == "small":
    # Small-profile evaluation parameters for fast turnaround.
    MANIFEST_CSV = Path("siamese/manifests/bhsig260_small_manifest.csv")
    RUN_NAME_PREFIX = "small_debug"
    DEFAULT_OUTPUT_DIR = Path("siamese/results/small_debug")
    EVAL_BATCH_SIZE = 256
    THRESHOLD_POINTS = 400
    MAX_SKILLED_FORGERIES_PER_WRITER = 20
    MAX_RANDOM_IMPOSTORS_PER_WRITER = 10
    NUM_WORKERS = 0
elif RUN_PROFILE == "full":
    # Full-profile evaluation parameters for report-quality metrics.
    MANIFEST_CSV = Path("siamese/manifests/bhsig260_manifest.csv")
    RUN_NAME_PREFIX = "siamese_full"
    DEFAULT_OUTPUT_DIR = Path("siamese/results/full")
    EVAL_BATCH_SIZE = 128
    THRESHOLD_POINTS = 2000
    MAX_SKILLED_FORGERIES_PER_WRITER = 720
    MAX_RANDOM_IMPOSTORS_PER_WRITER = 200
    NUM_WORKERS = 2
else:
    raise ValueError("RUN_PROFILE must be either 'small' or 'full'.")

output_dir_override = os.getenv("SIGNATURE_OUTPUT_DIR")
OUTPUT_DIR = Path(output_dir_override) if output_dir_override else DEFAULT_OUTPUT_DIR
num_workers_override = os.getenv("SIGNATURE_NUM_WORKERS")
if num_workers_override is not None:
    NUM_WORKERS = int(num_workers_override)
# Leave as None to auto-load latest run checkpoint by prefix.
checkpoint_override = os.getenv("SIGNATURE_CHECKPOINT_PATH")
CHECKPOINT_PATH: Path | None = (
    Path(checkpoint_override) if checkpoint_override else None
)

# Image geometry must match training preprocessing.
IMAGE_HEIGHT = 155
IMAGE_WIDTH = 220
# Fallback embedding dimension if checkpoint metadata is incomplete.
EMBEDDING_DIM = 128
# Include random cross-writer impostors in addition to skilled forgeries.
INCLUDE_CROSS_WRITER_IMPOSTORS = True
# If None, use validation-selected threshold from checkpoint.
LOCKED_THRESHOLD: float | None = None
# Runtime backend selector.
DEVICE = "auto"  # auto | cuda | mps | cpu
# Seed for deterministic pair generation.
SEED = 42
# -----------------------------------------------------------------------------


def _save_roc_csv(path: Path, curve_payload: dict[str, list[float]]) -> None:
    # Save dense threshold sweep for downstream plotting/reporting.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["threshold", "far", "frr", "fpr", "tpr"])
        for row in zip(
            curve_payload["thresholds"],
            curve_payload["far_curve"],
            curve_payload["frr_curve"],
            curve_payload["fpr_curve"],
            curve_payload["tpr_curve"],
        ):
            writer.writerow(row)


def main() -> None:
    # Resolve compute device for evaluation workload.
    preferred_device = None if DEVICE == "auto" else DEVICE
    device = resolve_device(preferred=preferred_device)

    # Load trained model state.
    checkpoint_path = resolve_checkpoint_path(CHECKPOINT_PATH, RUN_NAME_PREFIX)
    checkpoint = load_checkpoint(checkpoint_path)
    embedding_dim = infer_embedding_dim(checkpoint, fallback=EMBEDDING_DIM)

    model = SiameseNetwork(embedding_dim=embedding_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    # Rebuild deterministic evaluation pair set from manifest split.
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

    # Evaluate in batches to control memory usage.
    pair_dataset = VerificationPairDataset(base_dataset=dataset, pairs=pairs)
    pair_loader = DataLoader(
        dataset=pair_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=NUM_WORKERS > 0,
    )

    # Use explicit threshold if provided, otherwise lock to validation-best threshold.
    locked_threshold = LOCKED_THRESHOLD
    if locked_threshold is None:
        checkpoint_threshold = checkpoint.get("best_threshold")
        if checkpoint_threshold is not None:
            locked_threshold = float(checkpoint_threshold)

    # Run FAR/FRR/ROC/EER computation.
    result = evaluate_pair_loader(
        model=model,
        pair_loader=pair_loader,
        device=device,
        threshold_points=THRESHOLD_POINTS,
        locked_threshold=locked_threshold,
    )

    # Persist both compact metrics and full ROC sweep.
    output_dir = ensure_dir(OUTPUT_DIR)
    metrics_path = output_dir / f"metrics_{SPLIT}.json"
    roc_path = output_dir / f"roc_{SPLIT}.csv"

    metric_payload = {
        "run_profile": RUN_PROFILE,
        "split": SPLIT,
        "device": device.type,
        "checkpoint": str(checkpoint_path),
        "locked_threshold": locked_threshold,
        "pair_counts": pair_counts,
        "metrics": eval_result_to_dict(result),
    }
    dump_json(metrics_path, metric_payload)

    _save_roc_csv(
        roc_path,
        curve_payload={
            "thresholds": result.thresholds.tolist(),
            "far_curve": result.far_curve.tolist(),
            "frr_curve": result.frr_curve.tolist(),
            "fpr_curve": result.fpr_curve.tolist(),
            "tpr_curve": result.tpr_curve.tolist(),
        },
    )

    print(f"Evaluation profile: {RUN_PROFILE}")
    print(f"Evaluation split: {SPLIT}")
    print(f"Device: {device.type}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Pairs: {result.n_pairs} | Pair composition: {pair_counts}")
    print(
        "Metrics: "
        f"AUC={result.auc:.4f}, "
        f"EER={result.eer:.4f}, "
        f"EER threshold={result.eer_threshold:.6f}, "
        f"FAR@EER={result.far_at_eer:.4f}, "
        f"FRR@EER={result.frr_at_eer:.4f}, "
        f"ACC@EER={result.accuracy_at_eer:.4f}"
    )
    if locked_threshold is not None:
        print(
            f"Locked threshold={locked_threshold:.6f} | "
            f"FAR={result.far_at_locked_threshold:.4f} | "
            f"FRR={result.frr_at_locked_threshold:.4f} | "
            f"ACC={result.accuracy_at_locked_threshold:.4f}"
        )
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved ROC CSV: {roc_path}")


if __name__ == "__main__":
    main()
