"""Zero-shot CEDAR evaluation with the locked threshold transferred from BHSig.

Mode A keeps the model and threshold fixed. The evaluator scores the current
best BHSig checkpoint directly on unseen CEDAR writers without recalibration.

Run with:

    SIGNATURE_DATA_ROOT=/path/to/cedar-bhsig260 \
    PYTHONPATH=siamese/src \
    conda run -n machine-learning python -m signature_siamese.evaluate_cedar_zero_shot
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader, Dataset

from .cedar_eval import (
    CedarVerificationPairDataset,
    build_cedar_verification_pairs,
    compute_locked_threshold_pair_type_metrics,
    records_to_samples,
    save_roc_csv,
    select_cedar_root,
)
from .checkpoints import (
    infer_embedding_dim,
    load_checkpoint,
    resolve_checkpoint_path,
)
from .data.indexing import SignatureRecord, merge_records, scan_cedar_root
from .data.pairs import VerificationPair, iter_pair_types
from .device import resolve_device
from .eval_utils import (
    collect_pair_outputs,
    eval_result_to_dict,
    evaluate_pair_arrays,
)
from .model.siamese import SiameseNetwork
from .paths import resolve_data_root
from .utils import dump_json, ensure_dir

# -----------------------------------------------------------------------------
# Top-level configuration
# -----------------------------------------------------------------------------
RUN_NAME_PREFIX = os.getenv("SIGNATURE_RUN_NAME_PREFIX", "siamese_full_hpo_lr5e4_m075")
checkpoint_override = os.getenv("SIGNATURE_CHECKPOINT_PATH")
CHECKPOINT_PATH: Path | None = (
    Path(checkpoint_override) if checkpoint_override else None
)

DATA_ROOT: Path | None = None
CEDAR_DIRNAME = os.getenv("SIGNATURE_CEDAR_DIRNAME", "CEDAR")
OUTPUT_ROOT = Path(os.getenv("SIGNATURE_OUTPUT_ROOT", "siamese/results/cedar_zero_shot"))

IMAGE_HEIGHT = 155
IMAGE_WIDTH = 220
EMBEDDING_DIM = 128
EVAL_BATCH_SIZE = int(os.getenv("SIGNATURE_EVAL_BATCH_SIZE", "128"))
THRESHOLD_POINTS = int(os.getenv("SIGNATURE_THRESHOLD_POINTS", "2000"))
NUM_WORKERS = int(os.getenv("SIGNATURE_NUM_WORKERS", "2"))
SEED = int(os.getenv("SIGNATURE_SEED", "42"))
DEVICE = os.getenv("SIGNATURE_DEVICE", "auto")  # auto | cuda | mps | cpu

# Mode A intentionally transfers the threshold unchanged from the BHSig model.
threshold_override = os.getenv("SIGNATURE_LOCKED_THRESHOLD")
LOCKED_THRESHOLD: float | None = (
    float(threshold_override) if threshold_override else None
)

include_cross_writer_env = os.getenv("SIGNATURE_INCLUDE_CROSS_WRITER_IMPOSTORS", "1")
INCLUDE_CROSS_WRITER_IMPOSTORS = include_cross_writer_env != "0"

max_random_impostors_env = os.getenv("SIGNATURE_MAX_RANDOM_IMPOSTORS_PER_WRITER")
MAX_RANDOM_IMPOSTORS_PER_WRITER = (
    int(max_random_impostors_env) if max_random_impostors_env else 200
)

max_skilled_env = os.getenv("SIGNATURE_MAX_SKILLED_FORGERIES_PER_WRITER")
MAX_SKILLED_FORGERIES_PER_WRITER: int | None = (
    int(max_skilled_env) if max_skilled_env else None
)

max_writers_env = os.getenv("SIGNATURE_CEDAR_MAX_WRITERS")
MAX_WRITERS: int | None = int(max_writers_env) if max_writers_env else None
# -----------------------------------------------------------------------------


def main() -> None:
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
                "Mode A requires a locked threshold from the BHSig checkpoint or "
                "an explicit SIGNATURE_LOCKED_THRESHOLD override."
            )
        locked_threshold = float(checkpoint_threshold)

    data_root = resolve_data_root(explicit=DATA_ROOT)
    cedar_root = select_cedar_root(data_root, CEDAR_DIRNAME)
    cedar_records = scan_cedar_root(cedar_root, path_root=data_root)
    if MAX_WRITERS is not None:
        cedar_records = merge_records(
            [
                [
                    record
                    for record in cedar_records
                    if record.writer_id <= MAX_WRITERS
                ]
            ]
        )
    samples = records_to_samples(cedar_records)
    pairs = build_cedar_verification_pairs(
        samples=samples,
        seed=SEED,
        max_skilled_forgeries_per_writer=MAX_SKILLED_FORGERIES_PER_WRITER,
        max_random_impostors_per_writer=MAX_RANDOM_IMPOSTORS_PER_WRITER,
        include_cross_writer_impostors=INCLUDE_CROSS_WRITER_IMPOSTORS,
    )
    pair_counts = dict(Counter(iter_pair_types(pairs)))

    model = SiameseNetwork(embedding_dim=embedding_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    pair_dataset = CedarVerificationPairDataset(
        samples=samples,
        pairs=pairs,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        data_root=data_root,
    )
    pair_loader = DataLoader(
        dataset=pair_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=NUM_WORKERS > 0,
    )

    outputs = collect_pair_outputs(model=model, pair_loader=pair_loader, device=device)
    result = evaluate_pair_arrays(
        distances=outputs.distances,
        labels=outputs.labels,
        threshold_points=THRESHOLD_POINTS,
        locked_threshold=locked_threshold,
    )
    pair_type_locked_metrics = compute_locked_threshold_pair_type_metrics(
        distances=outputs.distances,
        labels=outputs.labels,
        pair_types=outputs.pair_types,
        locked_threshold=locked_threshold,
    )

    output_dir = ensure_dir(OUTPUT_ROOT / checkpoint_path.parent.parent.name)
    metrics_path = output_dir / "metrics_mode_a.json"
    roc_path = output_dir / "roc_mode_a.csv"

    metric_payload = {
        "evaluation_mode": "cedar_zero_shot_mode_a",
        "device": device.type,
        "checkpoint": str(checkpoint_path),
        "cedar_root": str(cedar_root),
        "locked_threshold": locked_threshold,
        "pair_counts": pair_counts,
        "pair_type_locked_threshold_metrics": pair_type_locked_metrics,
        "max_writers": MAX_WRITERS,
        "max_skilled_forgeries_per_writer": MAX_SKILLED_FORGERIES_PER_WRITER,
        "max_random_impostors_per_writer": MAX_RANDOM_IMPOSTORS_PER_WRITER,
        "metrics": eval_result_to_dict(result),
    }
    dump_json(metrics_path, metric_payload)
    save_roc_csv(
        roc_path,
        curve_payload={
            "thresholds": result.thresholds.tolist(),
            "far_curve": result.far_curve.tolist(),
            "frr_curve": result.frr_curve.tolist(),
            "fpr_curve": result.fpr_curve.tolist(),
            "tpr_curve": result.tpr_curve.tolist(),
        },
    )

    print("Mode A CEDAR zero-shot evaluation complete.")
    print(f"Device: {device.type}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Transferred threshold: {locked_threshold:.6f}")
    print(f"Pair composition: {pair_counts}")
    print(
        "Metrics: "
        f"AUC={result.auc:.4f}, "
        f"EER={result.eer:.4f}, "
        f"FAR@locked={result.far_at_locked_threshold:.4f}, "
        f"FRR@locked={result.frr_at_locked_threshold:.4f}"
    )
    for key, value in sorted(pair_type_locked_metrics.items()):
        if isinstance(value, int):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved ROC: {roc_path}")


if __name__ == "__main__":
    main()
