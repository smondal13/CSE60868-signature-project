"""Mode B CEDAR evaluation with threshold calibration on disjoint writers.

Mode B keeps the BHSig-trained model fixed, learns a threshold on a CEDAR
calibration-writer subset, then applies that threshold to held-out CEDAR test
writers. This isolates calibration from representation learning.

Run with:

    SIGNATURE_DATA_ROOT=/path/to/cedar-bhsig260 \
    PYTHONPATH=siamese/src \
    conda run -n machine-learning python -m signature_siamese.evaluate_cedar_mode_b
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader

from .cedar_eval import (
    CedarVerificationPairDataset,
    build_cedar_verification_pairs,
    compute_locked_threshold_pair_type_metrics,
    filter_records_by_writer_ids,
    records_to_samples,
    save_roc_csv,
    select_cedar_root,
    split_cedar_writer_ids,
)
from .checkpoints import (
    infer_embedding_dim,
    load_checkpoint,
    resolve_checkpoint_path,
)
from .data.indexing import merge_records, scan_cedar_root
from .data.pairs import iter_pair_types
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
OUTPUT_ROOT = Path(os.getenv("SIGNATURE_OUTPUT_ROOT", "siamese/results/cedar_mode_b"))

IMAGE_HEIGHT = 155
IMAGE_WIDTH = 220
EMBEDDING_DIM = 128
EVAL_BATCH_SIZE = int(os.getenv("SIGNATURE_EVAL_BATCH_SIZE", "128"))
THRESHOLD_POINTS = int(os.getenv("SIGNATURE_THRESHOLD_POINTS", "2000"))
NUM_WORKERS = int(os.getenv("SIGNATURE_NUM_WORKERS", "2"))
SEED = int(os.getenv("SIGNATURE_SEED", "42"))
DEVICE = os.getenv("SIGNATURE_DEVICE", "auto")  # auto | cuda | mps | cpu

CALIBRATION_RATIO = float(os.getenv("SIGNATURE_CEDAR_CALIBRATION_RATIO", "0.2"))
calibration_count_env = os.getenv("SIGNATURE_CEDAR_CALIBRATION_WRITERS")
CALIBRATION_WRITER_COUNT: int | None = (
    int(calibration_count_env) if calibration_count_env else None
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


def _build_pair_loader(
    samples,
    pairs,
    *,
    data_root: Path,
    device_type: str,
) -> DataLoader:
    pair_dataset = CedarVerificationPairDataset(
        samples=samples,
        pairs=pairs,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        data_root=data_root,
    )
    return DataLoader(
        dataset=pair_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device_type == "cuda"),
        persistent_workers=NUM_WORKERS > 0,
    )


def main() -> None:
    preferred_device = None if DEVICE == "auto" else DEVICE
    device = resolve_device(preferred=preferred_device)

    checkpoint_path = resolve_checkpoint_path(CHECKPOINT_PATH, RUN_NAME_PREFIX)
    checkpoint = load_checkpoint(checkpoint_path)
    embedding_dim = infer_embedding_dim(checkpoint, fallback=EMBEDDING_DIM)

    data_root = resolve_data_root(explicit=DATA_ROOT)
    cedar_root = select_cedar_root(data_root, CEDAR_DIRNAME)
    cedar_records = scan_cedar_root(cedar_root, path_root=data_root)
    if MAX_WRITERS is not None:
        cedar_records = merge_records(
            [[record for record in cedar_records if record.writer_id <= MAX_WRITERS]]
        )

    calibration_writers, test_writers = split_cedar_writer_ids(
        cedar_records,
        seed=SEED,
        calibration_ratio=CALIBRATION_RATIO,
        calibration_count=CALIBRATION_WRITER_COUNT,
    )

    calibration_records = filter_records_by_writer_ids(
        cedar_records,
        set(calibration_writers),
    )
    test_records = filter_records_by_writer_ids(
        cedar_records,
        set(test_writers),
    )

    calibration_samples = records_to_samples(calibration_records)
    calibration_pairs = build_cedar_verification_pairs(
        samples=calibration_samples,
        seed=SEED,
        max_skilled_forgeries_per_writer=MAX_SKILLED_FORGERIES_PER_WRITER,
        max_random_impostors_per_writer=MAX_RANDOM_IMPOSTORS_PER_WRITER,
        include_cross_writer_impostors=INCLUDE_CROSS_WRITER_IMPOSTORS,
    )
    test_samples = records_to_samples(test_records)
    test_pairs = build_cedar_verification_pairs(
        samples=test_samples,
        # Offset the RNG so test pair subsampling is deterministic but distinct.
        seed=SEED + 1,
        max_skilled_forgeries_per_writer=MAX_SKILLED_FORGERIES_PER_WRITER,
        max_random_impostors_per_writer=MAX_RANDOM_IMPOSTORS_PER_WRITER,
        include_cross_writer_impostors=INCLUDE_CROSS_WRITER_IMPOSTORS,
    )

    model = SiameseNetwork(embedding_dim=embedding_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    calibration_loader = _build_pair_loader(
        calibration_samples,
        calibration_pairs,
        data_root=data_root,
        device_type=device.type,
    )
    calibration_outputs = collect_pair_outputs(
        model=model,
        pair_loader=calibration_loader,
        device=device,
    )
    calibration_result = evaluate_pair_arrays(
        distances=calibration_outputs.distances,
        labels=calibration_outputs.labels,
        threshold_points=THRESHOLD_POINTS,
        locked_threshold=None,
    )
    calibration_threshold = calibration_result.eer_threshold

    test_loader = _build_pair_loader(
        test_samples,
        test_pairs,
        data_root=data_root,
        device_type=device.type,
    )
    test_outputs = collect_pair_outputs(
        model=model,
        pair_loader=test_loader,
        device=device,
    )
    test_result = evaluate_pair_arrays(
        distances=test_outputs.distances,
        labels=test_outputs.labels,
        threshold_points=THRESHOLD_POINTS,
        locked_threshold=calibration_threshold,
    )
    test_pair_type_locked_metrics = compute_locked_threshold_pair_type_metrics(
        distances=test_outputs.distances,
        labels=test_outputs.labels,
        pair_types=test_outputs.pair_types,
        locked_threshold=calibration_threshold,
    )

    output_dir = ensure_dir(OUTPUT_ROOT / checkpoint_path.parent.parent.name)
    metrics_path = output_dir / "metrics_mode_b.json"
    calibration_roc_path = output_dir / "roc_mode_b_calibration.csv"
    test_roc_path = output_dir / "roc_mode_b_test.csv"

    metric_payload = {
        "evaluation_mode": "cedar_mode_b",
        "device": device.type,
        "checkpoint": str(checkpoint_path),
        "cedar_root": str(cedar_root),
        "calibration_writers": calibration_writers,
        "test_writers": test_writers,
        "calibration_writer_count": len(calibration_writers),
        "test_writer_count": len(test_writers),
        "max_writers": MAX_WRITERS,
        "max_skilled_forgeries_per_writer": MAX_SKILLED_FORGERIES_PER_WRITER,
        "max_random_impostors_per_writer": MAX_RANDOM_IMPOSTORS_PER_WRITER,
        "calibration_pair_counts": dict(Counter(iter_pair_types(calibration_pairs))),
        "test_pair_counts": dict(Counter(iter_pair_types(test_pairs))),
        "calibration_threshold": calibration_threshold,
        "calibration_metrics": eval_result_to_dict(calibration_result),
        "test_metrics": eval_result_to_dict(test_result),
        "test_pair_type_locked_threshold_metrics": test_pair_type_locked_metrics,
    }
    dump_json(metrics_path, metric_payload)
    save_roc_csv(
        calibration_roc_path,
        curve_payload={
            "thresholds": calibration_result.thresholds.tolist(),
            "far_curve": calibration_result.far_curve.tolist(),
            "frr_curve": calibration_result.frr_curve.tolist(),
            "fpr_curve": calibration_result.fpr_curve.tolist(),
            "tpr_curve": calibration_result.tpr_curve.tolist(),
        },
    )
    save_roc_csv(
        test_roc_path,
        curve_payload={
            "thresholds": test_result.thresholds.tolist(),
            "far_curve": test_result.far_curve.tolist(),
            "frr_curve": test_result.frr_curve.tolist(),
            "fpr_curve": test_result.fpr_curve.tolist(),
            "tpr_curve": test_result.tpr_curve.tolist(),
        },
    )

    print("Mode B CEDAR evaluation complete.")
    print(f"Device: {device.type}")
    print(f"Checkpoint: {checkpoint_path}")
    print(
        f"Calibration writers: {len(calibration_writers)} | "
        f"Test writers: {len(test_writers)}"
    )
    print(f"Calibration threshold: {calibration_threshold:.6f}")
    print(
        "Calibration metrics: "
        f"AUC={calibration_result.auc:.4f}, "
        f"EER={calibration_result.eer:.4f}"
    )
    print(
        "Test metrics: "
        f"AUC={test_result.auc:.4f}, "
        f"EER={test_result.eer:.4f}, "
        f"FAR@calibration_threshold={test_result.far_at_locked_threshold:.4f}, "
        f"FRR@calibration_threshold={test_result.frr_at_locked_threshold:.4f}"
    )
    for key, value in sorted(test_pair_type_locked_metrics.items()):
        if isinstance(value, int):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved calibration ROC: {calibration_roc_path}")
    print(f"Saved test ROC: {test_roc_path}")


if __name__ == "__main__":
    main()
