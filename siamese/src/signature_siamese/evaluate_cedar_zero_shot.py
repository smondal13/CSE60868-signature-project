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
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset

from .checkpoints import (
    infer_embedding_dim,
    load_checkpoint,
    resolve_checkpoint_path,
)
from .data.indexing import SignatureRecord, merge_records, scan_cedar_root
from .data.pairs import VerificationPair, iter_pair_types
from .data.transforms import SignaturePreprocessor
from .device import resolve_device
from .eval_utils import (
    collect_pair_outputs,
    eval_result_to_dict,
    evaluate_pair_arrays,
)
from .model.siamese import SiameseNetwork
from .paths import resolve_data_root, resolve_manifest_image_path
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


@dataclass(frozen=True)
class CedarSample:
    writer_id: int
    writer_key: str
    image_path: Path
    file_name: str
    is_genuine: int
    sample_index: int


class CedarVerificationPairDataset(Dataset[dict[str, Any]]):
    """Dataset that serves deterministic CEDAR verification pairs."""

    def __init__(
        self,
        samples: list[CedarSample],
        pairs: list[VerificationPair],
        image_height: int,
        image_width: int,
        data_root: Path | None = None,
    ) -> None:
        self.samples = samples
        self.pairs = pairs
        self.data_root = data_root
        self.preprocessor = SignaturePreprocessor(
            image_height=image_height,
            image_width=image_width,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        sample_a = self.samples[pair.idx_a]
        sample_b = self.samples[pair.idx_b]

        image_a = self.preprocessor(
            resolve_manifest_image_path(sample_a.image_path, data_root=self.data_root)
        )
        image_b = self.preprocessor(
            resolve_manifest_image_path(sample_b.image_path, data_root=self.data_root)
        )

        return {
            "image_a": image_a,
            "image_b": image_b,
            "label": pair.label,
            "pair_type": pair.pair_type,
            "writer_key_a": sample_a.writer_key,
            "writer_key_b": sample_b.writer_key,
        }


def _select_cedar_root(data_root: Path, dirname: str) -> Path:
    root = data_root / dirname
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Could not find CEDAR directory at {root}.")
    return root


def _sample_unique_cartesian_pairs(
    rng: random.Random,
    left: list[int],
    right: list[int],
    max_pairs: int | None,
) -> list[tuple[int, int]]:
    """Sample unique pairs from a Cartesian product without full materialization."""
    if not left or not right:
        return []

    full_count = len(left) * len(right)
    if max_pairs is None or full_count <= max_pairs:
        return [(i, j) for i in left for j in right]

    sampled: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    while len(out) < max_pairs:
        pair = (rng.choice(left), rng.choice(right))
        if pair in sampled:
            continue
        sampled.add(pair)
        out.append(pair)
    return out


def _records_to_samples(records: list[SignatureRecord]) -> list[CedarSample]:
    return [
        CedarSample(
            writer_id=record.writer_id,
            writer_key=record.writer_key,
            image_path=Path(record.image_path),
            file_name=record.file_name,
            is_genuine=record.is_genuine,
            sample_index=record.sample_index,
        )
        for record in records
    ]


def _build_cedar_verification_pairs(
    samples: list[CedarSample],
    *,
    seed: int,
    max_skilled_forgeries_per_writer: int | None,
    max_random_impostors_per_writer: int,
    include_cross_writer_impostors: bool,
) -> list[VerificationPair]:
    """Build the Mode A CEDAR verification set.

    Time complexity is O(P_pos + P_skill + W * K_rand), where K_rand is the
    impostor cap per writer. This keeps cross-writer negatives manageable
    without changing the harder same-writer skilled-forgery evaluation.
    """
    rng = random.Random(seed)
    pairs: list[VerificationPair] = []

    writer_to_genuine: dict[str, list[int]] = defaultdict(list)
    writer_to_forgery: dict[str, list[int]] = defaultdict(list)
    writer_keys: list[str] = []

    for idx, sample in enumerate(samples):
        if sample.writer_key not in writer_to_genuine and sample.writer_key not in writer_to_forgery:
            writer_keys.append(sample.writer_key)
        if sample.is_genuine:
            writer_to_genuine[sample.writer_key].append(idx)
        else:
            writer_to_forgery[sample.writer_key].append(idx)

    writer_keys.sort()
    for writer_key in writer_keys:
        genuine = writer_to_genuine.get(writer_key, [])
        forgery = writer_to_forgery.get(writer_key, [])

        for idx_a, idx_b in combinations(genuine, 2):
            pairs.append(
                VerificationPair(
                    idx_a=idx_a,
                    idx_b=idx_b,
                    label=1,
                    pair_type="positive",
                )
            )

        skilled_pairs = _sample_unique_cartesian_pairs(
            rng=rng,
            left=genuine,
            right=forgery,
            max_pairs=max_skilled_forgeries_per_writer,
        )
        for idx_a, idx_b in skilled_pairs:
            pairs.append(
                VerificationPair(
                    idx_a=idx_a,
                    idx_b=idx_b,
                    label=0,
                    pair_type="skilled",
                )
            )

    if include_cross_writer_impostors:
        seen_random_impostors: set[tuple[int, int]] = set()
        for writer_key in writer_keys:
            own = writer_to_genuine.get(writer_key, [])
            other = [
                idx
                for other_writer in writer_keys
                if other_writer != writer_key
                for idx in writer_to_genuine.get(other_writer, [])
            ]
            random_pairs = _sample_unique_cartesian_pairs(
                rng=rng,
                left=own,
                right=other,
                max_pairs=max_random_impostors_per_writer,
            )
            for idx_a, idx_b in random_pairs:
                pair_key = (min(idx_a, idx_b), max(idx_a, idx_b))
                if pair_key in seen_random_impostors:
                    continue
                seen_random_impostors.add(pair_key)
                pairs.append(
                    VerificationPair(
                        idx_a=idx_a,
                        idx_b=idx_b,
                        label=0,
                        pair_type="random_impostor",
                    )
                )

    rng.shuffle(pairs)
    return pairs


def _save_roc_csv(path: Path, curve_payload: dict[str, list[float]]) -> None:
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


def _compute_locked_threshold_pair_type_metrics(
    distances: np.ndarray,
    labels: np.ndarray,
    pair_types: np.ndarray | None,
    locked_threshold: float,
) -> dict[str, float | int]:
    """Report per-pair-type acceptance rates at the transferred threshold."""
    if pair_types is None:
        return {}

    accepted = distances <= locked_threshold
    metrics: dict[str, float | int] = {}

    for pair_type in sorted({str(value) for value in pair_types.tolist()}):
        mask = pair_types == pair_type
        metrics[f"{pair_type}_n_pairs"] = int(np.sum(mask))

        if pair_type == "positive":
            n_positive = max(1, int(np.sum(mask & (labels == 1))))
            false_reject = int(np.sum((~accepted) & mask & (labels == 1)))
            metrics[f"{pair_type}_frr_at_locked_threshold"] = false_reject / n_positive
        else:
            n_negative = max(1, int(np.sum(mask & (labels == 0))))
            false_accept = int(np.sum(accepted & mask & (labels == 0)))
            metrics[f"{pair_type}_far_at_locked_threshold"] = false_accept / n_negative

    return metrics


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
    cedar_root = _select_cedar_root(data_root, CEDAR_DIRNAME)
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
    samples = _records_to_samples(cedar_records)
    pairs = _build_cedar_verification_pairs(
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
    pair_type_locked_metrics = _compute_locked_threshold_pair_type_metrics(
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
