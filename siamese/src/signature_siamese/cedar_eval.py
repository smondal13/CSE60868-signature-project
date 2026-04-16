"""Shared helpers for CEDAR external evaluation modes."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from .data.indexing import SignatureRecord
from .data.pairs import VerificationPair
from .data.transforms import SignaturePreprocessor
from .paths import resolve_manifest_image_path


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


def select_cedar_root(data_root: Path, dirname: str) -> Path:
    """Resolve the CEDAR dataset directory under the configured data root."""
    root = data_root / dirname
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Could not find CEDAR directory at {root}.")
    return root


def records_to_samples(records: list[SignatureRecord]) -> list[CedarSample]:
    """Convert normalized signature records into CEDAR sample objects."""
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


def filter_records_by_writer_ids(
    records: list[SignatureRecord],
    writer_ids: set[int],
) -> list[SignatureRecord]:
    """Select only the writer records needed for one evaluation split."""
    return [record for record in records if record.writer_id in writer_ids]


def split_cedar_writer_ids(
    records: list[SignatureRecord],
    *,
    seed: int,
    calibration_ratio: float,
    calibration_count: int | None = None,
) -> tuple[list[int], list[int]]:
    """Split CEDAR writers into disjoint calibration and test partitions.

    The split is deterministic given the seed and keeps writer identities
    disjoint so threshold calibration does not leak into the test writers.
    """
    writer_ids = sorted({record.writer_id for record in records})
    if len(writer_ids) < 2:
        raise RuntimeError("Mode B requires at least two CEDAR writers.")

    shuffled = writer_ids[:]
    random.Random(seed).shuffle(shuffled)

    if calibration_count is None:
        if not (0.0 < calibration_ratio < 1.0):
            raise ValueError("calibration_ratio must be in (0, 1) for Mode B.")
        calibration_count = int(round(len(shuffled) * calibration_ratio))

    calibration_count = max(1, min(len(shuffled) - 1, calibration_count))
    calibration_writers = sorted(shuffled[:calibration_count])
    test_writers = sorted(shuffled[calibration_count:])
    return calibration_writers, test_writers


def sample_unique_cartesian_pairs(
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


def build_cedar_verification_pairs(
    samples: list[CedarSample],
    *,
    seed: int,
    max_skilled_forgeries_per_writer: int | None,
    max_random_impostors_per_writer: int,
    include_cross_writer_impostors: bool,
) -> list[VerificationPair]:
    """Build deterministic CEDAR verification pairs.

    Time complexity is O(P_pos + P_skill + W * K_rand), where K_rand is the
    impostor cap per writer. This keeps cross-writer negatives manageable
    without weakening same-writer skilled-forgery evaluation.
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

        skilled_pairs = sample_unique_cartesian_pairs(
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
            random_pairs = sample_unique_cartesian_pairs(
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


def save_roc_csv(path: Path, curve_payload: dict[str, list[float]]) -> None:
    """Persist dense ROC/FAR/FRR curves for downstream plotting."""
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


def compute_locked_threshold_pair_type_metrics(
    distances: np.ndarray,
    labels: np.ndarray,
    pair_types: np.ndarray | None,
    locked_threshold: float,
) -> dict[str, float | int]:
    """Report per-pair-type acceptance rates at a fixed threshold."""
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
