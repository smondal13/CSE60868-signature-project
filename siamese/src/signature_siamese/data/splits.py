"""Writer-independent split assignment utilities."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path


def _split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    # Integer split with remainder assigned to test.
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def _script_from_writer_key(writer_key: str) -> str:
    return writer_key.split("_", maxsplit=1)[0]


def assign_writer_independent_splits(
    records: list[dict[str, str]],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Assign train/val/test split labels at writer granularity.

    The split is script-aware: Bengali and Hindi writers are split separately to
    preserve script proportion in each partition.
    """
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0):
        raise ValueError("Split ratios must be in (0, 1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.")

    script_to_writers: dict[str, list[str]] = defaultdict(list)
    for writer_key in sorted({row["writer_key"] for row in records}):
        script_to_writers[_script_from_writer_key(writer_key)].append(writer_key)

    writer_to_split: dict[str, str] = {}
    for script_index, script in enumerate(sorted(script_to_writers)):
        writers = list(script_to_writers[script])
        # Script-dependent seed offset preserves deterministic but distinct shuffles.
        rng = random.Random(seed + script_index)
        rng.shuffle(writers)

        train_count, val_count, _ = _split_counts(
            total=len(writers),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        for writer_key in writers[:train_count]:
            writer_to_split[writer_key] = "train"
        for writer_key in writers[train_count : train_count + val_count]:
            writer_to_split[writer_key] = "val"
        for writer_key in writers[train_count + val_count :]:
            writer_to_split[writer_key] = "test"

    output_rows: list[dict[str, str]] = []
    for row in records:
        # Propagate writer-level split assignment to each sample row.
        row_with_split = dict(row)
        row_with_split["split"] = writer_to_split[row["writer_key"]]
        output_rows.append(row_with_split)

    output_rows.sort(
        key=lambda row: (
            row["split"],
            row["script"],
            int(row["writer_id"]),
            -int(row["is_genuine"]),
            int(row["sample_index"]),
            row["file_name"],
        )
    )
    return output_rows


def write_manifest_csv(rows: list[dict[str, str]], output_csv: Path) -> None:
    """Write final manifest with split assignment."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "script",
        "writer_id",
        "writer_key",
        "image_path",
        "file_name",
        "is_genuine",
        "sample_index",
        "split",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_split(rows: list[dict[str, str]]) -> dict[str, int]:
    """Count samples per split for quick validation checks."""
    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        split_name = row["split"]
        counts[split_name] = counts.get(split_name, 0) + 1
    return counts
