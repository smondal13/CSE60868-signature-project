"""Build writer-independent train/val/test manifests for BHSig.

This module intentionally uses top-level editable variables instead of argparse.
Edit the configuration block below, then run:

    PYTHONPATH=siamese-shuvo/src conda run -n machine-learning python -m signature_siamese.prepare_data
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .data.indexing import (
    SignatureRecord,
    merge_records,
    scan_bhsig_root,
    write_records_csv,
)
from .data.splits import (
    assign_writer_independent_splits,
    summarize_split,
    write_manifest_csv,
)
from .utils import dump_json

# -----------------------------------------------------------------------------
# Top-level configuration (edit these values directly)
# -----------------------------------------------------------------------------
# Root directory for Bengali signatures. The resolver accepts either:
# 1) a direct writer root (.../BHSig260-Bengali), or
# 2) a parent that contains one writer-root child folder.
BENGALI_ROOT = Path(
    "/Users/shuvashishmondal/Library/CloudStorage/GoogleDrive-smondal@nd.edu/My Drive/1. ND/1. PhD courses/Spring 26/Project/cedar-bhsig260/BHSig260-Bengali"
)
# Root directory for Hindi signatures (same resolution behavior as above).
HINDI_ROOT = Path(
    "/Users/shuvashishmondal/Library/CloudStorage/GoogleDrive-smondal@nd.edu/My Drive/1. ND/1. PhD courses/Spring 26/Project/cedar-bhsig260/BHSig260-Hindi"
)

# Choose one: "full" (all signatures) or "small" (sanity-check subset).
DATASET_PROFILE = "small"

# Writer-level split ratios. These are applied per script to preserve script balance.
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
# Global seed used for deterministic split assignment.
SEED = 42

# Outputs are profile-specific to avoid accidental overwrite.
# RAW_INDEX_CSV contains all parsed samples before split assignment.
RAW_INDEX_CSV = Path("siamese-shuvo/manifests/bhsig260_index_raw.csv")
# FULL_* outputs are produced when DATASET_PROFILE == "full".
FULL_MANIFEST_CSV = Path("siamese-shuvo/manifests/bhsig260_manifest.csv")
FULL_STATS_JSON = Path("siamese-shuvo/manifests/bhsig260_manifest_stats.json")
# SMALL_* outputs are produced when DATASET_PROFILE == "small".
SMALL_MANIFEST_CSV = Path("siamese-shuvo/manifests/bhsig260_small_manifest.csv")
SMALL_STATS_JSON = Path("siamese-shuvo/manifests/bhsig260_small_manifest_stats.json")

# Small-profile controls.
# Number of writers kept per script in the small-profile subset.
SMALL_MAX_WRITERS_PER_SCRIPT = 20
# Per selected writer, keep this many genuine signatures.
SMALL_MAX_GENUINE_PER_WRITER = 10
# Per selected writer, keep this many forged signatures.
SMALL_MAX_FORGERY_PER_WRITER = 10
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestTargets:
    manifest_csv: Path
    stats_json: Path


def _looks_like_writer_root(path: Path) -> bool:
    # A writer root is defined as a folder that directly contains numeric writer dirs.
    if not path.exists() or not path.is_dir():
        return False
    return any(child.is_dir() and child.name.isdigit() for child in path.iterdir())


def resolve_writer_root(path: Path) -> Path:
    """Resolve user-provided path to a directory that directly contains writer dirs."""
    if _looks_like_writer_root(path):
        return path

    candidates = [child for child in path.iterdir() if child.is_dir()]
    for candidate in candidates:
        if _looks_like_writer_root(candidate):
            return candidate

    raise RuntimeError(
        f"Could not resolve writer root under {path}. "
        "Expected directory with numeric writer subdirectories."
    )


def _select_profile_targets(profile: str) -> ManifestTargets:
    profile = profile.strip().lower()
    if profile == "full":
        return ManifestTargets(
            manifest_csv=FULL_MANIFEST_CSV, stats_json=FULL_STATS_JSON
        )
    if profile == "small":
        return ManifestTargets(
            manifest_csv=SMALL_MANIFEST_CSV, stats_json=SMALL_STATS_JSON
        )
    raise ValueError("DATASET_PROFILE must be either 'full' or 'small'.")


def _subsample_records(records: list[SignatureRecord]) -> list[SignatureRecord]:
    """Build a deterministic small subset for quick pipeline validation."""
    # Group records by script -> writer to keep script and writer provenance explicit.
    records_by_script_writer: dict[str, dict[int, list[SignatureRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        records_by_script_writer[record.script][record.writer_id].append(record)

    selected: list[SignatureRecord] = []
    for script in sorted(records_by_script_writer):
        # Select a fixed writer prefix per script for deterministic small runs.
        writer_ids = sorted(records_by_script_writer[script])[
            :SMALL_MAX_WRITERS_PER_SCRIPT
        ]
        for writer_id in writer_ids:
            writer_records = records_by_script_writer[script][writer_id]
            genuine = sorted(
                [record for record in writer_records if record.is_genuine == 1],
                key=lambda record: (record.sample_index, record.file_name),
            )
            forgery = sorted(
                [record for record in writer_records if record.is_genuine == 0],
                key=lambda record: (record.sample_index, record.file_name),
            )

            if len(genuine) < 2:
                raise RuntimeError(
                    f"Writer {script}_{writer_id:03d} has <2 genuine samples in small-profile selection."
                )

            # Preserve label balance by slicing genuine/forgery separately.
            selected.extend(genuine[:SMALL_MAX_GENUINE_PER_WRITER])
            selected.extend(forgery[:SMALL_MAX_FORGERY_PER_WRITER])

    return merge_records([selected])


def validate_writer_independence(rows: list[dict[str, str]]) -> None:
    """Ensure no writer key appears in more than one split."""
    split_to_writers: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_to_writers[row["split"]].add(row["writer_key"])

    train_writers = split_to_writers.get("train", set())
    val_writers = split_to_writers.get("val", set())
    test_writers = split_to_writers.get("test", set())

    if train_writers & val_writers:
        raise RuntimeError("Writer leakage detected between train and val splits.")
    if train_writers & test_writers:
        raise RuntimeError("Writer leakage detected between train and test splits.")
    if val_writers & test_writers:
        raise RuntimeError("Writer leakage detected between val and test splits.")


def summarize_manifest(rows: list[dict[str, str]]) -> dict[str, object]:
    """Summarize sample counts, writer counts, and script/label distribution."""
    split_counts = summarize_split(rows)

    split_writer_counts: dict[str, int] = {}
    split_script_counts: dict[str, dict[str, int]] = {}
    split_label_counts: dict[str, dict[str, int]] = {}

    for split_name in {"train", "val", "test"}:
        split_rows = [row for row in rows if row["split"] == split_name]
        split_writer_counts[split_name] = len({row["writer_key"] for row in split_rows})

        script_counts: dict[str, int] = defaultdict(int)
        label_counts: dict[str, int] = {"genuine": 0, "forgery": 0}
        for row in split_rows:
            script_counts[row["script"]] += 1
            if int(row["is_genuine"]) == 1:
                label_counts["genuine"] += 1
            else:
                label_counts["forgery"] += 1

        split_script_counts[split_name] = dict(sorted(script_counts.items()))
        split_label_counts[split_name] = label_counts

    return {
        "sample_counts": split_counts,
        "writer_counts": split_writer_counts,
        "script_sample_counts": split_script_counts,
        "label_sample_counts": split_label_counts,
        "profile": DATASET_PROFILE,
        "total_rows": len(rows),
    }


def _records_to_rows(records: list[SignatureRecord]) -> list[dict[str, str]]:
    return [
        {
            "script": record.script,
            "writer_id": str(record.writer_id),
            "writer_key": record.writer_key,
            "image_path": record.image_path,
            "file_name": record.file_name,
            "is_genuine": str(record.is_genuine),
            "sample_index": str(record.sample_index),
        }
        for record in records
    ]


def main() -> None:
    targets = _select_profile_targets(DATASET_PROFILE)

    # Resolve the exact writer roots from user-provided top-level paths.
    bengali_root = resolve_writer_root(BENGALI_ROOT)
    hindi_root = resolve_writer_root(HINDI_ROOT)

    bengali_records = scan_bhsig_root(bengali_root, script="bengali")
    hindi_records = scan_bhsig_root(hindi_root, script="hindi")
    merged_records = merge_records([bengali_records, hindi_records])

    # Keep a full raw index for traceability, even when building the small profile.
    write_records_csv(merged_records, output_csv=RAW_INDEX_CSV)

    # Build either the full or reduced-profile working set.
    if DATASET_PROFILE == "small":
        profile_records = _subsample_records(merged_records)
    else:
        profile_records = merged_records

    # Assign strict writer-independent splits and persist artifacts.
    rows = _records_to_rows(profile_records)
    manifest_rows = assign_writer_independent_splits(
        records=rows,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )
    validate_writer_independence(manifest_rows)
    write_manifest_csv(manifest_rows, output_csv=targets.manifest_csv)

    summary = summarize_manifest(manifest_rows)
    dump_json(targets.stats_json, summary)

    print("Manifest generated:")
    print(f"  Profile: {DATASET_PROFILE}")
    print(f"  Raw index: {RAW_INDEX_CSV}")
    print(f"  Split manifest: {targets.manifest_csv}")
    print(f"  Stats JSON: {targets.stats_json}")
    print("Summary:")
    print(f"  Samples: {summary['sample_counts']}")
    print(f"  Writers: {summary['writer_counts']}")
    print(f"  Script sample counts: {summary['script_sample_counts']}")
    print(f"  Label sample counts: {summary['label_sample_counts']}")


if __name__ == "__main__":
    main()
