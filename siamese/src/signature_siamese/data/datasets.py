"""PyTorch datasets for signature samples and verification pairs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from ..paths import resolve_manifest_image_path
from .transforms import SignaturePreprocessor


@dataclass(frozen=True)
class SampleRecord:
    script: str
    writer_id: int
    writer_key: str
    image_path: Path
    file_name: str
    is_genuine: int
    sample_index: int
    split: str


def load_manifest_rows(manifest_csv: Path) -> list[dict[str, str]]:
    """Read the split manifest CSV into list form."""
    with manifest_csv.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def build_writer_label_map(rows: list[dict[str, str]]) -> dict[str, int]:
    """Build deterministic numeric writer labels from writer keys."""
    # Stable ordering keeps label ids consistent across runs.
    writer_keys = sorted({row["writer_key"] for row in rows})
    return {writer_key: idx for idx, writer_key in enumerate(writer_keys)}


def _row_to_record(row: dict[str, str]) -> SampleRecord:
    return SampleRecord(
        script=row["script"],
        writer_id=int(row["writer_id"]),
        writer_key=row["writer_key"],
        image_path=Path(row["image_path"]),
        file_name=row["file_name"],
        is_genuine=int(row["is_genuine"]),
        sample_index=int(row["sample_index"]),
        split=row["split"],
    )


class SignatureDataset(Dataset[dict[str, Any]]):
    """Single-image dataset filtered by split.

    This dataset is used in two modes:
    1) Train mode with writer-aware batching and in-batch mining.
    2) Base image source for verification-pair datasets.
    """

    def __init__(
        self,
        manifest_csv: Path,
        split: str,
        image_height: int = 155,
        image_width: int = 220,
        data_root: Path | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split '{split}'. Expected train/val/test.")

        self.manifest_csv = manifest_csv
        self.split = split
        self.data_root = data_root
        self.preprocessor = SignaturePreprocessor(
            image_height=image_height,
            image_width=image_width,
        )

        all_rows = load_manifest_rows(manifest_csv)
        self.writer_to_int = build_writer_label_map(all_rows)
        script_labels = sorted({row["script"] for row in all_rows})
        self.script_to_int = {
            script_label: idx for idx, script_label in enumerate(script_labels)
        }

        self.samples = [
            _row_to_record(row)
            for row in all_rows
            if row["split"] == split
        ]
        if not self.samples:
            raise RuntimeError(f"No rows found for split '{split}' in {manifest_csv}.")

        self.writer_to_indices: dict[str, list[int]] = {}
        self.writer_to_genuine_indices: dict[str, list[int]] = {}
        self.writer_to_forgery_indices: dict[str, list[int]] = {}

        for idx, sample in enumerate(self.samples):
            # Build lookup maps used by writer-aware sampler and pair generation.
            self.writer_to_indices.setdefault(sample.writer_key, []).append(idx)
            if sample.is_genuine:
                self.writer_to_genuine_indices.setdefault(sample.writer_key, []).append(idx)
            else:
                self.writer_to_forgery_indices.setdefault(sample.writer_key, []).append(idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_path = resolve_manifest_image_path(
            sample.image_path,
            data_root=self.data_root,
        )
        image = self.preprocessor(image_path)

        return {
            "image": image,
            "writer_id": self.writer_to_int[sample.writer_key],
            "script_id": self.script_to_int[sample.script],
            "script": sample.script,
            "writer_key": sample.writer_key,
            "is_genuine": sample.is_genuine,
            "path": str(image_path),
        }
