"""Scan BHSig directory trees and build a normalized manifest."""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ..paths import to_manifest_path

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
# Expected BHSig naming pattern:
#   <script>-S-<writer_id>-<F|G>-<sample_id>.<ext>
FILENAME_RE = re.compile(r"^[A-Za-z]-S-(\d+)-([FG])-(\d+)\.[A-Za-z0-9]+$")
CEDAR_FILENAME_RE = re.compile(
    r"^(original|forgeries)_(\d+)_(\d+)\.[A-Za-z0-9]+$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SignatureRecord:
    script: str
    writer_id: int
    writer_key: str
    image_path: str
    file_name: str
    is_genuine: int
    sample_index: int


def _normalize_script_label(script: str) -> str:
    value = script.strip().lower()
    if value not in {"bengali", "hindi", "cedar"}:
        raise ValueError(
            "Unknown script label. Expected one of {'bengali', 'hindi', 'cedar'}; "
            f"got '{script}'."
        )
    return value


def _iter_writer_dirs(root_dir: Path) -> list[Path]:
    # Numeric folder names are treated as writer identities.
    writer_dirs = [path for path in root_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    writer_dirs.sort(key=lambda p: int(p.name))
    return writer_dirs


def _iter_image_files(writer_dir: Path) -> list[Path]:
    # Restrict to common image extensions and keep deterministic order.
    image_files = [
        path
        for path in writer_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_files.sort(key=lambda p: p.name)
    return image_files


def scan_bhsig_root(
    root_dir: Path,
    script: str,
    path_root: Path | None = None,
) -> list[SignatureRecord]:
    """Scan one dataset root and return normalized signature records."""
    script = _normalize_script_label(script)
    records: list[SignatureRecord] = []

    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_dir}")

    writer_dirs = _iter_writer_dirs(root_dir)
    if not writer_dirs:
        raise RuntimeError(
            f"No numeric writer directories found in {root_dir}. "
            "Expected one directory per writer id."
        )

    for writer_dir in writer_dirs:
        writer_id = int(writer_dir.name)
        writer_key = f"{script}_{writer_id:03d}"

        for image_path in _iter_image_files(writer_dir):
            match = FILENAME_RE.match(image_path.name)
            if not match:
                # Ignore files that do not follow expected dataset naming.
                continue

            _, type_token, sample_token = match.groups()
            is_genuine = 1 if type_token == "G" else 0

            records.append(
                SignatureRecord(
                    script=script,
                    writer_id=writer_id,
                    writer_key=writer_key,
                    image_path=to_manifest_path(image_path, path_root)
                    if path_root is not None
                    else str(image_path.resolve()),
                    file_name=image_path.name,
                    is_genuine=is_genuine,
                    sample_index=int(sample_token),
                )
            )

    if not records:
        raise RuntimeError(f"No signature records parsed under: {root_dir}")
    return records


def scan_cedar_root(
    root_dir: Path,
    path_root: Path | None = None,
) -> list[SignatureRecord]:
    """Scan the CEDAR directory tree and return normalized signature records."""
    script = _normalize_script_label("cedar")
    records: list[SignatureRecord] = []

    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_dir}")

    writer_dirs = _iter_writer_dirs(root_dir)
    if not writer_dirs:
        raise RuntimeError(
            f"No numeric writer directories found in {root_dir}. "
            "Expected one directory per writer id."
        )

    for writer_dir in writer_dirs:
        writer_id = int(writer_dir.name)
        writer_key = f"{script}_{writer_id:03d}"

        for image_path in _iter_image_files(writer_dir):
            match = CEDAR_FILENAME_RE.match(image_path.name)
            if not match:
                # Ignore files that do not follow the expected CEDAR naming.
                continue

            type_token, file_writer_token, sample_token = match.groups()
            file_writer_id = int(file_writer_token)
            if file_writer_id != writer_id:
                raise RuntimeError(
                    f"CEDAR file {image_path.name} is under writer {writer_id}, "
                    f"but encodes writer {file_writer_id}."
                )

            records.append(
                SignatureRecord(
                    script=script,
                    writer_id=writer_id,
                    writer_key=writer_key,
                    image_path=to_manifest_path(image_path, path_root)
                    if path_root is not None
                    else str(image_path.resolve()),
                    file_name=image_path.name,
                    is_genuine=1 if type_token.lower() == "original" else 0,
                    sample_index=int(sample_token),
                )
            )

    if not records:
        raise RuntimeError(f"No CEDAR signature records parsed under: {root_dir}")
    return records


def merge_records(record_groups: Iterable[list[SignatureRecord]]) -> list[SignatureRecord]:
    """Merge record groups and return a globally sorted list."""
    merged: list[SignatureRecord] = []
    for group in record_groups:
        merged.extend(group)

    merged.sort(
        key=lambda row: (
            row.script,
            row.writer_id,
            -row.is_genuine,
            row.sample_index,
            row.file_name,
        )
    )
    return merged


def write_records_csv(records: list[SignatureRecord], output_csv: Path) -> None:
    """Write records to CSV without split labels."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "script",
        "writer_id",
        "writer_key",
        "image_path",
        "file_name",
        "is_genuine",
        "sample_index",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def read_records_csv(input_csv: Path) -> list[dict[str, str]]:
    """Read records CSV into a list of dictionaries."""
    with input_csv.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))
