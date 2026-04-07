"""Path resolution helpers for portable dataset manifests."""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV_VAR = "SIGNATURE_DATA_ROOT"

# Keep a short list of known local roots so home/office machines work without
# repeatedly editing code. The cluster path can be supplied via environment var.
DEFAULT_DATA_ROOT_CANDIDATES = (
    Path(
        "/Users/smondal/Library/CloudStorage/GoogleDrive-smondal@nd.edu/My Drive/1. ND/1. PhD courses/Spring 26/Project/cedar-bhsig260"
    ),
    Path(
        "/Users/shuvashishmondal/Library/CloudStorage/GoogleDrive-smondal@nd.edu/My Drive/1. ND/1. PhD courses/Spring 26/Project/cedar-bhsig260"
    ),
)


def resolve_data_root(explicit: Path | None = None) -> Path:
    """Resolve the root directory that contains BHSig/CEDAR datasets."""

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())

    env_value = os.environ.get(DATA_ROOT_ENV_VAR)
    if env_value:
        candidates.append(Path(env_value).expanduser())

    candidates.extend(DEFAULT_DATA_ROOT_CANDIDATES)

    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists() and candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not resolve the signature dataset root. "
        f"Set {DATA_ROOT_ENV_VAR} or update the configured path. "
        f"Checked: {checked}"
    )


def resolve_manifest_image_path(
    image_path: str | Path,
    data_root: Path | None = None,
) -> Path:
    """Resolve a manifest path that may be absolute or relative to the data root."""

    path = Path(image_path).expanduser()
    if path.is_absolute():
        return path
    return resolve_data_root(explicit=data_root) / path


def to_manifest_path(image_path: Path, data_root: Path) -> str:
    """Convert an absolute image path into a portable manifest path when possible."""

    image_path = image_path.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    try:
        return str(image_path.relative_to(data_root))
    except ValueError:
        return str(image_path)
