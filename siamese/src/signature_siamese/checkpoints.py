"""Checkpoint-loading helpers shared by train/evaluation entry points."""

from __future__ import annotations

from pathlib import Path

import torch


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, object]:
    """Load a checkpoint payload with metadata preserved."""
    # weights_only=False is required because checkpoints include config dicts.
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected checkpoint format at {path}.")
    return payload


def infer_embedding_dim(checkpoint: dict[str, object], fallback: int) -> int:
    """Infer embedding dimension from checkpoint metadata when available."""
    config = checkpoint.get("config", {})
    if isinstance(config, dict) and "embedding_dim" in config:
        return int(config["embedding_dim"])
    return fallback


def resolve_latest_checkpoint_by_prefix(
    prefix: str, runs_root: Path = Path("siamese/runs")
) -> Path:
    """Resolve the newest run directory whose name starts with the given prefix."""
    candidates = sorted(
        [
            path
            for path in runs_root.iterdir()
            if path.is_dir() and path.name.startswith(f"{prefix}_")
        ],
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No run directories found under {runs_root} with prefix '{prefix}_'."
        )

    latest = candidates[-1]
    checkpoint = latest / "checkpoints" / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint at expected path: {checkpoint}")
    return checkpoint


def resolve_checkpoint_path(
    checkpoint_path: Path | None,
    run_name_prefix: str,
) -> Path:
    """Resolve explicit or latest-by-prefix checkpoint path."""
    if checkpoint_path is not None:
        return checkpoint_path
    return resolve_latest_checkpoint_by_prefix(run_name_prefix)
