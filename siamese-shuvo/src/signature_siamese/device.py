"""Device resolution helpers for local Apple Silicon and CUDA clusters."""

from __future__ import annotations

from typing import Optional

import torch


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    """Resolve compute device with explicit fallback order.

    Args:
        preferred: One of {"cuda", "mps", "cpu"}. If provided and available,
            it is used directly.

    Returns:
        torch.device selected from available backends.
    """
    if preferred:
        preferred = preferred.lower()
        if preferred == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            raise RuntimeError("Requested cuda device but CUDA is not available.")
        if preferred == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            raise RuntimeError("Requested mps device but MPS is not available.")
        if preferred == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Unknown preferred device '{preferred}'.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_enabled(device: torch.device) -> bool:
    """Enable mixed precision only on CUDA.

    MPS autocast behavior can vary across PyTorch versions, so we keep it disabled
    by default for stability in this project.
    """
    return device.type == "cuda"
