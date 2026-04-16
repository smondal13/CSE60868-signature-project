"""Core package for writer-independent signature verification."""

from __future__ import annotations

from typing import Any


def resolve_device(*args: Any, **kwargs: Any):
    """Lazily import torch-dependent device resolution.

    Keeping this import lazy makes lightweight utilities, such as dataset
    indexing, importable without immediately pulling in the Torch runtime.
    """
    from .device import resolve_device as _resolve_device

    return _resolve_device(*args, **kwargs)

__all__ = ["resolve_device"]
