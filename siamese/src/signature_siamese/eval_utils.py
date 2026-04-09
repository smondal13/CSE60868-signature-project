"""Shared evaluation helpers for validation and test scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import (
    build_thresholds,
    compute_auc,
    compute_eer,
    compute_threshold_metrics,
)


@dataclass(frozen=True)
class PairEvalResult:
    n_pairs: int
    auc: float
    eer: float
    eer_threshold: float
    far_at_eer: float
    frr_at_eer: float
    far_at_locked_threshold: float | None
    frr_at_locked_threshold: float | None
    thresholds: np.ndarray
    far_curve: np.ndarray
    frr_curve: np.ndarray
    fpr_curve: np.ndarray
    tpr_curve: np.ndarray


@dataclass(frozen=True)
class PairEvalInputs:
    distances: np.ndarray
    labels: np.ndarray
    pair_types: np.ndarray | None


def _to_numpy(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.array([], dtype=np.float64)
    return np.concatenate(parts, axis=0)


@torch.no_grad()
def collect_pair_outputs(
    model: torch.nn.Module,
    pair_loader: DataLoader,
    device: torch.device,
) -> PairEvalInputs:
    """Run pairwise verification and return distances/labels for downstream metrics."""
    model.eval()

    distance_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    pair_type_parts: list[np.ndarray] = []

    for batch in pair_loader:
        image_a = batch["image_a"].to(device)
        image_b = batch["image_b"].to(device)

        embedding_a = model.forward_once(image_a)
        embedding_b = model.forward_once(image_b)
        distances = torch.norm(embedding_a - embedding_b, p=2, dim=1)

        distance_parts.append(distances.detach().cpu().numpy())
        label_parts.append(batch["label"].detach().cpu().numpy())
        pair_types = batch.get("pair_type")
        if pair_types is not None:
            pair_type_parts.append(np.asarray(pair_types, dtype=object))

    all_distances = _to_numpy(distance_parts)
    all_labels = _to_numpy(label_parts).astype(np.int32)
    if all_distances.size == 0:
        raise RuntimeError("Pair loader produced no evaluation samples.")

    all_pair_types: np.ndarray | None = None
    if pair_type_parts:
        all_pair_types = _to_numpy(pair_type_parts).astype(object)

    return PairEvalInputs(
        distances=all_distances,
        labels=all_labels,
        pair_types=all_pair_types,
    )


def evaluate_pair_arrays(
    distances: np.ndarray,
    labels: np.ndarray,
    threshold_points: int,
    locked_threshold: float | None = None,
) -> PairEvalResult:
    """Compute FAR/FRR/ROC/EER statistics from precomputed pair distances."""
    if distances.size == 0:
        raise RuntimeError("No pair distances supplied for evaluation.")

    thresholds = build_thresholds(distances, points=threshold_points)
    threshold_metrics = compute_threshold_metrics(
        distances=distances,
        labels=labels,
        thresholds=thresholds,
    )
    eer = compute_eer(threshold_metrics)
    auc = compute_auc(threshold_metrics.fpr, threshold_metrics.tpr)

    far_locked: float | None = None
    frr_locked: float | None = None
    if locked_threshold is not None:
        single_metrics = compute_threshold_metrics(
            distances=distances,
            labels=labels,
            thresholds=np.array([locked_threshold], dtype=np.float64),
        )
        far_locked = float(single_metrics.far[0])
        frr_locked = float(single_metrics.frr[0])

    return PairEvalResult(
        n_pairs=int(distances.size),
        auc=float(auc),
        eer=float(eer.eer),
        eer_threshold=float(eer.threshold),
        far_at_eer=float(eer.far_at_eer),
        frr_at_eer=float(eer.frr_at_eer),
        far_at_locked_threshold=far_locked,
        frr_at_locked_threshold=frr_locked,
        thresholds=threshold_metrics.thresholds,
        far_curve=threshold_metrics.far,
        frr_curve=threshold_metrics.frr,
        fpr_curve=threshold_metrics.fpr,
        tpr_curve=threshold_metrics.tpr,
    )


@torch.no_grad()
def evaluate_pair_loader(
    model: torch.nn.Module,
    pair_loader: DataLoader,
    device: torch.device,
    threshold_points: int,
    locked_threshold: float | None = None,
) -> PairEvalResult:
    """Run pairwise verification and compute FAR/FRR/ROC/EER statistics."""
    outputs = collect_pair_outputs(model=model, pair_loader=pair_loader, device=device)
    return evaluate_pair_arrays(
        distances=outputs.distances,
        labels=outputs.labels,
        threshold_points=threshold_points,
        locked_threshold=locked_threshold,
    )


def eval_result_to_dict(result: PairEvalResult) -> dict[str, Any]:
    """Convert evaluation dataclass to JSON-serializable dictionary."""
    return {
        "n_pairs": result.n_pairs,
        "auc": result.auc,
        "eer": result.eer,
        "eer_threshold": result.eer_threshold,
        "far_at_eer": result.far_at_eer,
        "frr_at_eer": result.frr_at_eer,
        "far_at_locked_threshold": result.far_at_locked_threshold,
        "frr_at_locked_threshold": result.frr_at_locked_threshold,
    }
