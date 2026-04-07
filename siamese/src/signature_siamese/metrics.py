"""Verification metrics for biometric evaluation (FAR/FRR/ROC/EER)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdMetrics:
    thresholds: np.ndarray
    far: np.ndarray
    frr: np.ndarray
    tpr: np.ndarray
    fpr: np.ndarray


@dataclass(frozen=True)
class EERResult:
    eer: float
    threshold: float
    far_at_eer: float
    frr_at_eer: float


def build_thresholds(distances: np.ndarray, points: int = 2000) -> np.ndarray:
    """Create evenly spaced threshold candidates from min/max distances."""
    d_min = float(np.min(distances))
    d_max = float(np.max(distances))
    if d_min == d_max:
        d_max = d_min + 1e-6
    return np.linspace(d_min, d_max, num=points)


def compute_threshold_metrics(
    distances: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> ThresholdMetrics:
    """Compute FAR/FRR and ROC coordinates across thresholds."""
    labels = labels.astype(np.int32)

    is_positive = labels == 1
    is_negative = labels == 0

    n_pos = max(1, int(np.sum(is_positive)))
    n_neg = max(1, int(np.sum(is_negative)))

    far = np.zeros_like(thresholds, dtype=np.float64)
    frr = np.zeros_like(thresholds, dtype=np.float64)

    for i, threshold in enumerate(thresholds):
        # Accept signature as genuine if distance is at or below threshold.
        accepted = distances <= threshold

        false_accept = np.sum(accepted & is_negative)
        false_reject = np.sum((~accepted) & is_positive)

        far[i] = false_accept / n_neg
        frr[i] = false_reject / n_pos

    tpr = 1.0 - frr
    fpr = far

    return ThresholdMetrics(
        thresholds=thresholds,
        far=far,
        frr=frr,
        tpr=tpr,
        fpr=fpr,
    )


def compute_eer(metrics: ThresholdMetrics) -> EERResult:
    """Find the threshold where FAR and FRR are closest."""
    idx = int(np.argmin(np.abs(metrics.far - metrics.frr)))
    eer_value = float((metrics.far[idx] + metrics.frr[idx]) / 2.0)
    return EERResult(
        eer=eer_value,
        threshold=float(metrics.thresholds[idx]),
        far_at_eer=float(metrics.far[idx]),
        frr_at_eer=float(metrics.frr[idx]),
    )


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Compute ROC AUC with trapezoidal integration."""
    order = np.argsort(fpr)
    # NumPy 2.x prefers trapezoid; keep a fallback for older environments.
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y=tpr[order], x=fpr[order]))
    return float(np.trapz(y=tpr[order], x=fpr[order]))
