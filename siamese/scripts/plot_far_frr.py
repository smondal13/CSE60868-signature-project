"""Plot FAR/FRR-vs-threshold curves for Siamese validation and test splits.

Usage:
    PYTHONPATH=siamese/src conda run -n machine-learning \
    python siamese/scripts/plot_far_frr.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "siamese" / "results" / "full"
OUTPUT_DIR = REPO_ROOT / "presentation_plots"
OUTPUT_PATH = OUTPUT_DIR / "far_frr_val_test.png"
FONT_SIZE = 20


def _read_curve(csv_path: Path) -> dict[str, list[float]]:
    thresholds: list[float] = []
    far_curve: list[float] = []
    frr_curve: list[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            thresholds.append(float(row["threshold"]))
            far_curve.append(float(row["far"]))
            frr_curve.append(float(row["frr"]))

    return {
        "thresholds": thresholds,
        "far": far_curve,
        "frr": frr_curve,
    }


def _read_metrics(metrics_path: Path) -> dict[str, float]:
    with metrics_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return payload["metrics"]


def main() -> None:
    val_curve = _read_curve(RESULTS_DIR / "roc_val.csv")
    test_curve = _read_curve(RESULTS_DIR / "roc_test.csv")
    val_metrics = _read_metrics(RESULTS_DIR / "metrics_val.json")
    test_metrics = _read_metrics(RESULTS_DIR / "metrics_test.json")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    split_payloads = [
        ("Validation", val_curve, val_metrics),
        ("Test", test_curve, test_metrics),
    ]

    for ax, (title, curve, metrics) in zip(axes, split_payloads):
        ax.plot(curve["thresholds"], curve["far"], label="FAR", linewidth=2)
        ax.plot(curve["thresholds"], curve["frr"], label="FRR", linewidth=2)
        eer_t = metrics["eer_threshold"]
        ax.axvline(
            eer_t,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"EER threshold ({eer_t:.3f})",
        )
        ax.set_title(f"{title} FAR/FRR vs Threshold")
        ax.set_xlabel("Threshold")
        ax.grid(alpha=0.25)
        ax.legend()

    axes[0].set_ylabel("Rate")
    fig.suptitle("Siamese Verification Operating Curves")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=220)
    plt.close(fig)

    print(f"Saved plot: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
