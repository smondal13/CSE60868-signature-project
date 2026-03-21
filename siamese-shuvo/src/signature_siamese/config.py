"""Dataclass-based run configuration for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    manifest_csv: Path
    image_height: int = 155
    image_width: int = 220


@dataclass(frozen=True)
class BatchConfig:
    writers_per_batch: int = 8
    samples_per_writer: int = 6
    min_genuine_per_writer: int = 2
    train_steps_per_epoch: int = 200
    num_workers: int = 4


@dataclass(frozen=True)
class MiningConfig:
    hard_negatives_per_positive: int = 2
    include_cross_writer_negatives: bool = True


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    margin: float = 1.0
    embedding_dim: int = 128
    grad_clip_norm: float = 5.0
    device: str = "auto"
    seed: int = 42
    eval_batch_size: int = 64


@dataclass(frozen=True)
class EvalConfig:
    eval_batch_size: int = 64
    threshold_points: int = 2000
    max_random_impostors_per_writer: int = 200
    include_cross_writer_impostors: bool = True
