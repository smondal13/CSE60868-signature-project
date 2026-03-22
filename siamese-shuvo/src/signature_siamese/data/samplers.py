"""Custom samplers for writer-aware dynamic batching."""

from __future__ import annotations

import random
from typing import Iterator

from torch.utils.data import Sampler

from .datasets import SignatureDataset


class WriterAwareBatchSampler(Sampler[list[int]]):
    """Build batches with multiple samples per writer.

    This sampler enforces a minimum number of genuine samples per writer so each
    batch provides positive pairs for contrastive learning.
    """

    def __init__(
        self,
        dataset: SignatureDataset,
        writers_per_batch: int,
        samples_per_writer: int,
        min_genuine_per_writer: int = 2,
        steps_per_epoch: int = 200,
        seed: int = 42,
    ) -> None:
        if writers_per_batch <= 0:
            raise ValueError("writers_per_batch must be > 0")
        if samples_per_writer <= 1:
            raise ValueError("samples_per_writer must be > 1")
        if min_genuine_per_writer < 2:
            raise ValueError("min_genuine_per_writer must be >= 2")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be > 0")

        self.dataset = dataset
        self.writers_per_batch = writers_per_batch
        self.samples_per_writer = samples_per_writer
        self.min_genuine_per_writer = min_genuine_per_writer
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = 0

        self.writer_keys = sorted(dataset.writer_to_indices)
        if len(self.writer_keys) < 2:
            raise RuntimeError("Writer-aware batching requires at least 2 writers.")

        for writer_key in self.writer_keys:
            if len(dataset.writer_to_genuine_indices.get(writer_key, [])) < 2:
                raise RuntimeError(
                    "Each writer needs >=2 genuine signatures for online pair mining. "
                    f"Writer '{writer_key}' does not satisfy this requirement."
                )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _pick_with_replacement(self, rng: random.Random, pool: list[int], k: int) -> list[int]:
        if not pool:
            raise RuntimeError("Cannot sample from an empty pool.")
        if len(pool) >= k:
            # Prefer sampling without replacement when enough candidates exist.
            return rng.sample(pool, k)
        # Fall back to replacement when pool is smaller than required sample count.
        return [rng.choice(pool) for _ in range(k)]

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)

        for _ in range(self.steps_per_epoch):
            selected_writers = self._pick_with_replacement(
                rng,
                list(range(len(self.writer_keys))),
                self.writers_per_batch,
            )

            batch_indices: list[int] = []
            for writer_idx in selected_writers:
                writer_key = self.writer_keys[writer_idx]
                genuine_pool = self.dataset.writer_to_genuine_indices[writer_key]
                all_pool = self.dataset.writer_to_indices[writer_key]

                # Force at least n_genuine samples to ensure positive pairs exist.
                n_genuine = min(self.min_genuine_per_writer, self.samples_per_writer)
                n_remaining = self.samples_per_writer - n_genuine

                chosen_genuine = self._pick_with_replacement(rng, genuine_pool, n_genuine)
                chosen_remaining = self._pick_with_replacement(rng, all_pool, n_remaining)
                batch_indices.extend(chosen_genuine + chosen_remaining)

            rng.shuffle(batch_indices)
            yield batch_indices
