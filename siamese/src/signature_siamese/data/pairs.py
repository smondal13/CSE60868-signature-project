"""Verification pair generation and pair dataset implementation."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterator

from torch.utils.data import Dataset

from ..paths import resolve_manifest_image_path
from .datasets import SignatureDataset


@dataclass(frozen=True)
class VerificationPair:
    idx_a: int
    idx_b: int
    label: int
    pair_type: str


def _sample_unique_cartesian_pairs(
    rng: random.Random,
    left: list[int],
    right: list[int],
    max_pairs: int | None,
) -> list[tuple[int, int]]:
    """Sample pairs from Cartesian product without materializing full large matrices."""
    if not left or not right:
        return []

    full_count = len(left) * len(right)
    if max_pairs is None or full_count <= max_pairs:
        return [(i, j) for i in left for j in right]

    sampled: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    while len(out) < max_pairs:
        # Draw until enough unique pairs are collected.
        i = rng.choice(left)
        j = rng.choice(right)
        pair = (i, j)
        if pair in sampled:
            continue
        sampled.add(pair)
        out.append(pair)
    return out


def build_verification_pairs(
    dataset: SignatureDataset,
    seed: int = 42,
    max_skilled_forgeries_per_writer: int | None = None,
    max_random_impostors_per_writer: int = 200,
    include_cross_writer_impostors: bool = True,
) -> list[VerificationPair]:
    """Build deterministic verification pairs for val/test evaluation.

    Positive pairs:
      - genuine vs genuine of the same writer.

    Negative pairs:
      - genuine vs skilled forgery of the same writer.
      - optional same-script random cross-writer genuine-vs-genuine impostor pairs.
    """
    rng = random.Random(seed)
    pairs: list[VerificationPair] = []

    writer_keys = sorted(dataset.writer_to_indices)

    for writer_key in writer_keys:
        genuine = list(dataset.writer_to_genuine_indices.get(writer_key, []))
        forgery = list(dataset.writer_to_forgery_indices.get(writer_key, []))

        # Genuine-vs-genuine positive pairs for FRR estimation.
        for idx_a, idx_b in combinations(genuine, 2):
            pairs.append(VerificationPair(idx_a=idx_a, idx_b=idx_b, label=1, pair_type="positive"))

        # Genuine-vs-forgery negatives for FAR on skilled forgeries.
        skilled_pairs = _sample_unique_cartesian_pairs(
            rng=rng,
            left=genuine,
            right=forgery,
            max_pairs=max_skilled_forgeries_per_writer,
        )
        for idx_a, idx_b in skilled_pairs:
            pairs.append(VerificationPair(idx_a=idx_a, idx_b=idx_b, label=0, pair_type="skilled"))

    if include_cross_writer_impostors:
        # Cross-writer impostors model random forgery attempts while keeping the
        # script fixed so the task does not get artificially easier.
        all_genuine_by_writer = {
            writer_key: list(dataset.writer_to_genuine_indices.get(writer_key, []))
            for writer_key in writer_keys
        }
        script_to_writer_keys: dict[str, list[str]] = defaultdict(list)
        for writer_key in writer_keys:
            script = writer_key.split("_", maxsplit=1)[0]
            script_to_writer_keys[script].append(writer_key)

        seen_random_impostors: set[tuple[int, int]] = set()

        for writer_key in writer_keys:
            own = all_genuine_by_writer[writer_key]
            script = writer_key.split("_", maxsplit=1)[0]
            other = [
                idx
                for other_writer in script_to_writer_keys[script]
                if other_writer != writer_key
                for idx in all_genuine_by_writer[other_writer]
            ]
            random_pairs = _sample_unique_cartesian_pairs(
                rng=rng,
                left=own,
                right=other,
                max_pairs=max_random_impostors_per_writer,
            )
            for idx_a, idx_b in random_pairs:
                pair_key = (min(idx_a, idx_b), max(idx_a, idx_b))
                if pair_key in seen_random_impostors:
                    continue
                seen_random_impostors.add(pair_key)
                pairs.append(
                    VerificationPair(
                        idx_a=idx_a,
                        idx_b=idx_b,
                        label=0,
                        pair_type="random_impostor",
                    )
                )

    rng.shuffle(pairs)
    return pairs


class VerificationPairDataset(Dataset[dict[str, Any]]):
    """Dataset that serves deterministic verification pairs."""

    def __init__(
        self,
        base_dataset: SignatureDataset,
        pairs: list[VerificationPair],
        query_transform: Any | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.pairs = pairs
        self.query_transform = query_transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]

        sample_a = self.base_dataset.samples[pair.idx_a]
        sample_b = self.base_dataset.samples[pair.idx_b]

        image_a_pil = self.base_dataset.preprocessor.load_image(
            resolve_manifest_image_path(
                sample_a.image_path,
                data_root=self.base_dataset.data_root,
            )
        )
        image_b_pil = self.base_dataset.preprocessor.load_image(
            resolve_manifest_image_path(
                sample_b.image_path,
                data_root=self.base_dataset.data_root,
            )
        )
        if self.query_transform is not None:
            image_b_pil = self.query_transform(image_b_pil)

        image_a = self.base_dataset.preprocessor.to_tensor(image_a_pil)
        image_b = self.base_dataset.preprocessor.to_tensor(image_b_pil)

        return {
            "image_a": image_a,
            "image_b": image_b,
            "label": pair.label,
            "pair_type": pair.pair_type,
            "writer_key_a": sample_a.writer_key,
            "writer_key_b": sample_b.writer_key,
        }


def iter_pair_types(pairs: list[VerificationPair]) -> Iterator[str]:
    """Expose pair types for quick diagnostics."""
    for pair in pairs:
        yield pair.pair_type
