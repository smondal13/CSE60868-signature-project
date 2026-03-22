"""Online hard negative mining from in-batch embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MiningOutput:
    pair_distances: torch.Tensor
    pair_labels: torch.Tensor
    n_positive_pairs: int
    n_negative_candidates: int
    n_hard_negatives: int


def _tensor_from_pairs(
    pairs: list[tuple[int, int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not pairs:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    idx_i = torch.tensor([pair[0] for pair in pairs], dtype=torch.long, device=device)
    idx_j = torch.tensor([pair[1] for pair in pairs], dtype=torch.long, device=device)
    return idx_i, idx_j


def mine_pairs_from_batch(
    embeddings: torch.Tensor,
    writer_ids: torch.Tensor,
    is_genuine: torch.Tensor,
    hard_negatives_per_positive: int = 2,
    include_cross_writer_negatives: bool = True,
) -> MiningOutput:
    """Create positive pairs and mine the hardest negatives in current batch.

    Positive definition:
      same writer and both signatures genuine.

    Negative candidates:
      - genuine/forgery mix for same writer
      - optional cross-writer pairs

    Hard negatives are chosen by smallest embedding distance.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [batch, embedding_dim].")

    batch_size = embeddings.shape[0]
    positive_pairs: list[tuple[int, int]] = []
    negative_candidates: list[tuple[int, int]] = []

    writer_ids = writer_ids.detach().cpu().tolist()
    is_genuine = is_genuine.detach().cpu().tolist()

    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            same_writer = writer_ids[i] == writer_ids[j]
            both_genuine = bool(is_genuine[i]) and bool(is_genuine[j])

            if same_writer and both_genuine:
                # Positive pairs are genuine-genuine signatures from the same writer.
                positive_pairs.append((i, j))
                continue

            mixed_same_writer = same_writer and (bool(is_genuine[i]) != bool(is_genuine[j]))
            cross_writer = (not same_writer) and include_cross_writer_negatives
            if mixed_same_writer or cross_writer:
                # Candidate negatives include skilled (same writer, mixed label)
                # and optionally random cross-writer pairs.
                negative_candidates.append((i, j))

    device = embeddings.device
    pos_i, pos_j = _tensor_from_pairs(positive_pairs, device=device)
    neg_i, neg_j = _tensor_from_pairs(negative_candidates, device=device)

    if pos_i.numel() > 0:
        positive_distances = torch.norm(embeddings[pos_i] - embeddings[pos_j], p=2, dim=1)
    else:
        positive_distances = torch.empty(0, dtype=embeddings.dtype, device=device)

    if neg_i.numel() > 0:
        negative_distances = torch.norm(embeddings[neg_i] - embeddings[neg_j], p=2, dim=1)
    else:
        negative_distances = torch.empty(0, dtype=embeddings.dtype, device=device)

    if negative_distances.numel() > 0 and positive_distances.numel() > 0:
        # Keep only closest negatives (hardest confusions) for stronger gradients.
        n_hard = min(negative_distances.numel(), hard_negatives_per_positive * positive_distances.numel())
        hard_values, hard_idx = torch.topk(negative_distances, k=n_hard, largest=False)
        hard_negatives = hard_values
        _ = hard_idx  # index retained for debugging extension if needed.
    else:
        hard_negatives = torch.empty(0, dtype=embeddings.dtype, device=device)

    all_distances = torch.cat([positive_distances, hard_negatives], dim=0)
    all_labels = torch.cat(
        [
            torch.ones_like(positive_distances),
            torch.zeros_like(hard_negatives),
        ],
        dim=0,
    )

    return MiningOutput(
        pair_distances=all_distances,
        pair_labels=all_labels,
        n_positive_pairs=int(positive_distances.numel()),
        n_negative_candidates=int(negative_distances.numel()),
        n_hard_negatives=int(hard_negatives.numel()),
    )
