"""Training entry point for Siamese signature verification.

This module intentionally uses top-level editable variables instead of argparse.
Edit the configuration block below, then run:

    SIGNATURE_DATA_ROOT=/path/to/cedar-bhsig260 \
    PYTHONPATH=siamese/src \
    conda run -n machine-learning python -m signature_siamese.train
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .data.datasets import SignatureDataset
from .data.pairs import VerificationPairDataset, build_verification_pairs
from .data.samplers import WriterAwareBatchSampler
from .device import autocast_enabled, resolve_device
from .eval_utils import eval_result_to_dict, evaluate_pair_loader
from .loss import ContrastiveLoss
from .mining import mine_pairs_from_batch
from .model.siamese import SiameseNetwork
from .utils import dump_json, ensure_dir, set_seed, timestamp_tag

# -----------------------------------------------------------------------------
# Top-level configuration (edit these values directly)
# -----------------------------------------------------------------------------
# "small": fast sanity-check run on the small manifest.
# "full": full-scale training settings.
RUN_PROFILE = "full"  # small | full

# Shared image geometry expected by the preprocessing pipeline.
IMAGE_HEIGHT = 155
IMAGE_WIDTH = 220
# Embedding dimension produced by the Siamese backbone.
EMBEDDING_DIM = 128
# Optimizer parameters.
LR = 1e-3
WEIGHT_DECAY = 1e-4
# Contrastive loss margin.
MARGIN = 1.0
# Gradient clipping threshold to stabilize optimization.
GRAD_CLIP_NORM = 5.0
# Number of hard negatives selected per positive pair in a batch.
HARD_NEGATIVES_PER_POSITIVE = 2
# Include cross-writer pairs as additional negatives during training.
INCLUDE_CROSS_WRITER_NEGATIVES = True
# Include cross-writer impostors in validation/test pair generation.
INCLUDE_CROSS_WRITER_IMPOSTORS = True
# Number of threshold points used when sweeping FAR/FRR curves.
THRESHOLD_POINTS = 400
# Reproducibility seed for sampling and initialization.
SEED = 42
# Device selector: auto prefers cuda, then mps, then cpu.
DEVICE = "auto"  # auto | cuda | mps | cpu

# Root output directory for runs/checkpoints/tensorboard.
OUTPUT_ROOT = Path("siamese/runs")

if RUN_PROFILE == "small":
    # Small-profile manifest and conservative settings for rapid debugging.
    MANIFEST_CSV = Path("siamese/manifests/bhsig260_small_manifest.csv")
    RUN_NAME = "small_debug"
    EPOCHS = 20
    WRITERS_PER_BATCH = 4
    SAMPLES_PER_WRITER = 6
    MIN_GENUINE_PER_WRITER = 2
    TRAIN_STEPS_PER_EPOCH = 80
    EVAL_BATCH_SIZE = 256
    MAX_SKILLED_FORGERIES_PER_WRITER = 60
    MAX_RANDOM_IMPOSTORS_PER_WRITER = 30
    NUM_WORKERS = 0
elif RUN_PROFILE == "full":
    # Full-profile manifest and larger settings for final experiments.
    MANIFEST_CSV = Path("siamese/manifests/bhsig260_manifest.csv")
    RUN_NAME = "siamese_full"
    EPOCHS = 30
    WRITERS_PER_BATCH = 8
    SAMPLES_PER_WRITER = 6
    MIN_GENUINE_PER_WRITER = 2
    TRAIN_STEPS_PER_EPOCH = 200
    EVAL_BATCH_SIZE = 128
    MAX_SKILLED_FORGERIES_PER_WRITER = 720
    MAX_RANDOM_IMPOSTORS_PER_WRITER = 200
    NUM_WORKERS = -1
else:
    raise ValueError("RUN_PROFILE must be either 'small' or 'full'.")
# -----------------------------------------------------------------------------


def _auto_workers(requested_workers: int, device: torch.device) -> int:
    # Respect explicit worker count when provided.
    if requested_workers >= 0:
        return requested_workers
    if device.type == "mps":
        # Keep conservative workers on Apple Silicon to avoid dataloader contention.
        return 2
    return 4


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    # Ensure deterministic behavior across Python/NumPy/Torch.
    set_seed(SEED)

    # Resolve the runtime compute backend.
    preferred_device = None if DEVICE == "auto" else DEVICE
    device = resolve_device(preferred=preferred_device)
    use_amp = autocast_enabled(device)

    # Configure dataloader behavior for the current hardware target.
    workers = _auto_workers(NUM_WORKERS, device=device)
    pin_memory = device.type == "cuda"

    # Create run-specific artifact directories.
    run_dir = ensure_dir(OUTPUT_ROOT / f"{RUN_NAME}_{timestamp_tag()}")
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    tensorboard_dir = ensure_dir(run_dir / "tensorboard")
    summary_path = run_dir / "training_summary.json"

    # Persist effective run configuration for reproducibility.
    dump_json(
        run_dir / "run_config.json",
        {
            "run_profile": RUN_PROFILE,
            "manifest_csv": str(MANIFEST_CSV),
            "output_root": str(OUTPUT_ROOT),
            "run_name": RUN_NAME,
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
            "embedding_dim": EMBEDDING_DIM,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "margin": MARGIN,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "writers_per_batch": WRITERS_PER_BATCH,
            "samples_per_writer": SAMPLES_PER_WRITER,
            "min_genuine_per_writer": MIN_GENUINE_PER_WRITER,
            "train_steps_per_epoch": TRAIN_STEPS_PER_EPOCH,
            "hard_negatives_per_positive": HARD_NEGATIVES_PER_POSITIVE,
            "include_cross_writer_negatives": INCLUDE_CROSS_WRITER_NEGATIVES,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "threshold_points": THRESHOLD_POINTS,
            "max_skilled_forgeries_per_writer": MAX_SKILLED_FORGERIES_PER_WRITER,
            "max_random_impostors_per_writer": MAX_RANDOM_IMPOSTORS_PER_WRITER,
            "include_cross_writer_impostors": INCLUDE_CROSS_WRITER_IMPOSTORS,
            "resolved_num_workers": workers,
            "seed": SEED,
            "device": DEVICE,
            "resolved_device": device.type,
        },
    )

    # Build split-specific datasets from the precomputed manifest.
    train_dataset = SignatureDataset(
        manifest_csv=MANIFEST_CSV,
        split="train",
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
    )
    val_dataset = SignatureDataset(
        manifest_csv=MANIFEST_CSV,
        split="val",
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
    )

    # Writer-aware sampler guarantees positive pairs per batch.
    train_batch_sampler = WriterAwareBatchSampler(
        dataset=train_dataset,
        writers_per_batch=WRITERS_PER_BATCH,
        samples_per_writer=SAMPLES_PER_WRITER,
        min_genuine_per_writer=MIN_GENUINE_PER_WRITER,
        steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
        seed=SEED,
    )

    # Batch sampler controls indices, so batch_size is not set in DataLoader.
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )

    # Validation pairs are deterministic to make epoch-to-epoch comparison stable.
    val_pairs = build_verification_pairs(
        dataset=val_dataset,
        seed=SEED,
        max_skilled_forgeries_per_writer=MAX_SKILLED_FORGERIES_PER_WRITER,
        max_random_impostors_per_writer=MAX_RANDOM_IMPOSTORS_PER_WRITER,
        include_cross_writer_impostors=INCLUDE_CROSS_WRITER_IMPOSTORS,
    )
    val_pair_dataset = VerificationPairDataset(val_dataset, val_pairs)
    val_pair_loader = DataLoader(
        dataset=val_pair_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )

    # Core model, loss, optimizer, and scheduler stack.
    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM).to(device)
    criterion = ContrastiveLoss(margin=MARGIN)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    best_val_eer = float("inf")
    best_threshold: float | None = None

    print(f"Training run directory: {run_dir}")
    print(f"Profile: {RUN_PROFILE}")
    print(f"Device: {device.type} | AMP: {use_amp} | Workers: {workers}")
    print(f"Train samples: {len(train_dataset)} | Val pairs: {len(val_pair_dataset)}")

    for epoch in range(EPOCHS):
        model.train()
        # Change epoch seed in sampler for controlled sampling variation.
        train_batch_sampler.set_epoch(epoch)

        epoch_loss_sum = 0.0
        epoch_updates = 0
        epoch_pos_pairs = 0
        epoch_neg_candidates = 0
        epoch_hard_negs = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        for batch in progress:
            # Move inputs and metadata to device.
            images = batch["image"].to(device)
            writer_ids = batch["writer_id"].to(device)
            script_ids = batch["script_id"].to(device)
            is_genuine = batch["is_genuine"].to(device)

            optimizer.zero_grad(set_to_none=True)
            # Compute embeddings and mine informative training pairs in-batch.
            with torch.autocast(device_type=device.type, enabled=use_amp):
                embeddings = model.forward_once(images)
                mined = mine_pairs_from_batch(
                    embeddings=embeddings,
                    writer_ids=writer_ids,
                    is_genuine=is_genuine,
                    script_ids=script_ids,
                    hard_negatives_per_positive=HARD_NEGATIVES_PER_POSITIVE,
                    include_cross_writer_negatives=INCLUDE_CROSS_WRITER_NEGATIVES,
                )
                if mined.pair_distances.numel() == 0:
                    continue
                loss = criterion(mined.pair_distances, mined.pair_labels)

            # Standard backward/update path with optional AMP.
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=GRAD_CLIP_NORM
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=GRAD_CLIP_NORM
                    )
                optimizer.step()

            # Aggregate per-step diagnostics.
            epoch_loss_sum += float(loss.detach().item())
            epoch_updates += 1
            epoch_pos_pairs += mined.n_positive_pairs
            epoch_neg_candidates += mined.n_negative_candidates
            epoch_hard_negs += mined.n_hard_negatives

            progress.set_postfix(
                loss=f"{loss.detach().item():.4f}",
                pos=mined.n_positive_pairs,
                hard=mined.n_hard_negatives,
            )

        if epoch_updates == 0:
            raise RuntimeError(
                "No optimization updates were applied in this epoch. "
                "Increase batch size/writers or relax mining constraints."
            )

        # Compute averaged training diagnostics.
        train_loss = epoch_loss_sum / epoch_updates
        avg_pos_pairs = epoch_pos_pairs / epoch_updates
        avg_neg_candidates = epoch_neg_candidates / epoch_updates
        avg_hard_negs = epoch_hard_negs / epoch_updates

        # Run deterministic validation evaluation and threshold sweep.
        val_result = evaluate_pair_loader(
            model=model,
            pair_loader=val_pair_loader,
            device=device,
            threshold_points=THRESHOLD_POINTS,
            locked_threshold=best_threshold,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", current_lr, epoch)
        writer.add_scalar("train/margin", MARGIN, epoch)
        writer.add_scalar("train/avg_positive_pairs", avg_pos_pairs, epoch)
        writer.add_scalar("train/avg_negative_candidates", avg_neg_candidates, epoch)
        writer.add_scalar("train/avg_hard_negatives", avg_hard_negs, epoch)

        writer.add_scalar("val/eer", val_result.eer, epoch)
        writer.add_scalar("val/auc", val_result.auc, epoch)
        writer.add_scalar("val/far_at_eer", val_result.far_at_eer, epoch)
        writer.add_scalar("val/frr_at_eer", val_result.frr_at_eer, epoch)
        writer.add_scalar("val/eer_threshold", val_result.eer_threshold, epoch)
        if val_result.far_at_locked_threshold is not None:
            writer.add_scalar(
                "val/far_locked_threshold", val_result.far_at_locked_threshold, epoch
            )
        if val_result.frr_at_locked_threshold is not None:
            writer.add_scalar(
                "val/frr_locked_threshold", val_result.frr_at_locked_threshold, epoch
            )

        # Update LR schedule based on validation EER.
        scheduler.step(val_result.eer)

        # Track the best checkpoint using validation EER.
        is_best = val_result.eer < best_val_eer
        if is_best:
            best_val_eer = val_result.eer
            best_threshold = val_result.eer_threshold

        # Save the latest state each epoch; keep a separate best checkpoint.
        checkpoint_payload: dict[str, object] = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_eer": best_val_eer,
            "best_threshold": best_threshold,
            "config": {
                "run_profile": RUN_PROFILE,
                "embedding_dim": EMBEDDING_DIM,
                "margin": MARGIN,
            },
            "resolved_device": device.type,
        }
        _save_checkpoint(checkpoint_dir / "latest.pt", checkpoint_payload)
        if is_best:
            _save_checkpoint(checkpoint_dir / "best.pt", checkpoint_payload)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_eer={val_result.eer:.4f} | "
            f"val_auc={val_result.auc:.4f} | "
            f"best_val_eer={best_val_eer:.4f}"
        )

    writer.close()

    # Persist a concise summary for downstream reporting.
    final_summary = {
        "run_dir": str(run_dir),
        "profile": RUN_PROFILE,
        "best_val_eer": best_val_eer,
        "best_threshold": best_threshold,
        "device": device.type,
        "train_samples": len(train_dataset),
        "val_pairs": len(val_pair_dataset),
        "last_val": eval_result_to_dict(val_result),
    }
    dump_json(summary_path, final_summary)

    print("Training complete.")
    print(f"Best validation EER: {best_val_eer:.6f}")
    print(f"Validation threshold (locked for test): {best_threshold}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
