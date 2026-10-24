"""
Training Script for BrainDualMem-LLM
Supports continual learning with experience replay and REINFORCE controller.

Usage:
    python train.py --config configs/default.yaml --dataset vqa-rad --mode continual
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json

from models import DualMemoryReasoner, MultiModalVQAEncoder
from data import get_dataloader
from utils import ContinualLearningEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Train BrainDualMem-LLM")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default="vqa-rad")
    parser.add_argument("--mode", type=str, default="continual",
                        choices=["standard", "continual"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_loss(logits, target_ids, aux_losses, config):
    """Compute total loss: LM loss + replay + RL + consolidation."""
    # Language modeling loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = target_ids[:, 1:].contiguous()
    lm_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=0,
    )

    total_loss = lm_loss

    # Auxiliary losses
    if "replay_loss" in aux_losses:
        total_loss = total_loss + config["training"]["replay_loss_weight"] * aux_losses["replay_loss"]
    if "consolidation_loss" in aux_losses:
        total_loss = total_loss + config["training"]["consolidation_loss_weight"] * aux_losses["consolidation_loss"]

    return total_loss, {"lm_loss": lm_loss.item()}


def train_one_epoch(model, encoder, dataloader, optimizer, scheduler, config, device, epoch):
    model.train()
    encoder.train()
    total_loss = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)

        # Encode multi-modal input
        fused_hidden = encoder(images, input_ids)

        # Forward through dual-memory reasoner
        logits, aux_losses, actions = model(fused_hidden, teacher_forcing=True)

        # Compute loss (use input_ids as target for proof-of-concept)
        loss, loss_dict = compute_loss(logits, input_ids, aux_losses, config)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(encoder.parameters()),
            config["training"]["gradient_clip"],
        )
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(n_batches, 1)


def main():
    args = parse_args()
    config = load_config(args.config)

    # Override with CLI args
    if args.epochs: config["training"]["max_epochs"] = args.epochs
    if args.batch_size: config["training"]["batch_size"] = args.batch_size
    if args.lr: config["training"]["learning_rate"] = args.lr

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # Initialize models
    embed_dim = config["model"]["embed_dim"]
    encoder = MultiModalVQAEncoder(
        image_size=config["model"]["image_size"],
        patch_size=config["model"]["patch_size"],
        embed_dim=embed_dim,
        vocab_size=config["model"]["vocab_size"],
        max_text_len=config["model"]["max_text_len"],
    ).to(device)

    model = DualMemoryReasoner(
        embed_dim=embed_dim,
        llm_hidden=embed_dim,
        vocab_size=config["model"]["vocab_size"],
        buffer_size=config["model"]["buffer_size"],
        max_reasoning_steps=config["model"]["max_reasoning_steps"],
    ).to(device)

    # Optimizer
    optimizer = AdamW(
        list(model.parameters()) + list(encoder.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # DataLoader (proof-of-concept synthetic data)
    dataloader = get_dataloader(
        dataset_name=args.dataset,
        split="train",
        batch_size=config["training"]["batch_size"],
        num_workers=0,  # synthetic data, no multiprocessing needed
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=len(dataloader) * config["training"]["max_epochs"]
    )

    # Continual learning evaluator
    cl_evaluator = ContinualLearningEvaluator()

    # Training loop
    print(f"[INFO] Starting training for {config['training']['max_epochs']} epochs...")
    history = []

    for epoch in range(1, config["training"]["max_epochs"] + 1):
        avg_loss = train_one_epoch(
            model, encoder, dataloader, optimizer, scheduler, config, device, epoch
        )
        print(f"Epoch {epoch}/{config['training']['max_epochs']} — Avg Loss: {avg_loss:.4f}")
        history.append({"epoch": epoch, "avg_loss": avg_loss})

        # Save checkpoint
        if epoch % 5 == 0 or epoch == config["training"]["max_epochs"]:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"[INFO] Checkpoint saved: {ckpt_path}")

    # Save training history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save final model
    final_path = os.path.join(args.output_dir, "best_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": encoder.state_dict(),
        "config": config,
    }, final_path)
    print(f"[INFO] Final model saved: {final_path}")
    print("[INFO] Training complete!")


if __name__ == "__main__":
    main()
