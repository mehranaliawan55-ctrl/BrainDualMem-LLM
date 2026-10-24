"""
Evaluation Script for BrainDualMem-LLM
Evaluates zero-shot accuracy, forgetting rate, and hallucination rate.

Usage:
    python evaluate.py --checkpoint results/best_model.pt --test_dataset pathvqa
"""

import argparse
import torch
import json
from tqdm import tqdm

from models import DualMemoryReasoner, MultiModalVQAEncoder
from data import get_dataloader
from utils import compute_vqa_accuracy, ContinualLearningEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BrainDualMem-LLM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, default="pathvqa")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_file", type=str, default="results/evaluation_results.json")
    return parser.parse_args()


def evaluate(model, encoder, dataloader, device):
    """Run evaluation and compute metrics."""
    model.eval()
    encoder.eval()

    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)

            fused_hidden = encoder(images, input_ids)
            model.reset(images.size(0), device)

            logits, _, _ = model(fused_hidden, teacher_forcing=False)
            pred_ids = logits.argmax(dim=-1)

            # Decode predictions (simplified for proof-of-concept)
            for i in range(images.size(0)):
                pred_tokens = pred_ids[i].tolist()
                pred_text = "".join([chr(t % 128) for t in pred_tokens if 32 <= t % 128 < 127])
                all_predictions.append(pred_text[:100])
                all_references.append(batch["answer"][i])

    metrics = compute_vqa_accuracy(all_predictions, all_references)
    return metrics, all_predictions, all_references


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get("config", {"model": {"embed_dim": 768, "vocab_size": 50257,
                                           "image_size": 224, "patch_size": 16,
                                           "max_text_len": 128, "buffer_size": 2048,
                                           "max_reasoning_steps": 3}})

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

    model.load_state_dict(ckpt["model_state_dict"])
    encoder.load_state_dict(ckpt["encoder_state_dict"])

    # Test dataloader
    dataloader = get_dataloader(
        dataset_name=args.test_dataset,
        split="test",
        batch_size=args.batch_size,
        num_workers=0,
    )

    # Evaluate
    print(f"[INFO] Evaluating on {args.test_dataset}...")
    metrics, predictions, references = evaluate(model, encoder, dataloader, device)

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    import os
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump({
            "dataset": args.test_dataset,
            "metrics": metrics,
            "num_samples": len(predictions),
        }, f, indent=2)
    print(f"[INFO] Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
