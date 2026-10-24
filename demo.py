"""
Interactive Demo for BrainDualMem-LLM
Shows how the dual-memory system reasons over a medical image + question.

Usage:
    python demo.py --image examples/sample_xray.jpg --question "What abnormality is visible?"
"""

import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms

from models import DualMemoryReasoner, MultiModalVQAEncoder, ActionInterpreter


def parse_args():
    parser = argparse.ArgumentParser(description="BrainDualMem-LLM Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to medical image")
    parser.add_argument("--question", type=str, default="What abnormality is visible?",
                        help="Clinical question")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("  BrainDualMem-LLM: Brain-Inspired Dual-Memory Medical VQA")
    print("  Aligned with NSFC #62576037 (Prof. Xiuxing Li, BIT)")
    print("=" * 60)
    print(f"\nQuestion: {args.question}")
    print(f"Image: {args.image or '(synthetic demo image)'}")

    # Initialize models (proof-of-concept)
    embed_dim = 768
    vocab_size = 50257

    encoder = MultiModalVQAEncoder(
        image_size=224, patch_size=16, embed_dim=embed_dim,
        vocab_size=vocab_size, max_text_len=128,
    ).to(device)

    model = DualMemoryReasoner(
        embed_dim=embed_dim, llm_hidden=embed_dim, vocab_size=vocab_size,
        buffer_size=2048, max_reasoning_steps=3,
    ).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        print("[INFO] Loaded trained checkpoint.")

    model.eval()
    encoder.eval()

    # Prepare input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.image:
        image = Image.open(args.image).convert("RGB")
    else:
        image = Image.new("RGB", (224, 224), color=(128, 128, 128))

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Simple tokenization
    input_ids = torch.zeros(1, 128, dtype=torch.long, device=device)
    for i, c in enumerate(args.question[:128]):
        input_ids[0, i] = ord(c) % vocab_size

    # Run inference
    with torch.no_grad():
        fused_hidden = encoder(image_tensor, input_ids)
        model.reset(1, device)
        logits, aux_losses, actions = model(fused_hidden, teacher_forcing=False)

    # Interpret controller actions
    print("\n--- Reasoning Trace ---")
    for step, action in enumerate(actions):
        action_idx = action.mean().int().item()
        action_name = ActionInterpreter.interpret(action_idx)
        print(f"  Step {step+1}: Controller chose → {action_name}")

    # Generate answer (simplified)
    pred_ids = logits.argmax(dim=-1)[0].tolist()
    answer = "".join([chr(t % 128) for t in pred_ids if 32 <= t % 128 < 127])[:200]

    print(f"\n--- Generated Answer ---")
    print(f"  {answer or '(Proof-of-concept: full LLM backbone required for coherent text)'}")
    print("\n[Note] This demo uses a lightweight proof-of-concept backbone.")
    print("Full implementation requires LLaMA-2/Med-PaLM backbone + real medical VQA datasets.")
    print("This framework will be fully developed during Master's research at BIT.")


if __name__ == "__main__":
    main()
