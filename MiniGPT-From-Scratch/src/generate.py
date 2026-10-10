import argparse
from pathlib import Path
import torch

from .gpt import MiniGPT
from .tokenizer import CharacterTokenizer


def load_checkpoint(checkpoint_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

    config = checkpoint["config"]
    tokenizer = CharacterTokenizer(checkpoint["chars"])

    model = MiniGPT(
        checkpoint["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        context_length=config["context_length"],
        dropout=config["dropout"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, checkpoint, device


def generate_text(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
    seed=None
):
    if not prompt:
        raise ValueError("Prompt must not be empty.")

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device
    )

    output_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )
    return tokenizer.decode(output_ids[0].tolist())


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with the trained MiniGPT checkpoint."
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        default=str(
            Path(__file__).resolve().parents[1]
            / "checkpoints"
            / "minigpt.pt"
        )
    )
    args = parser.parse_args()

    model, tokenizer, checkpoint, device = load_checkpoint(
        args.checkpoint
    )

    text = generate_text(
        model,
        tokenizer,
        args.prompt,
        device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed
    )

    print(f"device: {device}")
    print(
        "model parameters:",
        f"{checkpoint.get('parameter_count', sum(p.numel() for p in model.parameters())):,}"
    )
    print("context length:", checkpoint["config"]["context_length"])
    print("temperature:", args.temperature)
    print("\n" + text)


if __name__ == "__main__":
    main()
