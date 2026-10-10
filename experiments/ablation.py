import argparse
import math
import time

import numpy as np
import pandas as pd
import torch

from src.dataset import load_dataset, get_batch
from src.gpt import MiniGPT


SEED = 42


def run_experiment(
    train_ids,
    val_ids,
    tokenizer,
    device,
    d_model=64,
    n_heads=4,
    n_layers=4,
    context_length=128,
    steps=150,
    batch_size=16,
    learning_rate=3e-4
):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = MiniGPT(
        tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        context_length=context_length,
        dropout=0.0
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    start = time.time()
    model.train()

    for _ in range(steps):
        x, y = get_batch(
            train_ids,
            batch_size,
            context_length,
            device
        )
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )
        optimizer.step()

    elapsed = time.time() - start

    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(10):
            x, y = get_batch(
                val_ids,
                batch_size,
                context_length,
                device
            )
            _, loss = model(x, y)
            val_losses.append(loss.item())

    val_loss = float(np.mean(val_losses))
    tokens_seen = steps * batch_size * context_length

    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available()
        else float("nan")
    )

    return {
        "d_model": d_model,
        "heads": n_heads,
        "layers": n_layers,
        "context": context_length,
        "steps": steps,
        "batch": batch_size,
        "parameters": sum(
            p.numel() for p in model.parameters()
        ),
        "validation_loss": val_loss,
        "perplexity": math.exp(min(val_loss, 20)),
        "training_seconds": elapsed,
        "tokens_per_second": tokens_seen / max(elapsed, 1e-9),
        "peak_memory_mb": peak_memory_mb
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="outputs/ablation_results_reproduced.csv"
    )
    args = parser.parse_args()

    _, tokenizer, train_ids, val_ids = load_dataset()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = {}

    def cached(n_heads=4, n_layers=4, context_length=128):
        key = (n_heads, n_layers, context_length)
        if key not in cache:
            cache[key] = run_experiment(
                train_ids,
                val_ids,
                tokenizer,
                device,
                n_heads=n_heads,
                n_layers=n_layers,
                context_length=context_length
            )
        return dict(cache[key])

    rows = []

    for depth in (2, 4, 6):
        row = cached(n_layers=depth)
        row.update({
            "experiment_family": "depth",
            "setting": depth
        })
        rows.append(row)

    for heads in (2, 4, 8):
        row = cached(n_heads=heads)
        row.update({
            "experiment_family": "heads",
            "setting": heads
        })
        rows.append(row)

    for context in (64, 128, 256):
        row = cached(context_length=context)
        row.update({
            "experiment_family": "context_length",
            "setting": context
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
