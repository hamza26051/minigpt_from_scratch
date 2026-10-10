import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import SinusoidalPositionalEncoding
from .transformer_block import TransformerBlock


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=3,
        context_length=128,
        dropout=0.1
    ):
        super().__init__()
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position = SinusoidalPositionalEncoding(
            d_model,
            context_length
        )
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                d_model,
                n_heads,
                context_length,
                dropout
            )
            for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        _, T = idx.shape

        if T > self.context_length:
            idx = idx[:, -self.context_length:]
            T = idx.size(1)
            if targets is not None:
                targets = targets[:, -T:]

        x = self.token_embedding(idx)
        x = self.position(x)
        x = self.blocks(x)
        logits = self.lm_head(self.final_ln(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=200,
        temperature=0.8,
        top_k=None
    ):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                values, _ = torch.topk(
                    logits,
                    min(top_k, logits.size(-1))
                )
                logits[logits < values[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)

        return idx
