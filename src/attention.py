import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, d_model, head_dim, context_length, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(d_model, head_dim, bias=False)
        self.query = nn.Linear(d_model, head_dim, bias=False)
        self.value = nn.Linear(d_model, head_dim, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_length, context_length))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        scores = scores.masked_fill(
            self.causal_mask[:T, :T] == 0,
            float("-inf")
        )
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v
        return out, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, context_length, dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        head_dim = d_model // n_heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(d_model, head_dim, context_length, dropout)
            for _ in range(n_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        results = [head(x) for head in self.heads]
        out = torch.cat([result[0] for result in results], dim=-1)
        out = self.dropout(self.proj(out))

        if return_attention:
            return out, [result[1] for result in results]
        return out
