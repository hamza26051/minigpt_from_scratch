import torch

from src.attention import SelfAttentionHead


def test_future_attention_is_zero():
    torch.manual_seed(42)
    head = SelfAttentionHead(
        d_model=8,
        head_dim=4,
        context_length=8,
        dropout=0.0
    )
    x = torch.randn(2, 5, 8)

    _, weights = head(x)
    future = torch.triu(weights, diagonal=1)

    assert torch.allclose(
        future,
        torch.zeros_like(future),
        atol=1e-7
    )
