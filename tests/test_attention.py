import torch

from src.attention import SelfAttentionHead


def test_self_attention_shapes():
    torch.manual_seed(42)
    head = SelfAttentionHead(
        d_model=8,
        head_dim=4,
        context_length=8,
        dropout=0.0
    )
    x = torch.randn(2, 5, 8)

    out, weights = head(x)

    assert out.shape == (2, 5, 4)
    assert weights.shape == (2, 5, 5)
