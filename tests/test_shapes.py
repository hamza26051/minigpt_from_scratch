import torch

from src.gpt import MiniGPT
from src.transformer_block import TransformerBlock


def test_transformer_block_shape():
    block = TransformerBlock(
        d_model=32,
        n_heads=4,
        context_length=16,
        dropout=0.0
    )
    x = torch.randn(4, 16, 32)
    out = block(x)
    assert out.shape == x.shape


def test_minigpt_logits_and_loss():
    vocab_size = 20
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=32,
        n_heads=4,
        n_layers=2,
        context_length=16,
        dropout=0.0
    )

    x = torch.randint(0, vocab_size, (4, 16))
    y = torch.randint(0, vocab_size, (4, 16))
    logits, loss = model(x, y)

    assert logits.shape == (4, 16, vocab_size)
    assert torch.isfinite(loss)
