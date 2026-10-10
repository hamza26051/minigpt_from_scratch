import torch

from src.gpt import MiniGPT


def test_tiny_batch_can_overfit():
    torch.manual_seed(42)

    vocab_size = 12
    context_length = 16
    batch_size = 8

    one_x = torch.randint(
        0,
        vocab_size,
        (1, context_length)
    )
    one_y = torch.roll(one_x, shifts=-1, dims=1)
    x = one_x.repeat(batch_size, 1)
    y = one_y.repeat(batch_size, 1)

    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=32,
        n_heads=4,
        n_layers=2,
        context_length=context_length,
        dropout=0.0
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3
    )

    model.train()
    with torch.no_grad():
        _, initial_loss = model(x, y)
        initial_loss = initial_loss.item()

    for _ in range(150):
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, final_loss = model(x, y)
        final_loss = final_loss.item()

    assert final_loss < initial_loss * 0.5
