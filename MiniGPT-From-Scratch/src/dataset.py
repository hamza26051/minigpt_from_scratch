from pathlib import Path
import urllib.request
import torch

from .tokenizer import CharacterTokenizer

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_tiny_shakespeare(path="data/tinyshakespeare.txt"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(DATA_URL, path)
    return path


def load_dataset(path="data/tinyshakespeare.txt", train_fraction=0.9):
    path = download_tiny_shakespeare(path)
    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    tokenizer = CharacterTokenizer(chars)

    all_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_idx = int(train_fraction * len(all_ids))
    train_ids = all_ids[:split_idx]
    val_ids = all_ids[split_idx:]
    return text, tokenizer, train_ids, val_ids


def get_batch(source, batch_size, block_size, device="cpu"):
    if len(source) <= block_size + 1:
        raise ValueError("Source is too short for the requested block size.")
    ix = torch.randint(len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
