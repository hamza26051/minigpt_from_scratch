class CharacterTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        missing = sorted(set(text) - set(self.stoi))
        if missing:
            raise ValueError(f"Characters not present in vocabulary: {missing}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)
