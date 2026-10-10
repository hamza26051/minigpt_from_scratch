import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "checkpoints" / "minigpt.onnx"
META_PATH = ROOT / "checkpoints" / "model_meta.json"


with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)


chars = list(metadata["chars"])
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

context_length = int(
    metadata["config"]["context_length"]
)

parameter_count = metadata.get("parameter_count")


session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)


def encode(text):
    missing = sorted(
        set(text) - set(stoi)
    )

    if missing:
        raise ValueError(
            f"Characters not present in vocabulary: {missing}"
        )

    return [stoi[ch] for ch in text]


def decode(ids):
    return "".join(
        itos[int(i)] for i in ids
    )


def softmax(x):
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


def generate_text(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
    seed=None
):
    if not prompt:
        raise ValueError("Prompt must not be empty.")

    if temperature <= 0:
        raise ValueError(
            "Temperature must be greater than zero."
        )

    ids = encode(prompt)

    rng = np.random.default_rng(seed)

    for _ in range(max_new_tokens):
        input_ids = np.asarray(
            [ids[-context_length:]],
            dtype=np.int64
        )

        logits = session.run(
            ["logits"],
            {
                "input_ids": input_ids
            }
        )[0]

        next_logits = (
            logits[0, -1, :].astype(np.float64)
            / max(temperature, 1e-5)
        )

        if top_k is not None:
            k = max(
                1,
                min(
                    int(top_k),
                    next_logits.shape[0]
                )
            )

            top_indices = np.argpartition(
                next_logits,
                -k
            )[-k:]

            filtered = np.full_like(
                next_logits,
                -np.inf
            )

            filtered[top_indices] = (
                next_logits[top_indices]
            )

            next_logits = filtered

        probabilities = softmax(
            next_logits
        )

        next_id = int(
            rng.choice(
                len(probabilities),
                p=probabilities
            )
        )

        ids.append(next_id)

    return decode(ids)


def runtime_info():
    return {
        "runtime": "ONNX Runtime",
        "provider": session.get_providers()[0],
        "parameters": parameter_count,
        "context_length": context_length
    }
