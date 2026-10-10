import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.generate import load_checkpoint


ROOT = Path(__file__).resolve().parent
PT_PATH = ROOT / "checkpoints" / "minigpt.pt"
ONNX_PATH = ROOT / "checkpoints" / "minigpt.onnx"
META_PATH = ROOT / "checkpoints" / "model_meta.json"


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        logits, _ = self.model(input_ids)
        return logits


print("Loading PyTorch checkpoint...")

model, tokenizer, checkpoint, device = load_checkpoint(
    PT_PATH,
    device="cpu"
)

model.eval()

wrapper = LogitsOnly(model)
wrapper.eval()

context_length = int(checkpoint["config"]["context_length"])
dummy_length = min(16, context_length)

dummy_input = torch.arange(
    dummy_length,
    dtype=torch.long
).unsqueeze(0) % checkpoint["vocab_size"]

print("Exporting ONNX model...")

torch.onnx.export(
    wrapper,
    dummy_input,
    str(ONNX_PATH),
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {
            0: "batch",
            1: "sequence"
        },
        "logits": {
            0: "batch",
            1: "sequence"
        }
    },
    opset_version=17,
    do_constant_folding=True,
    dynamo=False
)

metadata = {
    "chars": tokenizer.chars,
    "config": checkpoint["config"],
    "vocab_size": checkpoint["vocab_size"],
    "parameter_count": checkpoint.get(
        "parameter_count",
        sum(p.numel() for p in model.parameters())
    )
}

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("Checking ONNX model...")

onnx_model = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(onnx_model)

print("Comparing PyTorch and ONNX outputs...")

with torch.no_grad():
    torch_logits = wrapper(dummy_input).cpu().numpy()

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"]
)

onnx_logits = session.run(
    ["logits"],
    {
        "input_ids": dummy_input.numpy().astype(np.int64)
    }
)[0]

max_difference = float(
    np.max(np.abs(torch_logits - onnx_logits))
)

print()
print("ONNX export successful")
print("ONNX file:", ONNX_PATH)
print("Metadata file:", META_PATH)
print("Maximum output difference:", max_difference)

if max_difference > 1e-4:
    raise RuntimeError(
        f"ONNX verification failed. Difference = {max_difference}"
    )

print("PyTorch and ONNX outputs match.")
