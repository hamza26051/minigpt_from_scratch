# MiniGPT From Scratch

A small GPT-style causal language model built from first principles in PyTorch for the AI Engineering Transformer assignment.

The core Transformer architecture is implemented directly rather than loaded from a pretrained language-model checkpoint. The project uses character-level tokenization and Tiny Shakespeare to demonstrate causal self-attention, multi-head attention, sinusoidal positional encoding, residual Transformer blocks, next-token training, text generation, checkpointing, unit tests, and controlled ablations.

## Final training run

The submitted executed notebook produced:

- **610,625 trainable parameters**
- **128-token context length**
- **128-dimensional model state**
- **4 attention heads**
- **3 Transformer blocks**
- **1,200 training steps**
- **AdamW**, learning rate `3e-4`
- **Final training loss:** `1.8020`
- **Final validation loss:** `1.9275`
- **Validation perplexity:** `6.8726`
- **Recorded GPU training time:** `0.62 minutes`

The notebook's final automated acceptance section reports that the tiny-overfit gate, five fixed generations, checkpoint reload, causal invariance check, all three ablation families, throughput/memory measurements, and learning-rate logging passed.

## Repository structure

```text
MiniGPT-From-Scratch/
├── README.md
├── requirements.txt
├── MiniGPT_FINAL_EXECUTED.ipynb
├── src/
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── attention.py
│   ├── positional_encoding.py
│   ├── transformer_block.py
│   ├── gpt.py
│   ├── generate.py
│   └── api.py
├── tests/
│   ├── test_attention.py
│   ├── test_mask.py
│   ├── test_shapes.py
│   └── test_overfit.py
├── checkpoints/
│   └── minigpt.pt
├── outputs/
│   ├── loss_curve.png
│   ├── positional_encoding.png
│   ├── attention_heatmap.png
│   ├── ablation_results.csv
│   └── five_fixed_generations.txt
└── experiments/
    └── ablation.py
```

## Setup

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

The training notebook downloads Tiny Shakespeare automatically. The modular dataset helper does the same if you reproduce the experiments.

## Generate text from the trained checkpoint

The supplied checkpoint is the checkpoint created by the executed notebook.

```bash
python -m src.generate \
  --prompt "ROMEO:" \
  --max-new-tokens 100 \
  --temperature 0.8 \
  --top-k 40
```

The generation command loads `checkpoints/minigpt.pt`; it does **not** retrain the model.

## Run tests

```bash
python -m pytest -q
```

The tests cover attention tensor shapes, causal masking, Transformer/model shapes, finite loss, and the tiny-batch overfit gate.

## Reproduce the ablations

```bash
python -m experiments.ablation
```

The submitted `outputs/ablation_results.csv` contains the measurements produced by the executed Colab. The experiment script reproduces the same three controlled families:

- depth: 2 vs 4 vs 6 blocks
- heads: 2 vs 4 vs 8 heads
- context length: 64 vs 128 vs 256 tokens

## FastAPI inference

Run:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Health check:

```text
GET /health
```

Generation endpoint:

```text
POST /generate
```

Example JSON body:

```json
{
  "prompt": "ROMEO:",
  "max_new_tokens": 100,
  "temperature": 0.8,
  "top_k": 40
}
```

## Model design

The submitted model contains:

1. character-level tokenization
2. token embeddings
3. sinusoidal positional encoding
4. causal scaled dot-product self-attention
5. configurable multi-head attention
6. pre-normalized residual Transformer blocks
7. GELU feed-forward networks
8. a vocabulary projection head
9. shifted next-token cross-entropy training
10. autoregressive temperature/top-k sampling

No pretrained GPT, BERT, Llama, Mistral, or equivalent language-model checkpoint is used by the core implementation.

## Evidence and outputs

`outputs/` preserves the results from the completed notebook:

- `loss_curve.png` — training and validation loss
- `positional_encoding.png` — sinusoidal position representation
- `attention_heatmap.png` — attention from the trained model
- `ablation_results.csv` — controlled depth/head/context experiments
- `five_fixed_generations.txt` — fixed-prompt generations

The full executed notebook is retained as `MiniGPT_FINAL_EXECUTED.ipynb`.

## Known limitations

This is intentionally a compact educational language model. It uses a small character-level corpus and a small architecture, so the generated text can imitate Shakespeare-like structure while remaining grammatically inconsistent or semantically weak. It is not intended to be a factual or general-purpose language model.

CPU and GPU timings depend on hardware, so reproduced throughput and memory numbers can differ from the submitted Colab measurements.
