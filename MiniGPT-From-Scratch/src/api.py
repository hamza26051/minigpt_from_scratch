from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .generate import load_checkpoint, generate_text

CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "checkpoints"
    / "minigpt.pt"
)

model, tokenizer, checkpoint, device = load_checkpoint(CHECKPOINT)

app = FastAPI(
    title="MiniGPT From Scratch",
    version="1.0.0"
)


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=100, ge=1, le=500)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: int = Field(default=40, ge=1)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "parameters": checkpoint.get("parameter_count")
    }


@app.post("/generate")
def generate(request: GenerationRequest):
    try:
        text = generate_text(
            model,
            tokenizer,
            request.prompt,
            device,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k
        )
        return {
            "prompt": request.prompt,
            "generated_text": text
        }
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
