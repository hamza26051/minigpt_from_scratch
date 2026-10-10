from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .onnx_generate import generate_text, runtime_info


app = FastAPI(
    title="MiniGPT From Scratch",
    version="1.0.0",
    description="A GPT-style causal language model implemented and trained from scratch, served using ONNX Runtime."
)


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(
        default=100,
        ge=1,
        le=500
    )
    temperature: float = Field(
        default=0.8,
        gt=0.0,
        le=2.0
    )
    top_k: int = Field(
        default=40,
        ge=1
    )


@app.get("/")
def root():
    return {
        "name": "MiniGPT From Scratch",
        "status": "online",
        "runtime": "ONNX Runtime",
        "docs": "/docs",
        "health": "/health",
        "generate": "/generate"
    }


@app.get("/health")
def health():
    info = runtime_info()

    return {
        "status": "ok",
        **info
    }


@app.post("/generate")
def generate(request: GenerationRequest):
    try:
        text = generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k
        )

        return {
            "prompt": request.prompt,
            "generated_text": text
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )
