/**
 * Response of GET /health on the deployed backend, for example:
 * { "status": "ok", "runtime": "ONNX Runtime", "provider": "CPUExecutionProvider",
 *   "parameters": 610625, "context_length": 128 }
 * Only `status` is required; the other fields are null if the backend omits them.
 */
export interface HealthResponse {
  status: string
  runtime: string | null
  provider: string | null
  parameters: number | null
  context_length: number | null
}

/** Body of POST /generate. */
export interface GenerationRequest {
  prompt: string
  max_new_tokens: number
  temperature: number
  top_k: number
}

/** Response of POST /generate. `generated_text` starts with the prompt followed by the continuation. */
export interface GenerationResponse {
  prompt: string
  generated_text: string
}