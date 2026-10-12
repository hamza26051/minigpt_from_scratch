/**
 * Defaults, bounds and vocabulary used by the Playground.
 * - Defaults (100 / 0.8 / 40): taken from the original api.py GenerationRequest.
 * - Bounds: from the original api.py (max_new_tokens 1-500, temperature >0 and <=2, top_k >=1).
 *   These have NOT yet been confirmed against the deployed backend's /docs schema.
 * - Context length 128 and the 65-character vocabulary: verified from the checkpoint.
 */

export type GenerationParams = {
  maxNewTokens: number
  temperature: number
  topK: number
}

export const GENERATION_DEFAULTS: GenerationParams = {
  maxNewTokens: 100,
  temperature: 0.8,
  topK: 40,
}

export const CONTEXT_LENGTH = 128

export const MODEL_VOCABULARY =
  "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

const VOCABULARY_SET = new Set(MODEL_VOCABULARY)

export const VOCAB_SIZE = VOCABULARY_SET.size

export const GENERATION_LIMITS = {
  maxNewTokens: { min: 1, max: 500, step: 1 },
  // The 0.1 lower bound is a UI choice; the backend accepts any value above 0.
  temperature: { min: 0.1, max: 2, step: 0.05 },
  // The model clamps top_k to the vocabulary size.
  topK: { min: 1, max: VOCAB_SIZE, step: 1 },
} as const

export const EXAMPLE_PROMPTS = ['ROMEO:', 'JULIET:', 'HAMLET:', 'MACBETH:', 'KING RICHARD:'] as const

/** Characters in `text` that are not in the model's character vocabulary. */
export function findUnsupportedCharacters(text: string): string[] {
  const found = new Set<string>()
  for (const ch of text) {
    if (!VOCABULARY_SET.has(ch)) found.add(ch)
  }
  return [...found]
}