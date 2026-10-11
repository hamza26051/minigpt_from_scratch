import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, generateText, type ApiErrorKind } from '../services/api'
import type { GenerationRequest, GenerationResponse } from '../types/api'

export type GenerationStatus = 'idle' | 'loading' | 'success' | 'error'

export type GenerationError = {
  kind: ApiErrorKind | 'unknown'
  message: string
}

export type GenerationResult = {
  id: number
  request: GenerationRequest
  response: GenerationResponse
  durationMs: number
  completedAt: number
}

let nextResultId = 1

export function useGeneration() {
  const [status, setStatus] = useState<GenerationStatus>('idle')
  const [error, setError] = useState<GenerationError | null>(null)
  const controllerRef = useRef<AbortController | null>(null)

  /** Runs one generation. Resolves with the result, or null if it failed or was cancelled. */
  const run = useCallback(async (request: GenerationRequest): Promise<GenerationResult | null> => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller

    setStatus('loading')
    setError(null)

    const startedAt = performance.now()

    try {
      const response = await generateText(request, controller.signal)
      if (controller.signal.aborted) return null
      setStatus('success')
      return {
        id: nextResultId++,
        request,
        response,
        durationMs: performance.now() - startedAt,
        completedAt: Date.now(),
      }
    } catch (err) {
      if (controller.signal.aborted) return null
      setError({
        kind: err instanceof ApiError ? err.kind : 'unknown',
        message: err instanceof Error ? err.message : 'An unexpected error occurred.',
      })
      setStatus('error')
      return null
    }
  }, [])

  /** Cancels a running request (if any) and returns to the idle state. */
  const clear = useCallback(() => {
    controllerRef.current?.abort()
    controllerRef.current = null
    setStatus('idle')
    setError(null)
  }, [])

  useEffect(() => {
    return () => controllerRef.current?.abort()
  }, [])

  return { status, error, run, clear }
}