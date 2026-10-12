import { API_BASE_URL } from '../lib/config'
import type { GenerationRequest, GenerationResponse, HealthResponse } from '../types/api'

export type ApiErrorKind = 'network' | 'timeout' | 'http' | 'validation' | 'malformed' | 'aborted'

export class ApiError extends Error {
  kind: ApiErrorKind
  status?: number

  constructor(kind: ApiErrorKind, message: string, status?: number) {
    super(message)
    this.name = 'ApiError'
    this.kind = kind
    this.status = status
  }
}

// A serverless backend can take several seconds to wake up (cold start).
const HEALTH_TIMEOUT_MS = 20_000
const GENERATE_TIMEOUT_MS = 180_000

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** FastAPI returns `detail` as a string (HTTPException) or an array (422 validation). */
function formatDetail(detail: unknown): string | null {
  if (typeof detail === 'string') return detail

  if (Array.isArray(detail)) {
    const parts = detail
      .map((item): string | null => {
        if (!isRecord(item) || typeof item.msg !== 'string') return null
        const location = Array.isArray(item.loc)
          ? item.loc
              .filter((p): p is string | number => typeof p === 'string' || typeof p === 'number')
              .filter((p) => p !== 'body')
              .join('.')
          : ''
        return location ? `${location}: ${item.msg}` : item.msg
      })
      .filter((part): part is string => part !== null)
    return parts.length > 0 ? parts.join('; ') : null
  }

  return null
}

function buildHttpError(status: number, raw: string, platformCode: string | null): ApiError {
  let detail: string | null = null
  try {
    const parsed: unknown = JSON.parse(raw)
    if (isRecord(parsed)) detail = formatDetail(parsed.detail)
  } catch {
    // The body is not JSON (for example a platform error page); use the status-based messages below.
  }

  const suffix = platformCode ? ` [${platformCode}]` : ''

  if (status === 404) {
    return new ApiError(
      'http',
      `${detail ?? 'The endpoint was not found'} (HTTP 404). Check VITE_API_BASE_URL and the proxy or rewrite configuration.`,
      status,
    )
  }

  if (detail) return new ApiError(status === 422 ? 'validation' : 'http', detail, status)

  switch (status) {
    case 408:
    case 504:
      return new ApiError(
        'timeout',
        `The backend took too long to respond (HTTP ${status})${suffix}. Serverless deployments have an execution time limit, so try fewer new tokens.`,
        status,
      )
    case 429:
      return new ApiError('http', `The backend is rate limiting requests (HTTP 429)${suffix}. Wait a moment and try again.`, status)
    case 502:
    case 503:
      return new ApiError(
        'http',
        `The backend is temporarily unavailable (HTTP ${status})${suffix}. It may be starting up; try again in a few seconds.`,
        status,
      )
    default:
      return new ApiError(
        'http',
        status >= 500
          ? `The backend returned HTTP ${status} without details${suffix}.`
          : `The backend returned HTTP ${status}${suffix}.`,
        status,
      )
  }
}

type RequestOptions = {
  method?: 'GET' | 'POST'
  body?: unknown
  timeoutMs: number
  signal?: AbortSignal
}

async function requestJson(path: string, options: RequestOptions): Promise<unknown> {
  const controller = new AbortController()
  let timedOut = false

  const timer = window.setTimeout(() => {
    timedOut = true
    controller.abort()
  }, options.timeoutMs)

  const forwardAbort = () => controller.abort()
  options.signal?.addEventListener('abort', forwardAbort, { once: true })
  if (options.signal?.aborted) controller.abort()

  try {
    const hasBody = options.body !== undefined
    const response = await fetch(`${API_BASE_URL}${path}`, {
      method: options.method ?? 'GET',
      headers: hasBody
        ? { 'Content-Type': 'application/json', Accept: 'application/json' }
        : { Accept: 'application/json' },
      body: hasBody ? JSON.stringify(options.body) : undefined,
      signal: controller.signal,
    })

    const raw = await response.text()

    if (!response.ok) {
      throw buildHttpError(response.status, raw, response.headers.get('x-vercel-error'))
    }

    try {
      return JSON.parse(raw) as unknown
    } catch {
      throw new ApiError('malformed', 'The backend returned a response that is not valid JSON.', response.status)
    }
  } catch (err) {
    if (err instanceof ApiError) throw err

    if (timedOut) {
      throw new ApiError('timeout', `The request timed out after ${Math.round(options.timeoutMs / 1000)} seconds.`)
    }
    if (options.signal?.aborted) {
      throw new ApiError('aborted', 'The request was cancelled.')
    }
    throw new ApiError(
      'network',
      `Could not reach the backend through "${API_BASE_URL}". Check your internet connection and that the frontend dev server is running.`,
    )
  } finally {
    window.clearTimeout(timer)
    options.signal?.removeEventListener('abort', forwardAbort)
  }
}

function malformed(what: string): ApiError {
  return new ApiError('malformed', `The backend response did not match the expected ${what} format.`)
}

function optionalString(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function optionalNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function parseHealth(data: unknown): HealthResponse {
  if (!isRecord(data) || typeof data.status !== 'string') {
    throw malformed('/health')
  }

  return {
    status: data.status,
    runtime: optionalString(data.runtime),
    provider: optionalString(data.provider),
    parameters: optionalNumber(data.parameters),
    context_length: optionalNumber(data.context_length),
  }
}

function parseGeneration(data: unknown): GenerationResponse {
  if (!isRecord(data) || typeof data.generated_text !== 'string' || typeof data.prompt !== 'string') {
    throw malformed('/generate')
  }
  return { prompt: data.prompt, generated_text: data.generated_text }
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const data = await requestJson('/health', { timeoutMs: HEALTH_TIMEOUT_MS, signal })
  return parseHealth(data)
}

export async function generateText(
  request: GenerationRequest,
  signal?: AbortSignal,
): Promise<GenerationResponse> {
  const data = await requestJson('/generate', {
    method: 'POST',
    body: request,
    timeoutMs: GENERATE_TIMEOUT_MS,
    signal,
  })
  return parseGeneration(data)
}