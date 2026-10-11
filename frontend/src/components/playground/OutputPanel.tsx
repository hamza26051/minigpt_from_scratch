import { useEffect, useRef, useState } from 'react'
import { AlertTriangle, Check, Copy, Loader2, RefreshCw, Trash2, X } from 'lucide-react'
import type { GenerationError, GenerationResult, GenerationStatus } from '../../hooks/useGeneration'
import { countNewCharacters, formatSeconds } from '../../lib/format'
import { CONTENT_COLUMN } from '../../lib/ui'
import type { GenerationRequest } from '../../types/api'
import { Button } from '../ui/Button'

type OutputPanelProps = {
  status: GenerationStatus
  /** The run to display when nothing is loading and there is no error. */
  run: GenerationResult | null
  error: GenerationError | null
  onClear: () => void
  onRerun: (request: GenerationRequest) => void
  onRetry: () => void
}

const ERROR_TITLES: Record<GenerationError['kind'], string> = {
  network: 'Backend unreachable',
  timeout: 'Request timed out',
  http: 'Request failed',
  validation: 'Invalid request',
  malformed: 'Unexpected response',
  aborted: 'Request cancelled',
  unknown: 'Something went wrong',
}

export function OutputPanel({ status, run, error, onClear, onRerun, onRetry }: OutputPanelProps) {
  let content
  if (status === 'loading') {
    content = <LoadingState onCancel={onClear} />
  } else if (status === 'error' && error) {
    content = <ErrorState error={error} onRetry={onRetry} />
  } else if (run) {
    content = <GeneratedText key={run.id} run={run} onClear={onClear} onRerun={onRerun} />
  } else {
    content = <WelcomeState />
  }

  return (
    <div className={`${CONTENT_COLUMN} flex flex-1 flex-col py-8`}>
      <span className="sr-only" role="status" aria-live="polite">
        {status === 'loading' ? 'Generating text.' : status === 'success' && run ? 'Generation complete.' : ''}
      </span>
      {content}
    </div>
  )
}

const SAMPLE_PROMPT = 'ROMEO:'

/**
 * Shows what "character-level" means: every character of a prompt is one token, and the model's job
 * is to predict the next one. The tiles illustrate how the prompt is read; they are not model output.
 */
function WelcomeState() {
  const characters = Array.from(SAMPLE_PROMPT)

  return (
    <div className="flex flex-1 flex-col items-center justify-center pb-6 text-center">
      <div
        role="img"
        aria-label={`The prompt ${SAMPLE_PROMPT} is read as ${characters.length} tokens, one per character. The model predicts the next one.`}
        className="flex items-center gap-1.5 font-mono"
      >
        {characters.map((character, index) => (
          <span
            key={index}
            aria-hidden="true"
            className="flex h-11 w-9 items-center justify-center rounded-lg border border-line/70 bg-panel text-[15px]"
          >
            {character}
          </span>
        ))}
        <span
          aria-hidden="true"
          className="flex h-11 w-9 items-center justify-center rounded-lg border border-dashed border-accent/60 text-[15px] text-accent"
        >
          ?
        </span>
      </div>
      <p className="mt-3 text-xs text-faint">
        One character, one token. The model predicts the next one.
      </p>

      <h3 className="mt-9 text-3xl font-semibold tracking-tight">Explore next-token generation.</h3>
      <p className="mt-3 max-w-md text-[15px] leading-relaxed text-muted">
        MiniGPT is a small Transformer trained from scratch on Tiny Shakespeare. Write a prompt and it
        continues the text one character at a time.
      </p>
      <p className="mt-2 max-w-md text-sm text-faint">
        Expect Shakespeare-like structure rather than coherent meaning.
      </p>
    </div>
  )
}

/** Counts up while mounted, so it always measures the current request. Hidden from screen readers. */
function ElapsedTime() {
  const [elapsedMs, setElapsedMs] = useState(0)

  useEffect(() => {
    const start = performance.now()
    const id = window.setInterval(() => setElapsedMs(performance.now() - start), 100)
    return () => window.clearInterval(id)
  }, [])

  return (
    <span aria-hidden="true" className="font-mono tabular-nums text-ink">
      {(elapsedMs / 1000).toFixed(1)} s
    </span>
  )
}

function LoadingState({ onCancel }: { onCancel: () => void }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center pb-6 text-center">
      <Loader2 className="h-5 w-5 animate-spin text-accent" aria-hidden="true" />
      <p className="mt-3 text-sm font-medium">Generating on the server</p>
      <p className="mt-1 text-sm text-faint">
        Elapsed <ElapsedTime />
      </p>
      <p className="mt-1 max-w-xs text-xs text-faint">
        The first request after a quiet period can take longer than later ones.
      </p>
      <Button size="sm" onClick={onCancel} className="mt-4">
        <X className="h-3.5 w-3.5" aria-hidden="true" />
        Cancel
      </Button>
    </div>
  )
}

function ErrorState({ error, onRetry }: { error: GenerationError; onRetry: () => void }) {
  const retryable = error.kind !== 'validation'

  return (
    <div role="alert" className="rounded-xl border border-rose-400/25 bg-rose-400/[0.04] p-4">
      <div className="flex gap-3">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" aria-hidden="true" />
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium">{ERROR_TITLES[error.kind]}</p>
          <p className="mt-1 break-words text-sm text-muted">{error.message}</p>
          {!retryable && (
            <p className="mt-2 text-sm text-muted">Adjust the prompt or parameters, then generate again.</p>
          )}
        </div>
      </div>
      {retryable && (
        <div className="mt-3 flex justify-end">
          <Button size="sm" onClick={onRetry}>
            <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
            Try again
          </Button>
        </div>
      )}
    </div>
  )
}

type CopyState = 'idle' | 'copied' | 'failed'

type GeneratedTextProps = {
  run: GenerationResult
  onClear: () => void
  onRerun: (request: GenerationRequest) => void
}

function GeneratedText({ run, onClear, onRerun }: GeneratedTextProps) {
  const [copyState, setCopyState] = useState<CopyState>('idle')
  const timerRef = useRef<number | undefined>(undefined)

  useEffect(() => () => window.clearTimeout(timerRef.current), [])

  const { response, request, durationMs } = run
  const text = response.generated_text
  const prompt = response.prompt
  const newCharacters = countNewCharacters(prompt, text)
  const startsWithPrompt = newCharacters !== null

  async function handleCopy() {
    let next: CopyState = 'copied'
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      next = 'failed'
    }
    setCopyState(next)
    window.clearTimeout(timerRef.current)
    timerRef.current = window.setTimeout(() => setCopyState('idle'), 2000)
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        {startsWithPrompt ? (
          <p className="flex items-center gap-4 text-xs text-faint">
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-sm bg-accent" aria-hidden="true" />
              Prompt
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-sm bg-ink" aria-hidden="true" />
              Generated
            </span>
          </p>
        ) : (
          <span />
        )}
        <div className="flex items-center gap-1.5">
          <Button size="sm" onClick={handleCopy} className="min-w-24">
            {copyState === 'copied' ? (
              <Check className="h-3.5 w-3.5" aria-hidden="true" />
            ) : (
              <Copy className="h-3.5 w-3.5" aria-hidden="true" />
            )}
            {copyState === 'copied' ? 'Copied' : copyState === 'failed' ? 'Copy failed' : 'Copy'}
          </Button>
          <Button size="sm" onClick={() => onRerun(request)}>
            <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
            Regenerate
          </Button>
          <Button variant="ghost" size="sm" onClick={onClear}>
            <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
            Clear
          </Button>
        </div>
      </div>

      <div
        role="region"
        aria-label="Generated text"
        tabIndex={0}
        className="rounded-2xl border border-line/50 bg-panel px-5 py-4"
      >
        <pre className="whitespace-pre-wrap break-words font-mono text-[14.5px] leading-7 text-ink">
          {startsWithPrompt ? (
            <>
              <span className="text-accent">{prompt}</span>
              {text.slice(prompt.length)}
            </>
          ) : (
            text
          )}
        </pre>
      </div>

      <dl className="flex flex-wrap gap-x-6 gap-y-1 px-1 text-xs">
        {newCharacters !== null && <Stat label="New characters" value={String(newCharacters)} />}
        <Stat label="Round trip" value={formatSeconds(durationMs)} />
        <Stat label="Temperature" value={String(request.temperature)} />
        <Stat label="Top-k" value={String(request.top_k)} />
        <Stat label="Max new tokens" value={String(request.max_new_tokens)} />
      </dl>

      <span className="sr-only" role="status" aria-live="polite">
        {copyState === 'copied' ? 'Output copied to clipboard.' : copyState === 'failed' ? 'Copy failed.' : ''}
      </span>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <dt className="text-faint">{label}</dt>
      <dd className="font-mono tabular-nums text-muted">{value}</dd>
    </div>
  )
}