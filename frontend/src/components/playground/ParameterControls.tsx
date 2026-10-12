import { useId, useState, type CSSProperties } from 'react'
import { RotateCcw } from 'lucide-react'
import { PANE_TITLE } from '../../lib/ui'
import { GENERATION_DEFAULTS, GENERATION_LIMITS, type GenerationParams } from '../../lib/limits'
import { Button } from '../ui/Button'

type ParameterControlsProps = {
  params: GenerationParams
  onChange: (next: GenerationParams) => void
  disabled: boolean
}

type ParamFieldProps = {
  label: string
  hint: string
  value: number
  min: number
  max: number
  step: number
  integer?: boolean
  disabled?: boolean
  onChange: (value: number) => void
}

function normalize(value: number, min: number, max: number, integer: boolean): number {
  const clamped = Math.min(max, Math.max(min, value))
  return integer ? Math.round(clamped) : Math.round(clamped * 100) / 100
}

function ParamField({
  label,
  hint,
  value,
  min,
  max,
  step,
  integer = false,
  disabled,
  onChange,
}: ParamFieldProps) {
  const id = useId()
  const hintId = useId()
  const [draft, setDraft] = useState<string | null>(null)
  const fill = ((value - min) / (max - min)) * 100

  function handleText(text: string) {
    setDraft(text)
    const parsed = Number(text)
    const valid =
      text.trim() !== '' &&
      Number.isFinite(parsed) &&
      parsed >= min &&
      parsed <= max &&
      (!integer || Number.isInteger(parsed))
    if (valid) onChange(parsed)
  }

  function commit() {
    if (draft === null) return
    const parsed = Number(draft)
    if (draft.trim() !== '' && Number.isFinite(parsed)) {
      onChange(normalize(parsed, min, max, integer))
    }
    setDraft(null)
  }

  return (
    <div className="py-4 first:pt-1 last:pb-0">
      <div className="flex h-8 items-center justify-between gap-3">
        <label htmlFor={id} className="text-[13px] font-medium">
          {label}
        </label>
        <input
          id={id}
          type="text"
          inputMode={integer ? 'numeric' : 'decimal'}
          value={draft ?? String(value)}
          disabled={disabled}
          aria-describedby={hintId}
          onChange={(e) => handleText(e.target.value)}
          onBlur={commit}
          className="h-8 w-[4.5rem] rounded-lg border border-line/70 bg-panel px-2.5 text-right font-mono text-[13px] tabular-nums transition-colors hover:border-line focus-visible:border-accent/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/20 disabled:opacity-50"
        />
      </div>
      <input
        type="range"
        className="range mt-1.5"
        style={{ '--fill': `${fill}%` } as CSSProperties}
        aria-label={`${label} slider`}
        aria-describedby={hintId}
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => {
          setDraft(null)
          onChange(normalize(Number(e.target.value), min, max, integer))
        }}
      />
      <p id={hintId} className="mt-1 text-xs text-faint">
        {hint}
      </p>
    </div>
  )
}

export function ParameterControls({ params, onChange, disabled }: ParameterControlsProps) {
  const isDefault =
    params.maxNewTokens === GENERATION_DEFAULTS.maxNewTokens &&
    params.temperature === GENERATION_DEFAULTS.temperature &&
    params.topK === GENERATION_DEFAULTS.topK

  return (
    <div className="px-4 py-4">
      <div className="flex h-8 items-center justify-between">
        <h3 className={PANE_TITLE}>Parameters</h3>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onChange({ ...GENERATION_DEFAULTS })}
          disabled={disabled || isDefault}
          className="-mr-2"
        >
          <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
          Reset
        </Button>
      </div>

      <div className="mt-2 divide-y divide-line/40">
        <ParamField
          label="Max new tokens"
          hint="Characters to generate, 1–500."
          value={params.maxNewTokens}
          {...GENERATION_LIMITS.maxNewTokens}
          integer
          disabled={disabled}
          onChange={(maxNewTokens) => onChange({ ...params, maxNewTokens })}
        />
        <ParamField
          label="Temperature"
          hint="Randomness, up to 2. Lower is safer."
          value={params.temperature}
          {...GENERATION_LIMITS.temperature}
          disabled={disabled}
          onChange={(temperature) => onChange({ ...params, temperature })}
        />
        <ParamField
          label="Top-k"
          hint="Sample only the k likeliest, 1–65."
          value={params.topK}
          {...GENERATION_LIMITS.topK}
          integer
          disabled={disabled}
          onChange={(topK) => onChange({ ...params, topK })}
        />
      </div>

      <p className="mt-5 text-xs leading-relaxed text-faint">
        Sent with every request. No seed is sent, so output varies between runs.
      </p>
    </div>
  )
}