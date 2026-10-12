import type { ReactNode } from 'react'
import type { ConnectionStatus } from '../../hooks/useHealth'
import { PANE_TITLE } from '../../lib/ui'
import type { HealthResponse } from '../../types/api'

type ModelInfoProps = {
  status: ConnectionStatus
  health: HealthResponse | null
}

type RowProps = {
  label: string
  value: string
  note?: string
}

const UNAVAILABLE = '—'

function Row({ label, value, note }: RowProps) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-2">
      <dt className="shrink-0 text-[13px] text-muted">{label}</dt>
      <dd className="min-w-0 text-right text-[13px] tabular-nums">
        {value}
        {note && <span className="block whitespace-nowrap text-[11px] text-faint">{note}</span>}
      </dd>
    </div>
  )
}

function Group({ title, caption, children }: { title: ReactNode; caption: string; children: ReactNode }) {
  return (
    <section className="px-4 py-4">
      <h4 className="flex items-center gap-2 text-[13px] font-medium">{title}</h4>
      <p className="mt-0.5 text-xs text-faint">{caption}</p>
      <dl className="mt-2 divide-y divide-line/40">{children}</dl>
    </section>
  )
}

/**
 * Model facts, split by where they come from.
 * - Live: read from GET /health, shown only after the API responded.
 * - Training configuration: verified from the training checkpoint config (n_layers, n_heads, d_model,
 *   vocab_size, chars) and the source files (feed-forward width 4 * d_model, sinusoidal positional
 *   encoding), plus the Tiny Shakespeare dataset used in the notebook.
 */
export function ModelInfo({ status, health }: ModelInfoProps) {
  const live = status === 'connected' && health !== null
  const parameters = health?.parameters ?? null
  const contextLength = health?.context_length ?? null

  return (
    <div>
      <div className="px-4 pt-4">
        <h3 className={PANE_TITLE}>Model</h3>
        <p className="mt-1 text-xs text-faint">Trained from scratch, character-level</p>
      </div>

      <Group
        title={
          <>
            <span
              aria-hidden="true"
              className={`h-1.5 w-1.5 rounded-full ${live ? 'bg-emerald-400' : 'bg-faint'}`}
            />
            Live from the API
          </>
        }
        caption={live ? 'Reported by the health check.' : 'Available once the API responds.'}
      >
        <Row label="Parameters" value={parameters !== null ? parameters.toLocaleString() : UNAVAILABLE} />
        <Row
          label="Context length"
          value={contextLength !== null ? String(contextLength) : UNAVAILABLE}
          note="tokens (characters)"
        />
        <Row label="Runtime" value={health?.runtime ?? UNAVAILABLE} />
        <Row label="Provider" value={health?.provider ?? UNAVAILABLE} />
      </Group>

      <div className="mx-4 border-t border-line/40" />

      <Group title="Training configuration" caption="From the training checkpoint and source code.">
        <Row label="Tokenizer" value="Character-level" />
        <Row label="Vocabulary" value="65 characters" />
        <Row label="Transformer blocks" value="3" />
        <Row label="Attention heads" value="4" />
        <Row label="Model dimension" value="128" />
        <Row label="Feed-forward width" value="512" note="4 × model dimension" />
        <Row label="Positional encoding" value="Sinusoidal" />
        <Row label="Training data" value="Tiny Shakespeare" />
      </Group>
    </div>
  )
}