import type { ConnectionStatus } from '../../hooks/useHealth'
import type { HealthResponse } from '../../types/api'

type StatusPillProps = {
  status: ConnectionStatus
  health: HealthResponse | null
  error: string | null
  onRecheck: () => void
}

const LABELS: Record<ConnectionStatus, string> = {
  checking: 'Checking API…',
  connected: 'API connected',
  unreachable: 'API unreachable',
}

const SHORT_LABELS: Record<ConnectionStatus, string> = {
  checking: 'API…',
  connected: 'API',
  unreachable: 'API offline',
}

const DOT_STYLES: Record<ConnectionStatus, string> = {
  checking: 'bg-muted animate-pulse',
  connected: 'bg-emerald-400',
  unreachable: 'bg-rose-400',
}

export function StatusPill({ status, health, error, onRecheck }: StatusPillProps) {
  const label = LABELS[status]

  let title = 'Checking the backend health endpoint.'
  if (status === 'connected' && health) {
    const details = [
      health.runtime ?? 'runtime unknown',
      health.provider ?? 'provider unknown',
      health.parameters !== null ? `${health.parameters.toLocaleString()} parameters` : 'parameter count unavailable',
      health.context_length !== null ? `context length ${health.context_length}` : null,
    ].filter((part): part is string => part !== null)
    title = `${details.join(', ')}. Click to check again.`
  } else if (status === 'unreachable') {
    title = `${error ?? 'The health check failed.'} Click to check again.`
  }

  return (
    <button
      type="button"
      onClick={onRecheck}
      title={title}
      aria-label={`${label}. Activate to check the connection again.`}
      className="inline-flex shrink-0 items-center gap-2 rounded-md px-2 py-1 text-xs text-muted transition-colors hover:bg-raised hover:text-ink"
    >
      <span aria-hidden="true">{SHORT_LABELS[status]}</span>
      <span className={`h-2 w-2 rounded-full ${DOT_STYLES[status]}`} aria-hidden="true" />
    </button>
  )
}