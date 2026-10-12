import { RefreshCw, ServerOff } from 'lucide-react'
import { Button } from '../ui/Button'

type BackendNoticeProps = {
  error: string | null
  onRetry: () => void
}

export function BackendNotice({ error, onRetry }: BackendNoticeProps) {
  return (
    <div
      role="status"
      className="flex flex-col gap-3 rounded-xl border border-line bg-panel p-4 text-sm sm:flex-row sm:items-start"
    >
      <ServerOff className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" aria-hidden="true" />
      <div className="min-w-0 flex-1 space-y-1">
        <p className="font-medium">The MiniGPT API did not respond.</p>
        {error && <p className="break-words text-muted">{error}</p>}
        <p className="text-muted">
          A serverless backend can take a few seconds to wake up. The connection is also re-checked
          automatically every minute.
        </p>
      </div>
      <Button size="sm" onClick={onRetry} className="shrink-0 self-start">
        <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
        Check again
      </Button>
    </div>
  )
}