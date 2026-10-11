import { Braces, ChevronRight, History, Trash2 } from 'lucide-react'
import type { GenerationResult } from '../../hooks/useGeneration'
import { countNewCharacters, formatClockTime, formatSeconds, summarizePrompt } from '../../lib/format'
import { PANE_TITLE } from '../../lib/ui'
import type { GenerationRequest } from '../../types/api'
import { PaneHeader } from '../layout/PaneHeader'
import { Button } from '../ui/Button'
import { CollapsibleSection } from '../ui/CollapsibleSection'
import { SectionEmpty } from '../ui/SectionEmpty'

type RunsPaneProps = {
  runs: GenerationResult[]
  selectedRunId: number | null
  onSelect: (id: number) => void
  onClearHistory: () => void
  disabled: boolean
  onCollapse: () => void
  maxRuns: number
}

/** Pretty-printed request body. The text content is exactly JSON.stringify(request, null, 2). */
function RequestJson({ request }: { request: GenerationRequest }) {
  const entries = Object.entries(request) as [string, string | number][]

  return (
    <>
      {'{\n'}
      {entries.map(([key, value], index) => (
        <span key={key}>
          {'  '}
          <span className="text-faint">{`"${key}"`}</span>
          {': '}
          <span className={typeof value === 'number' ? 'text-accent' : 'text-ink'}>{JSON.stringify(value)}</span>
          {index < entries.length - 1 ? ',' : ''}
          {'\n'}
        </span>
      ))}
      {'}'}
    </>
  )
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1.5">
      <dt className="text-xs text-faint">{label}</dt>
      <dd className="font-mono text-xs tabular-nums text-muted">{value}</dd>
    </div>
  )
}

export function RunsPane({
  runs,
  selectedRunId,
  onSelect,
  onClearHistory,
  disabled,
  onCollapse,
  maxRuns,
}: RunsPaneProps) {
  const selectedRun = runs.find((run) => run.id === selectedRunId) ?? null
  const selectedNewCharacters = selectedRun
    ? countNewCharacters(selectedRun.request.prompt, selectedRun.response.generated_text)
    : null

  return (
    <>
      <PaneHeader>
        <Button variant="ghost" size="icon" onClick={onCollapse} aria-label="Hide session panel" title="Hide panel">
          <ChevronRight className="h-4 w-4" aria-hidden="true" />
        </Button>
        <h2 className={PANE_TITLE}>Session</h2>
      </PaneHeader>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <CollapsibleSection
          title="Runs"
          count={runs.length}
          icon={<History className="h-4 w-4" />}
          actions={
            runs.length > 0 ? (
              <Button
                variant="ghost"
                size="icon"
                onClick={onClearHistory}
                aria-label="Clear run history"
                title="Clear run history"
              >
                <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
              </Button>
            ) : undefined
          }
        >
          {runs.length === 0 ? (
            <SectionEmpty
              icon={<History className="h-4 w-4" />}
              title="No runs yet"
              description="Generated texts from this session will appear here."
            />
          ) : (
            <>
              <ul className="-mx-1 space-y-0.5">
                {runs.map((run) => {
                  const active = run.id === selectedRunId
                  const newCharacters = countNewCharacters(run.request.prompt, run.response.generated_text)
                  return (
                    <li key={run.id}>
                      <button
                        type="button"
                        disabled={disabled}
                        aria-current={active ? 'true' : undefined}
                        onClick={() => onSelect(run.id)}
                        className={`relative w-full rounded-lg px-3 py-2 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
                          active ? 'bg-raised' : 'hover:bg-raised/50'
                        }`}
                      >
                        {active && (
                          <span
                            aria-hidden="true"
                            className="absolute inset-y-2 left-0 w-0.5 rounded-full bg-accent"
                          />
                        )}
                        <span className="block truncate font-mono text-xs text-ink">
                          {summarizePrompt(run.request.prompt)}
                        </span>
                        <span className="mt-1 flex items-center justify-between gap-2 text-[11px] text-faint">
                          <span className="truncate">
                            {newCharacters !== null ? `${newCharacters} characters in ` : ''}
                            {formatSeconds(run.durationMs, 1)}
                          </span>
                          <span className="shrink-0 font-mono tabular-nums">{formatClockTime(run.completedAt)}</span>
                        </span>
                      </button>
                    </li>
                  )
                })}
              </ul>
              <p className="mt-3 text-[11px] leading-relaxed text-faint">
                Kept in memory until you reload the page. The latest {maxRuns} runs are stored.
              </p>
            </>
          )}
        </CollapsibleSection>

        <CollapsibleSection title="Request and response" icon={<Braces className="h-4 w-4" />}>
          {selectedRun === null ? (
            <SectionEmpty
              icon={<Braces className="h-4 w-4" />}
              title="No run selected"
              description="Select a run to inspect its request."
            />
          ) : (
            <div>
              <p className="text-xs text-faint">
                Body sent to <span className="font-mono text-muted">POST /generate</span>
              </p>
              <pre
                tabIndex={0}
                aria-label="Request body"
                className="mt-2 max-h-60 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-panel px-3 py-2.5 font-mono text-xs leading-relaxed"
              >
                <RequestJson request={selectedRun.request} />
              </pre>
              <dl className="mt-3 divide-y divide-line/40">
                {selectedNewCharacters !== null && (
                  <DetailRow label="New characters" value={String(selectedNewCharacters)} />
                )}
                <DetailRow label="Round trip" value={formatSeconds(selectedRun.durationMs)} />
                <DetailRow label="Completed" value={formatClockTime(selectedRun.completedAt)} />
              </dl>
            </div>
          )}
        </CollapsibleSection>
      </div>
    </>
  )
}