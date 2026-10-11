import { useMemo, useRef, useState } from 'react'
import { Cpu, PanelLeftOpen, PanelRightOpen } from 'lucide-react'
import { useGeneration, type GenerationResult } from '../../hooks/useGeneration'
import type { ConnectionStatus } from '../../hooks/useHealth'
import { findUnsupportedCharacters, GENERATION_DEFAULTS, type GenerationParams } from '../../lib/limits'
import { CONTENT_COLUMN, PANE_TITLE } from '../../lib/ui'
import type { GenerationRequest, HealthResponse } from '../../types/api'
import { AppShell } from '../layout/AppShell'
import { BackendNotice } from '../layout/BackendNotice'
import { PaneHeader } from '../layout/PaneHeader'
import { StatusPill } from '../layout/StatusPill'
import { Button } from '../ui/Button'
import { Composer } from './Composer'
import { ControlsPane } from './ControlsPane'
import { OutputPanel } from './OutputPanel'
import { RunsPane } from './RunsPane'

const MAX_RUNS = 20

type PlaygroundProps = {
  connectionStatus: ConnectionStatus
  health: HealthResponse | null
  connectionError: string | null
  onRecheck: () => void
}

export function Playground({ connectionStatus, health, connectionError, onRecheck }: PlaygroundProps) {
  const [prompt, setPrompt] = useState('')
  const [params, setParams] = useState<GenerationParams>({ ...GENERATION_DEFAULTS })
  const [runs, setRuns] = useState<GenerationResult[]>([])
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null)
  const { status, error, run: runGeneration, clear: clearGeneration } = useGeneration()
  const lastRequestRef = useRef<GenerationRequest | null>(null)

  const unsupportedCharacters = useMemo(() => findUnsupportedCharacters(prompt), [prompt])
  const loading = status === 'loading'
  const selectedRun = runs.find((run) => run.id === selectedRunId) ?? null

  async function execute(request: GenerationRequest) {
    if (loading) return
    lastRequestRef.current = request
    setSelectedRunId(null)
    const result = await runGeneration(request)
    if (result) {
      setRuns((previous) => [result, ...previous].slice(0, MAX_RUNS))
      setSelectedRunId(result.id)
    }
  }

  function submit() {
    if (prompt.length === 0 || loading) return
    void execute({
      prompt,
      max_new_tokens: params.maxNewTokens,
      temperature: params.temperature,
      top_k: params.topK,
    })
  }

  /** Re-sends the last request exactly as it was sent. */
  function retry() {
    if (lastRequestRef.current) void execute(lastRequestRef.current)
  }

  /** Clears the displayed output (and cancels a running request). History is kept. */
  function clearOutput() {
    clearGeneration()
    setSelectedRunId(null)
  }

  function selectRun(id: number) {
    if (loading) return
    if (status === 'error') clearGeneration()
    setSelectedRunId(id)
  }

  function clearHistory() {
    setRuns([])
    setSelectedRunId(null)
  }

  return (
    <AppShell
      left={(shell) => (
        <ControlsPane
          params={params}
          onParamsChange={setParams}
          disabled={loading}
          connectionStatus={connectionStatus}
          health={health}
          onCollapse={() => shell.setLeftOpen(false)}
        />
      )}
      center={(shell) => (
        <>
          <PaneHeader>
            {!shell.leftOpen && (
              <Button
                variant="ghost"
                size="icon"
                onClick={() => shell.setLeftOpen(true)}
                aria-label="Show controls panel"
                title="Show controls"
                className="-ml-1.5"
              >
                <PanelLeftOpen className="h-4 w-4" aria-hidden="true" />
              </Button>
            )}
            <h2 className={PANE_TITLE}>Playground</h2>
            <span className="inline-flex items-center gap-1.5 rounded-md bg-panel px-2 py-1 text-xs ring-1 ring-line/50">
              <Cpu className="h-3.5 w-3.5 text-faint" aria-hidden="true" />
              MiniGPT
              {health?.parameters != null && (
                <span className="hidden text-faint sm:inline">{health.parameters.toLocaleString()} parameters</span>
              )}
            </span>
            <div className="ml-auto flex items-center gap-1">
              <StatusPill status={connectionStatus} health={health} error={connectionError} onRecheck={onRecheck} />
              {!shell.rightOpen && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => shell.setRightOpen(true)}
                  aria-label="Show session panel"
                  title="Show session"
                  className="-mr-1.5"
                >
                  <PanelRightOpen className="h-4 w-4" aria-hidden="true" />
                </Button>
              )}
            </div>
          </PaneHeader>

          <div className="no-scrollbar flex h-10 shrink-0 items-center gap-6 overflow-x-auto border-b border-line/50 px-4 text-xs text-faint">
            <span className="shrink-0 whitespace-nowrap">
              Temperature <span className="ml-1 font-mono text-muted">{params.temperature}</span>
            </span>
            <span className="shrink-0 whitespace-nowrap">
              Top-k <span className="ml-1 font-mono text-muted">{params.topK}</span>
            </span>
            <span className="shrink-0 whitespace-nowrap">
              Max new tokens <span className="ml-1 font-mono text-muted">{params.maxNewTokens}</span>
            </span>
          </div>

          <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
            {connectionStatus === 'unreachable' && (
              <div className={`${CONTENT_COLUMN} pt-5`}>
                <BackendNotice error={connectionError} onRetry={onRecheck} />
              </div>
            )}
            <OutputPanel
              status={status}
              run={selectedRun}
              error={error}
              onClear={clearOutput}
              onRerun={(request) => void execute(request)}
              onRetry={retry}
            />
          </div>

          <Composer
            prompt={prompt}
            onPromptChange={setPrompt}
            onSubmit={submit}
            loading={loading}
            unsupportedCharacters={unsupportedCharacters}
          />
        </>
      )}
      right={(shell) => (
        <RunsPane
          runs={runs}
          selectedRunId={selectedRunId}
          onSelect={selectRun}
          onClearHistory={clearHistory}
          disabled={loading}
          onCollapse={() => shell.setRightOpen(false)}
          maxRuns={MAX_RUNS}
        />
      )}
    />
  )
}