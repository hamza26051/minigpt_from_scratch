import { useId, useState, type KeyboardEvent } from 'react'
import { ChevronLeft, Cpu, SlidersHorizontal } from 'lucide-react'
import type { ConnectionStatus } from '../../hooks/useHealth'
import type { GenerationParams } from '../../lib/limits'
import type { HealthResponse } from '../../types/api'
import { Logo } from '../layout/Logo'
import { PaneHeader } from '../layout/PaneHeader'
import { Button } from '../ui/Button'
import { ModelInfo } from './ModelInfo'
import { ParameterControls } from './ParameterControls'

type ControlsPaneProps = {
  params: GenerationParams
  onParamsChange: (next: GenerationParams) => void
  disabled: boolean
  connectionStatus: ConnectionStatus
  health: HealthResponse | null
  onCollapse: () => void
}

const TABS = [
  { id: 'parameters', label: 'Parameters', icon: SlidersHorizontal },
  { id: 'model', label: 'Model', icon: Cpu },
] as const

type TabId = (typeof TABS)[number]['id']

export function ControlsPane({
  params,
  onParamsChange,
  disabled,
  connectionStatus,
  health,
  onCollapse,
}: ControlsPaneProps) {
  const [tab, setTab] = useState<TabId>('parameters')
  const baseId = useId()

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return
    event.preventDefault()
    const index = TABS.findIndex((t) => t.id === tab)
    const offset = event.key === 'ArrowRight' ? 1 : -1
    const next = TABS[(index + offset + TABS.length) % TABS.length]
    setTab(next.id)
    document.getElementById(`${baseId}-tab-${next.id}`)?.focus()
  }

  return (
    <>
      <PaneHeader>
        <Logo className="h-7 w-7 shrink-0" />
        <span className="min-w-0 flex-1 truncate text-sm font-semibold tracking-wide">MiniGPT</span>
        <Button variant="ghost" size="icon" onClick={onCollapse} aria-label="Hide controls panel" title="Hide panel">
          <ChevronLeft className="h-4 w-4" aria-hidden="true" />
        </Button>
      </PaneHeader>

      <div
        role="tablist"
        aria-label="Controls"
        onKeyDown={handleKeyDown}
        className="grid shrink-0 grid-cols-2 border-b border-line/50"
      >
        {TABS.map(({ id, label, icon: Icon }) => {
          const selected = tab === id
          return (
            <button
              key={id}
              id={`${baseId}-tab-${id}`}
              type="button"
              role="tab"
              aria-selected={selected}
              aria-controls={`${baseId}-panel-${id}`}
              tabIndex={selected ? 0 : -1}
              onClick={() => setTab(id)}
              className={`-mb-px flex h-10 items-center justify-center gap-2 border-b-2 text-[13px] transition-colors ${
                selected
                  ? 'border-accent font-medium text-ink'
                  : 'border-transparent text-faint hover:text-ink'
              }`}
            >
              <Icon className="h-4 w-4" aria-hidden="true" />
              {label}
            </button>
          )
        })}
      </div>

      <div
        id={`${baseId}-panel-${tab}`}
        role="tabpanel"
        aria-labelledby={`${baseId}-tab-${tab}`}
        className="min-h-0 flex-1 overflow-y-auto"
      >
        {tab === 'parameters' ? (
          <ParameterControls params={params} onChange={onParamsChange} disabled={disabled} />
        ) : (
          <ModelInfo status={connectionStatus} health={health} />
        )}
      </div>
    </>
  )
}