import { useEffect, useState, type ReactNode } from 'react'

export type ShellControls = {
  leftOpen: boolean
  rightOpen: boolean
  setLeftOpen: (open: boolean) => void
  setRightOpen: (open: boolean) => void
}

type AppShellProps = {
  left: (controls: ShellControls) => ReactNode
  center: (controls: ShellControls) => ReactNode
  right: (controls: ShellControls) => ReactNode
}

const DESKTOP = '(min-width: 1024px)'
const WIDE = '(min-width: 1280px)'

function matches(query: string): boolean {
  return window.matchMedia(query).matches
}

/**
 * Three-pane application frame.
 * - Large screens: side panes sit next to the centre pane and can be collapsed.
 * - Small screens: side panes open as overlays on top of the centre pane.
 * The centre is the lightest surface; the side panes are slightly darker so they recede.
 */
export function AppShell({ left, center, right }: AppShellProps) {
  const [leftOpen, setLeftOpen] = useState(() => matches(DESKTOP))
  const [rightOpen, setRightOpen] = useState(() => matches(WIDE))

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape' && !matches(DESKTOP)) {
        setLeftOpen(false)
        setRightOpen(false)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const controls: ShellControls = { leftOpen, rightOpen, setLeftOpen, setRightOpen }

  return (
    <div className="flex h-dvh overflow-hidden bg-sidebar">
      <h1 className="sr-only">MiniGPT — From-Scratch Language Model</h1>

      {(leftOpen || rightOpen) && (
        <button
          type="button"
          aria-label="Close side panel"
          onClick={() => {
            setLeftOpen(false)
            setRightOpen(false)
          }}
          className="fixed inset-0 z-20 bg-black/60 lg:hidden"
        />
      )}

      <aside
        aria-label="Model controls"
        className={
          leftOpen
            ? 'fixed inset-y-0 left-0 z-30 flex w-80 max-w-[88vw] flex-col border-r border-line/50 bg-sidebar shadow-2xl shadow-black/60 lg:static lg:z-auto lg:w-[18rem] lg:max-w-none lg:shrink-0 lg:shadow-none'
            : 'hidden'
        }
      >
        {left(controls)}
      </aside>

      <main className="flex min-w-0 flex-1 flex-col bg-canvas">{center(controls)}</main>

      <aside
        aria-label="Session"
        className={
          rightOpen
            ? 'fixed inset-y-0 right-0 z-30 flex w-80 max-w-[88vw] flex-col border-l border-line/50 bg-sidebar shadow-2xl shadow-black/60 lg:static lg:z-auto lg:w-[19rem] lg:max-w-none lg:shrink-0 lg:shadow-none'
            : 'hidden'
        }
      >
        {right(controls)}
      </aside>
    </div>
  )
}