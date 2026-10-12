import type { ReactNode } from 'react'

/** Fixed-height header shared by all three panes so their baselines line up. */
export function PaneHeader({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-12 shrink-0 items-center gap-2.5 border-b border-line/50 px-4">{children}</div>
  )
}