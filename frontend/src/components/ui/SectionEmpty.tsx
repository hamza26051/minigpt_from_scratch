import type { ReactNode } from 'react'

type SectionEmptyProps = {
  icon: ReactNode
  title: string
  description: string
}

/** Compact empty state for side-panel sections. */
export function SectionEmpty({ icon, title, description }: SectionEmptyProps) {
  return (
    <div className="flex items-start gap-2.5 rounded-lg border border-dashed border-line/70 px-3 py-2.5">
      <span className="mt-0.5 shrink-0 text-faint" aria-hidden="true">
        {icon}
      </span>
      <div className="min-w-0">
        <p className="text-xs font-medium text-muted">{title}</p>
        <p className="mt-0.5 text-xs leading-relaxed text-faint">{description}</p>
      </div>
    </div>
  )
}