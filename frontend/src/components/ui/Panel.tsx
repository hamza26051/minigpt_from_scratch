import { useId, type ReactNode } from 'react'

type PanelProps = {
  title: string
  description?: string
  actions?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
}

export function Panel({
  title,
  description,
  actions,
  children,
  className = '',
  bodyClassName = '',
}: PanelProps) {
  const headingId = useId()

  return (
    <section
      aria-labelledby={headingId}
      className={`flex flex-col rounded-xl border border-line bg-panel ${className}`}
    >
      <div className="flex min-h-14 items-center justify-between gap-3 border-b border-line px-4 py-2.5">
        <div className="min-w-0">
          <h3 id={headingId} className="text-sm font-semibold">
            {title}
          </h3>
          {description && <p className="mt-0.5 truncate text-xs text-muted">{description}</p>}
        </div>
        {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
      </div>
      <div className={`min-h-0 flex-1 p-4 ${bodyClassName}`}>{children}</div>
    </section>
  )
}