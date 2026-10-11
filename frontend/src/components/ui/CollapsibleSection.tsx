import { useId, useState, type ReactNode } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'

type CollapsibleSectionProps = {
  title: string
  count?: number
  icon: ReactNode
  actions?: ReactNode
  defaultOpen?: boolean
  children: ReactNode
}

export function CollapsibleSection({
  title,
  count,
  icon,
  actions,
  defaultOpen = true,
  children,
}: CollapsibleSectionProps) {
  const [open, setOpen] = useState(defaultOpen)
  const contentId = useId()

  return (
    <section className="border-b border-line/40 last:border-b-0">
      <div className="flex items-center pr-2">
        <button
          type="button"
          aria-expanded={open}
          aria-controls={contentId}
          onClick={() => setOpen((value) => !value)}
          className="flex min-w-0 flex-1 items-center gap-2 px-4 py-3 text-left text-[13px] font-medium transition-colors hover:text-white"
        >
          {open ? (
            <ChevronDown className="h-3.5 w-3.5 shrink-0 text-faint" aria-hidden="true" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 shrink-0 text-faint" aria-hidden="true" />
          )}
          <span className="shrink-0 text-muted" aria-hidden="true">
            {icon}
          </span>
          <span className="truncate">
            {title}
            {count !== undefined && <span className="ml-1 font-normal text-faint">({count})</span>}
          </span>
        </button>
        {actions}
      </div>
      <div id={contentId} hidden={!open} className="px-4 pb-4">
        {children}
      </div>
    </section>
  )
}