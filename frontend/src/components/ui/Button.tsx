import type { ButtonHTMLAttributes } from 'react'

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary' | 'ghost'
  /** sm = 32px, md = 36px, icon = 32px square (always give it an aria-label). */
  size?: 'sm' | 'md' | 'icon'
  /** Shows a busy state: the button is disabled and keeps its active colour. */
  loading?: boolean
}

const SIZES = {
  sm: 'h-8 px-3 text-xs',
  md: 'h-9 px-4 text-sm',
  icon: 'h-8 w-8 text-xs',
} as const

export function Button({
  variant = 'secondary',
  size = 'md',
  loading = false,
  disabled,
  className = '',
  ...props
}: ButtonProps) {
  const base = 'inline-flex shrink-0 items-center justify-center gap-2 rounded-lg font-medium transition-colors'

  const looks = {
    primary: loading
      ? 'bg-accent/80 text-canvas'
      : 'bg-accent text-canvas hover:bg-[#9bbcfa] disabled:bg-raised disabled:text-faint disabled:hover:bg-raised',
    secondary:
      'bg-raised text-ink ring-1 ring-inset ring-line/70 hover:bg-[#2e3646] disabled:opacity-50 disabled:hover:bg-raised',
    ghost:
      'text-muted hover:bg-raised hover:text-ink disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-muted',
  } as const

  const cursor = loading ? 'cursor-progress' : 'disabled:cursor-not-allowed'

  return (
    <button
      type="button"
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={`${base} ${SIZES[size]} ${looks[variant]} ${cursor} ${className}`}
      {...props}
    />
  )
}