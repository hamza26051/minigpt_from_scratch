type LogoProps = {
  className?: string
}

export function Logo({ className = 'h-9 w-9' }: LogoProps) {
  return (
    <svg
      viewBox="0 0 32 32"
      className={className}
      role="img"
      aria-label="MiniGPT logo"
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect width="32" height="32" rx="8" fill="#2a3140" />
      <g fill="#7fa9f8">
        <rect x="8" y="8" width="4" height="4" rx="1" />
        <rect x="8" y="14" width="4" height="4" rx="1" />
        <rect x="14" y="14" width="4" height="4" rx="1" />
        <rect x="8" y="20" width="4" height="4" rx="1" />
        <rect x="14" y="20" width="4" height="4" rx="1" />
        <rect x="20" y="20" width="4" height="4" rx="1" />
      </g>
    </svg>
  )
}