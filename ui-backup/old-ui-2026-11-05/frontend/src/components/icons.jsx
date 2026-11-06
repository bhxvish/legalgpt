// Small inline icons (1.5 stroke, 20px grid), so the app needs no icon library.
const base = { viewBox: '0 0 20 20', fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true }

export function SidebarIcon({ className = 'size-5' }) {
  return (
    <svg {...base} className={className}>
      <rect x="2.75" y="3.75" width="14.5" height="12.5" rx="2.5" />
      <path d="M7.75 3.75v12.5" />
    </svg>
  )
}

export function NewChatIcon({ className = 'size-5' }) {
  return (
    <svg {...base} className={className}>
      <path d="M9 3.75H5.25a2.5 2.5 0 0 0-2.5 2.5v8.5a2.5 2.5 0 0 0 2.5 2.5h8.5a2.5 2.5 0 0 0 2.5-2.5V11" />
      <path d="M14.6 2.9a1.5 1.5 0 0 1 2.12 2.12L10.5 11.25l-2.75.6.6-2.75 6.25-6.2Z" />
    </svg>
  )
}

export function SearchIcon({ className = 'size-4' }) {
  return (
    <svg {...base} className={className}>
      <circle cx="8.75" cy="8.75" r="5" />
      <path d="m12.5 12.5 4 4" />
    </svg>
  )
}

export function SendIcon({ className = 'size-4' }) {
  return (
    <svg {...base} strokeWidth={1.8} className={className}>
      <path d="M10 16V4M5 9l5-5 5 5" />
    </svg>
  )
}

export function StopIcon({ className = 'size-3.5' }) {
  return (
    <svg viewBox="0 0 20 20" fill="currentColor" aria-hidden="true" className={className}>
      <rect x="4" y="4" width="12" height="12" rx="2" />
    </svg>
  )
}

export function TrashIcon({ className = 'size-4' }) {
  return (
    <svg {...base} className={className}>
      <path d="M3.75 5.75h12.5M8 5.75V4.5a1.25 1.25 0 0 1 1.25-1.25h1.5A1.25 1.25 0 0 1 12 4.5v1.25M5.5 5.75l.7 9.6a1.5 1.5 0 0 0 1.5 1.4h4.6a1.5 1.5 0 0 0 1.5-1.4l.7-9.6" />
    </svg>
  )
}

export function CloseIcon({ className = 'size-4' }) {
  return (
    <svg {...base} className={className}>
      <path d="m5 5 10 10M15 5 5 15" />
    </svg>
  )
}
