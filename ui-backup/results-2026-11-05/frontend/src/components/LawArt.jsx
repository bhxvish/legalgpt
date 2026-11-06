// The app's mark, a section sign on red tape, and the icons of the navigation rail.

const line = { fill: 'none', stroke: 'currentColor', strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true }

/** Balance scales, drawn on a 24-unit grid: used for the logo and the Verify tab icon. */
function ScalesGlyph({ strokeWidth = 1.6, className }) {
  return (
    <svg viewBox="0 0 24 24" {...line} strokeWidth={strokeWidth} className={className}>
      <path d="M12 3.5v16" />
      <path d="M8 20.5h8" />
      <circle cx="12" cy="3.5" r="0.9" fill="currentColor" stroke="none" />
      <path d="M4.5 6.5h15" />
      <path d="M4.5 6.5 2 13M4.5 6.5 7 13" />
      <path d="M1.8 13a2.7 2.7 0 0 0 5.4 0Z" />
      <path d="M19.5 6.5 17 13M19.5 6.5 22 13" />
      <path d="M16.8 13a2.7 2.7 0 0 0 5.4 0Z" />
    </svg>
  )
}

export function LogoMark({ className = 'size-8' }) {
  return (
    <span aria-hidden="true" className={`grid shrink-0 place-items-center bg-field-red text-ink ${className}`}>
      <svg viewBox="0 0 24 24" className="size-full">
        <text x="12" y="18" textAnchor="middle" fontFamily="Switzer, Arial, sans-serif" fontSize="18" fontWeight="600" fill="currentColor">§</text>
      </svg>
    </span>
  )
}

/* ---------- tab icons (16 px, 20-unit grid) ---------- */

const tab = { viewBox: '0 0 20 20', ...line, strokeWidth: 1.5, className: 'size-5 shrink-0' }

export const TabIcons = {
  explain: (
    // a magnifier over a node: looking inside the model
    <svg {...tab}>
      <circle cx="8.5" cy="8.5" r="5.25" />
      <path d="m12.5 12.5 4.75 4.75" />
      <circle cx="8.5" cy="8.5" r="1.4" />
      <path d="M8.5 5.25v1.85M8.5 9.9v1.85M5.25 8.5h1.85M9.9 8.5h1.85" />
    </svg>
  ),
  system: (
    // three connected stages
    <svg {...tab}>
      <rect x="2.5" y="3" width="5" height="4.5" rx="1.2" />
      <rect x="12.5" y="3" width="5" height="4.5" rx="1.2" />
      <rect x="7.5" y="12.5" width="5" height="4.5" rx="1.2" />
      <path d="M7.5 5.25h5M15 7.5v2.5a1.5 1.5 0 0 1-1.5 1.5h-1M5 7.5v2.5a1.5 1.5 0 0 0 1.5 1.5h1" />
    </svg>
  ),
  chat: (
    <svg {...tab}>
      <path d="M4 4.5h12a1.5 1.5 0 0 1 1.5 1.5v7a1.5 1.5 0 0 1-1.5 1.5H9l-3.5 2.75V14.5H4A1.5 1.5 0 0 1 2.5 13V6A1.5 1.5 0 0 1 4 4.5Z" />
      <path d="M6.5 8.5h7M6.5 11h4.5" />
    </svg>
  ),
  review: (
    // gavel
    <svg {...tab}>
      <g transform="rotate(-45 8.5 8.5)">
        <rect x="4" y="3.5" width="9" height="4.5" rx="1.2" />
        <path d="M8.5 8v9.5" />
      </g>
      <path d="M11 17.5h6.5" />
    </svg>
  ),
  annotate: (
    // pen nib
    <svg {...tab}>
      <path d="M12.8 3.6 16.4 7.2 9 14.6 4.5 15.5 5.4 11Z" />
      <path d="m11 5.4 3.6 3.6" />
      <circle cx="8.3" cy="11.7" r="1" />
      <path d="M3.5 17.5h6" />
    </svg>
  ),
  verify: <ScalesGlyph strokeWidth={1.4} className="size-5 shrink-0" />,
  case: (
    // a case file (folder with a document)
    <svg {...tab}>
      <path d="M2.75 6.25A1.5 1.5 0 0 1 4.25 4.75h3.5l1.5 1.75h6.5a1.5 1.5 0 0 1 1.5 1.5v6.75a1.5 1.5 0 0 1-1.5 1.5H4.25a1.5 1.5 0 0 1-1.5-1.5Z" />
      <path d="M7 10.25h6M7 12.75h4" />
    </svg>
  ),
}

TabIcons.label = TabIcons.annotate // the Label tab (annotation and review, merged)

