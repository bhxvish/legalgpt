// The app's seal and the icons of the navigation rail.

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

const SEAL_EDGE = 'M24.00 0.00L26.44 2.34L29.34 0.60L31.20 3.42L34.41 2.38L35.60 5.54L38.96 5.24L39.41 8.59L42.76 9.04L42.46 12.40L45.62 13.59L44.58 16.80L47.40 18.66L45.66 21.56L48.00 24.00L45.66 26.44L47.40 29.34L44.58 31.20L45.62 34.41L42.46 35.60L42.76 38.96L39.41 39.41L38.96 42.76L35.60 42.46L34.41 45.62L31.20 44.58L29.34 47.40L26.44 45.66L24.00 48.00L21.56 45.66L18.66 47.40L16.80 44.58L13.59 45.62L12.40 42.46L9.04 42.76L8.59 39.41L5.24 38.96L5.54 35.60L2.38 34.41L3.42 31.20L0.60 29.34L2.34 26.44L0.00 24.00L2.34 21.56L0.60 18.66L3.42 16.80L2.38 13.59L5.54 12.40L5.24 9.04L8.59 8.59L9.04 5.24L12.40 5.54L13.59 2.38L16.80 3.42L18.66 0.60L21.56 2.34Z'

/** The mark: a notary's seal, gold, with the scales of justice pressed into it. */
export function LogoMark({ className = 'size-10' }) {
  return (
    <svg viewBox="0 0 48 48" aria-hidden="true" className={`shrink-0 text-ink ${className}`}>
      <path d={SEAL_EDGE} fill="var(--color-field-gold)" />
      <circle cx="24" cy="24" r="18.4" fill="none" stroke="currentColor" strokeWidth="0.9" />
      <circle cx="24" cy="12.6" r="1.7" fill="currentColor" />
      <rect x="23.05" y="13" width="1.9" height="20.5" fill="currentColor" />
      <rect x="12.2" y="15.4" width="23.6" height="1.9" fill="currentColor" />
      <path d="M15.2 16.6 11 25.4M15.2 16.6l4.2 8.8M32.8 16.6 28.6 25.4M32.8 16.6l4.2 8.8" fill="none" stroke="currentColor" strokeWidth="0.95" />
      <path d="M9.7 25.2h11a5.5 5.5 0 0 1-11 0ZM27.3 25.2h11a5.5 5.5 0 0 1-11 0Z" fill="currentColor" />
      <path d="M18.4 33.2h11.2l1.4 2.2H17Z" fill="currentColor" />
    </svg>
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

