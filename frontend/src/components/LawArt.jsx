// Line drawings with a legal theme: the logo mark, small tab icons, and large faint header
// illustrations (scales, gavel, statute book). All use currentColor; motion stops under
// prefers-reduced-motion (index.css).

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

/** The app's logo: white scales on a deep ink tile with a soft inner highlight. */
export function LogoMark({ className = 'size-9' }) {
  return (
    <span
      aria-hidden="true"
      className={`relative grid shrink-0 place-items-center overflow-hidden rounded-xl bg-[linear-gradient(145deg,oklch(52%_0.13_256),oklch(32%_0.09_262))] text-white shadow-[0_6px_16px_-6px_oklch(35%_0.1_260/0.6),inset_0_1px_0_rgb(255_255_255/0.25)] ring-1 ring-black/5 ${className}`}
    >
      <span className="absolute inset-x-0 top-0 h-1/2 bg-gradient-to-b from-white/15 to-transparent" />
      <ScalesGlyph strokeWidth={1.7} className="relative size-[62%]" />
    </span>
  )
}

/* ---------- tab icons (16 px, 20-unit grid) ---------- */

const tab = { viewBox: '0 0 20 20', ...line, strokeWidth: 1.5, className: 'size-4 shrink-0' }

export const TabIcons = {
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
  verify: <ScalesGlyph strokeWidth={1.4} className="size-4 shrink-0" />,
}

/* ---------- large faint header illustrations ---------- */

const art = { ...line, strokeWidth: 1.1 }

/** Scales of justice; the beam and pans sway gently. */
export function ScalesArt({ className }) {
  return (
    <svg viewBox="0 0 200 200" {...art} className={className}>
      <path d="M100 40v128" />
      <path d="M70 176h60M80 168h40" />
      <path d="M92 168c0-10 3-14 8-16 5 2 8 6 8 16" />
      <circle cx="100" cy="32" r="6" />
      <g className="origin-[100px_52px] animate-[sway_7s_ease-in-out_infinite_alternate]">
        <path d="M34 52h132" />
        <circle cx="100" cy="52" r="3" />
        <path d="M40 52 22 108M40 52l18 56" />
        <path d="M16 108c4 16 44 16 48 0Z" />
        <path d="M160 52l-18 56M160 52l18 56" />
        <path d="M136 108c4 16 44 16 48 0Z" />
      </g>
    </svg>
  )
}

/** A gavel above its round sound block; it gives a small tap now and then. */
export function GavelArt({ className }) {
  return (
    <svg viewBox="0 0 200 200" {...art} className={className}>
      <g className="origin-[150px_150px] animate-[tap_6s_ease-in-out_infinite]">
        <g transform="rotate(-38 92 78)">
          <rect x="44" y="58" width="96" height="40" rx="9" />
          <path d="M62 58v40M122 58v40" />
          <rect x="34" y="64" width="10" height="28" rx="3" />
          <rect x="140" y="64" width="10" height="28" rx="3" />
          <path d="M88 98v84M96 98v84" />
          <path d="M88 182c0 5 8 5 8 0" />
        </g>
      </g>
      <ellipse cx="128" cy="166" rx="44" ry="9" />
      <path d="M84 166v10c0 5 20 9 44 9s44-4 44-9v-10" />
    </svg>
  )
}

/** An open statute book with a section mark. */
export function BookArt({ className }) {
  return (
    <svg viewBox="0 0 200 200" {...art} className={className}>
      <path d="M100 58c-18-12-48-14-72-8v104c24-6 54-4 72 8 18-12 48-14 72-8V50c-24-6-54-4-72 8Z" />
      <path d="M100 58v104" />
      <path d="M40 74c16-3 32-2 46 4M40 90c16-3 32-2 46 4M40 106c16-3 32-2 46 4M40 122c16-3 32-2 46 4" />
      <path d="M114 78c14-6 30-7 46-4M114 94c14-6 30-7 46-4M114 126c14-6 30-7 46-4" />
      <text x="137" y="117" textAnchor="middle" fontFamily="Georgia, serif" fontSize="22" fill="currentColor" stroke="none">
        §
      </text>
    </svg>
  )
}

/** A faint illustration in the top-right of a tab header (wide screens), floating gently. */
export function HeaderArt({ children }) {
  return (
    <div aria-hidden="true" className="pointer-events-none absolute -right-6 -top-10 -z-10 hidden animate-[fade_1s_ease-out] md:block">
      <div className="size-56 animate-[bob_8s_ease-in-out_infinite_alternate] text-indigo-900/[0.09] [&>svg]:size-full">{children}</div>
    </div>
  )
}
