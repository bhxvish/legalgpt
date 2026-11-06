// The app's legal objects, drawn in flat shapes: scales that really weigh, a rubber stamp, the red
// tape round a case file and a court front. All use currentColor unless a colour is theirs.

/** Scales of justice. `tilt` is in degrees: negative lowers the left pan, positive the right. The pans
 * stay level and hang from the beam, so they move by the arm's rise and fall. `busy` weighs to and fro. */
export function Scales({ tilt = 0, arm = 140, drop = 130, busy = false, base = true, className = '', style }) {
  const W = arm * 2 + 120
  const cx = W / 2
  const pivot = 46
  const H = pivot + drop + (base ? 96 : 66)
  const rise = arm * Math.sin((tilt * Math.PI) / 180)
  const move = 'transform 1100ms cubic-bezier(0.22, 1, 0.36, 1)'
  const pan = (x, dy, sway) => (
    <g style={{ transform: `translateY(${dy}px)`, transition: move, animation: busy ? `${sway} 1.5s ease-in-out infinite alternate` : undefined }}>
      <path d={`M${x} ${pivot + 4} L${x - 50} ${pivot + drop} M${x} ${pivot + 4} L${x + 50} ${pivot + drop}`} stroke="currentColor" strokeWidth="2.5" fill="none" />
      <path d={`M${x - 58} ${pivot + drop} h116 a58 42 0 0 1 -116 0 Z`} fill="currentColor" />
    </g>
  )
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={className} style={style} aria-hidden="true">
      <circle cx={cx} cy={pivot - 22} r="11" fill="currentColor" />
      <rect x={cx - 6} y={pivot - 16} width="12" height={H - pivot + 16} fill="currentColor" />
      {base && <path d={`M${cx - 74} ${H - 26} h148 l14 14 h-176 Z`} fill="currentColor" />}
      {base && <rect x={cx - 104} y={H - 12} width="208" height="12" fill="currentColor" />}
      <g
        style={{
          transform: `rotate(${tilt}deg)`,
          transformOrigin: `${cx}px ${pivot}px`,
          transition: move,
          animation: busy ? 'weigh-beam 1.5s ease-in-out infinite alternate' : undefined,
        }}
      >
        <rect x={cx - arm - 12} y={pivot - 5} width={arm * 2 + 24} height="10" fill="currentColor" />
      </g>
      {pan(cx - arm, -rise, 'weigh-left')}
      {pan(cx + arm, rise, 'weigh-right')}
    </svg>
  )
}

/** A rubber stamp, brought down on the page. */
export function Stamp({ children, className = '', ...rest }) {
  return (
    <span
      className={`inline-block -rotate-6 animate-stamp whitespace-nowrap border-[3px] border-current px-2.5 py-0.5 text-[15px] font-semibold uppercase leading-tight tracking-[0.06em] shadow-[inset_0_0_0_1.5px_var(--stamp-gap,transparent)] ${className}`}
      {...rest}
    >
      {children}
    </span>
  )
}

/** The red tape that ties an Indian court file, with its knot. */
export function RedTape({ className = '' }) {
  return (
    <span aria-hidden="true" className={`pointer-events-none absolute w-4 bg-field-red sm:w-5 ${className}`}>
      <svg viewBox="0 0 60 44" className="absolute left-1/2 top-[38%] w-14 -translate-x-1/2 text-field-red sm:w-16">
        <path d="M30 22 4 4v36Z M30 22 56 4v36Z" fill="currentColor" />
        <rect x="23" y="14" width="14" height="16" fill="#cf3620" />
      </svg>
    </span>
  )
}

/** The front of a court: pediment, columns, steps. */
export function CourtFront({ className = '', columns = 10 }) {
  const W = 1200
  const gap = (W - 160) / (columns - 1)
  return (
    <svg viewBox={`0 0 ${W} 300`} className={className} aria-hidden="true" preserveAspectRatio="xMidYMax meet">
      <path d={`M${W / 2} 0 L${W - 30} 78 H30 Z`} fill="currentColor" />
      <rect x="30" y="86" width={W - 60} height="16" fill="currentColor" />
      {Array.from({ length: columns }, (_, i) => {
        const x = 80 + i * gap
        return (
          <g key={i} fill="currentColor">
            <rect x={x - 24} y="110" width="48" height="9" />
            <rect x={x - 16} y="119" width="32" height="130" />
            <rect x={x - 24} y="249" width="48" height="9" />
          </g>
        )
      })}
      <rect x="10" y="264" width={W - 20} height="12" fill="currentColor" />
      <rect x="0" y="282" width={W} height="18" fill="currentColor" />
    </svg>
  )
}
