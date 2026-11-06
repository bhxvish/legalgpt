// The app's mark: a section sign on a square of rubric red, like a seal on a filed document.

export function LogoMark({ className = 'size-8' }) {
  return (
    <span aria-hidden="true" className={`grid shrink-0 place-items-center rounded-[3px] bg-indigo-600 text-slate-50 ${className}`}>
      <svg viewBox="0 0 24 24" className="size-full">
        <text x="12" y="18.2" textAnchor="middle" fontFamily="Newsreader, Georgia, serif" fontSize="19" fontWeight="500" fill="currentColor">§</text>
      </svg>
    </span>
  )
}
