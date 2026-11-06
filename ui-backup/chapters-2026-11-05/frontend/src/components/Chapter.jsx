// The opening of a part of the app: a full-bleed field of its colour with a very large two-line title
// that rises out of a mask. The colours belong to the parts (App.jsx: VIEWS).

export const TONES = {
  red: 'bg-field-red text-ink',
  gold: 'bg-field-gold text-ink',
  green: 'bg-field-green text-paper',
  blue: 'bg-field-blue text-paper',
  bone: 'bg-field-bone text-ink',
  black: 'bg-field-black text-paper',
}

export function Lines({ lines, as: Tag = 'h2', className = '', delay = 80 }) {
  return (
    <Tag className={className} aria-label={lines.join(' ')}>
      {lines.map((text, i) => (
        <span key={text} aria-hidden="true" className="mask-line">
          <span className="animate-line-up" style={{ animationDelay: `${delay + i * 110}ms` }}>{text}</span>
        </span>
      ))}
    </Tag>
  )
}

export default function Chapter({ tone, title, mark, children, aside }) {
  return (
    <header className={`relative flex min-h-[44svh] flex-col justify-between gap-7 overflow-hidden px-4 pb-6 pt-7 sm:px-8 sm:pb-7 sm:pt-8 ${TONES[tone]}`}>
      {mark && (
        <span aria-hidden="true" className="drift pointer-events-none absolute -right-[2%] top-1/2 hidden -translate-y-1/2 select-none whitespace-nowrap text-[calc((100vw-var(--rail))*0.3)] font-semibold leading-none tracking-[-0.06em] opacity-[0.09] lg:block">{mark}</span>
      )}
      <Lines lines={title} className="relative text-[clamp(3rem,calc((100vw-var(--rail))*0.078),7.5rem)] font-semibold leading-[0.93] tracking-[-0.045em]" />
      <div className="relative flex flex-wrap items-end justify-between gap-x-10 gap-y-5">
        <div className="min-w-0 max-w-2xl animate-rise" style={{ animationDelay: '320ms' }}>{children}</div>
        {aside && <div className="animate-rise" style={{ animationDelay: '420ms' }}>{aside}</div>}
      </div>
    </header>
  )
}
