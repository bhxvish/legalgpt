import { useEffect, useRef } from 'react'
import { Stamp } from './LegalArt.jsx'

// The desk where a judgment is marked up, shared by labelling and checking. The judgment lies on paper
// as a document, the sentence in hand in full ink and the rest quieter, each marked with a highlighter
// in the colour of its role. Beside it, the bench: the role pads, a map of the whole judgment and the
// tally. The classes are written out in full so Tailwind can see them.

// One colour per label, in LabelScheme order (Facts, Law Applied, Precedent, Argument, Ruling, None).
export const MARKS = [
  { mark: 'bg-sky-200', ink: 'text-sky-900', pad: 'bg-pen-facts', cell: 'bg-pen-facts' },
  { mark: 'bg-violet-200', ink: 'text-violet-900', pad: 'bg-pen-law', cell: 'bg-pen-law' },
  { mark: 'bg-amber-200', ink: 'text-amber-900', pad: 'bg-pen-precedent', cell: 'bg-pen-precedent' },
  { mark: 'bg-rose-200', ink: 'text-rose-900', pad: 'bg-pen-argument', cell: 'bg-pen-argument' },
  { mark: 'bg-emerald-200', ink: 'text-emerald-900', pad: 'bg-pen-ruling', cell: 'bg-pen-ruling' },
  { mark: 'bg-slate-200', ink: 'text-slate-800', pad: 'bg-pen-none', cell: 'bg-pen-none' },
]

export const marksOf = (scheme) => Object.fromEntries(scheme.map((l, i) => [l.name, MARKS[i % MARKS.length]]))

/** The screen: the paper on the left, the bench on the right. On a phone the bench sits below. */
export function Desk({ title, meta, onBack, backLabel, paper, bench }) {
  return (
    <div className="flex min-h-0 flex-1 animate-rise flex-col bg-field-black lg:flex-row">
      <div className="light-scope min-h-0 flex-1 overflow-y-auto bg-field-bone px-3 py-4 text-ink sm:px-6 sm:py-6" data-testid="desk-paper">
        <div className="mx-auto max-w-3xl animate-sheet bg-white px-4 py-6 sm:px-10 sm:py-9">
          <header className="border-b-2 border-ink pb-3">
            <h2 className="text-[clamp(1.5rem,1.2rem+1.3vw,2.25rem)] leading-[1.05] tracking-[-0.035em]">{title}</h2>
            {meta && <p className="mt-1.5 text-[15px] text-slate-600">{meta}</p>}
          </header>
          {paper}
        </div>
      </div>
      <aside className="flex max-h-[46dvh] shrink-0 flex-col gap-4 overflow-y-auto bg-field-black p-4 text-paper lg:max-h-none lg:w-[21rem] lg:p-5 [&>*]:shrink-0" aria-label="Marking tools">
        <button type="button" onClick={onBack} className="press inline-flex w-fit items-center gap-2 text-[15px] font-medium text-paper/75 hover:text-paper" aria-label={backLabel}>
          <span aria-hidden="true">←</span> Back to the shelf
        </button>
        {bench}
      </aside>
    </div>
  )
}

/** One sentence on the paper, with its number in the margin and its mark. */
export const Line = ({ n, active, mark, dashed, tag, onClick, children, innerRef, testId, below }) => (
  <li
    ref={innerRef}
    onClick={onClick}
    aria-current={active ? 'true' : undefined}
    data-testid={testId}
    className={`group relative grid cursor-pointer scroll-my-24 grid-cols-[2.25rem_minmax(0,1fr)] gap-x-2 py-1.5 transition-opacity duration-200 ${active ? '' : 'opacity-60 hover:opacity-100'}`}
  >
    <span className={`pt-[0.28em] text-right text-[13px] font-semibold tabular-nums ${active ? 'text-indigo-600' : 'text-slate-500'}`}>
      {active && <span aria-hidden="true" className="absolute -left-4 bottom-1.5 top-1.5 w-1 animate-fade bg-indigo-500 sm:-left-10" />}
      {n}
    </span>
    <div className="min-w-0">
      <p className={`break-words font-serif leading-[1.55] text-ink transition-[font-size] duration-200 ${active ? 'text-[1.1875rem]' : 'text-[1.0625rem]'}`}>
        <span className={`[box-decoration-break:clone] [-webkit-box-decoration-break:clone] ${mark ? `${mark.mark} px-0.5` : ''} ${dashed ? `underline decoration-2 decoration-dashed underline-offset-4 ${dashed.ink}` : ''}`}>{children}</span>
      </p>
      {tag}
      {below}
    </div>
  </li>
)

/** The role pads: the keys are the same as the numbers shown. */
export function Pads({ scheme, chosen, suggested, onPick, label }) {
  return (
    <div className="grid gap-1" role="toolbar" aria-label={label}>
      {scheme.map((l, i) => {
        const m = MARKS[i % MARKS.length]
        const on = chosen === l.name
        return (
          <button
            key={l.name}
            type="button"
            title={l.description}
            aria-label={`${l.name}${l.name === suggested ? ' (suggested)' : ''}, key ${l.shortcut}`}
            aria-pressed={on}
            onClick={() => onPick(l.name)}
            className={`press group flex h-12 items-center gap-3 px-3 text-left text-[17px] font-medium transition-colors duration-150 ${on ? `${m.pad} text-ink` : 'bg-field-panel text-paper hover:bg-field-panel-2'}`}
          >
            <span className={`grid size-7 shrink-0 place-items-center text-[15px] font-semibold ${on ? 'bg-ink text-paper' : `${m.pad} text-ink`}`}>{l.shortcut}</span>
            <span className="min-w-0 flex-1 truncate">{l.name}</span>
            {l.name === suggested && <span className={`shrink-0 text-[13px] ${on ? '' : 'text-paper/70'}`}>suggested</span>}
          </button>
        )
      })}
    </div>
  )
}

/** The whole judgment at a glance: one cell a sentence, in the colour of its mark. */
export function JudgmentMap({ cells, current, onJump, title }) {
  const box = useRef(null)
  useEffect(() => {
    // only the map scrolls: scrollIntoView would also move the bench around it
    const cell = box.current?.querySelector('[aria-current="true"]')
    if (cell) box.current.scrollTop = Math.max(0, cell.offsetTop - box.current.offsetTop - 24)
  }, [current])
  return (
    <section aria-label={title}>
      <p className="text-[14px] font-medium text-paper/75">{title}</p>
      <div ref={box} className="mt-1.5 flex max-h-28 flex-wrap gap-[3px] overflow-y-auto" data-testid="judgment-map">
        {cells.map((c) => (
          <button
            key={c.id}
            type="button"
            onClick={() => onJump(c.id)}
            title={c.title}
            aria-label={c.title}
            aria-current={c.id === current ? 'true' : undefined}
            className={`size-3 transition-transform duration-150 hover:scale-150 ${c.cell || 'bg-field-panel-2'} ${c.id === current ? 'outline outline-2 outline-offset-1 outline-paper' : ''}`}
          />
        ))}
      </div>
    </section>
  )
}

/** How many sentences carry each role so far. */
export function Tally({ scheme, counts }) {
  const most = Math.max(1, ...scheme.map((l) => counts[l.name] || 0))
  return (
    <ul className="space-y-1" aria-label="Marks so far">
      {scheme.map((l, i) => (
        <li key={l.name} className="grid grid-cols-[6.5rem_minmax(0,1fr)_1.75rem] items-center gap-2 text-[14px]">
          <span className="truncate text-paper/80">{l.name}</span>
          <span className="h-2 bg-field-panel">
            <span className={`block h-full origin-left transition-transform duration-500 ease-[cubic-bezier(0.22,1,0.36,1)] ${MARKS[i % MARKS.length].cell}`} style={{ transform: `scaleX(${(counts[l.name] || 0) / most})` }} />
          </span>
          <span className="text-right tabular-nums text-paper/80">{counts[l.name] || 0}</span>
        </li>
      ))}
    </ul>
  )
}

export const Keys = ({ items }) => (
  <p className="flex flex-wrap gap-x-3 gap-y-1 text-[13px] text-paper/70">
    {items.map(([key, what]) => (
      <span key={what}>
        <kbd className="mr-1 bg-field-panel-2 px-1.5 py-px font-sans text-paper">{key}</kbd>
        {what}
      </span>
    ))}
  </p>
)

export function Saving({ status }) {
  return (
    <span className={`text-[14px] ${status.state === 'error' ? 'font-medium text-field-red' : 'text-paper/70'}`} role="status" data-testid="save-status">
      {status.text}
    </span>
  )
}

export { Stamp }
