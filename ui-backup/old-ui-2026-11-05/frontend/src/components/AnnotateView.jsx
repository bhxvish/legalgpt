import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { annotationApi } from '../api.js'

// One colour per label, in LabelScheme order (Facts, Law Applied, Precedent, Argument, Ruling, None).
const PALETTE = [
  { pill: 'bg-sky-100 text-sky-800 ring-sky-300', bar: 'border-l-sky-400', dot: 'bg-sky-400' },
  { pill: 'bg-violet-100 text-violet-800 ring-violet-300', bar: 'border-l-violet-400', dot: 'bg-violet-400' },
  { pill: 'bg-amber-100 text-amber-800 ring-amber-300', bar: 'border-l-amber-400', dot: 'bg-amber-400' },
  { pill: 'bg-rose-100 text-rose-800 ring-rose-300', bar: 'border-l-rose-400', dot: 'bg-rose-400' },
  { pill: 'bg-emerald-100 text-emerald-800 ring-emerald-300', bar: 'border-l-emerald-400', dot: 'bg-emerald-400' },
  { pill: 'bg-slate-200 text-slate-700 ring-slate-300', bar: 'border-l-slate-400', dot: 'bg-slate-400' },
]

const stroke = { viewBox: '0 0 20 20', fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true }

/** Progress of one case: a bar that grows in, and "labelled / sentences". */
export function ProgressBar({ progress, wide = false }) {
  const [shown, setShown] = useState(0)
  const pct = progress?.sentences ? progress.labelled / progress.sentences : 0
  useEffect(() => {
    const t = requestAnimationFrame(() => setShown(pct))
    return () => cancelAnimationFrame(t)
  }, [pct])
  if (!progress) return <span className="text-xs text-slate-400">not opened</span>
  const done = pct === 1
  return (
    <div className="flex items-center gap-2.5" title={`${progress.labelled} of ${progress.sentences} sentences labelled`}>
      <div className={`h-1.5 overflow-hidden rounded-full bg-slate-100 ring-1 ring-slate-900/5 ${wide ? 'w-40' : 'w-24'}`}>
        <div
          className={`h-full origin-left rounded-full transition-transform duration-700 ease-[cubic-bezier(0.2,0,0,1)] ${done ? 'bg-emerald-500' : 'bg-indigo-500'}`}
          style={{ transform: `scaleX(${shown})` }}
        />
      </div>
      <span className={`w-16 text-right text-xs tabular-nums ${done ? 'font-medium text-emerald-700' : 'text-slate-500'}`}>
        {done ? 'Done' : `${progress.labelled}/${progress.sentences}`}
      </span>
    </div>
  )
}

export function Guideline({ scheme }) {
  const [openLabel, setOpenLabel] = useState(null)
  const current = scheme.find((l) => l.name === openLabel)
  return (
    <section className="rounded-2xl bg-white p-4 ring-1 ring-slate-200 sm:p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-slate-800">Labels</h3>
        <p className="text-xs text-slate-500">Full guideline with examples: docs/annotation_guideline.md</p>
      </div>
      <div className="mt-3 flex flex-wrap gap-1.5" role="tablist" aria-label="Label definitions">
        {scheme.map((l, i) => (
          <button
            key={l.name}
            type="button"
            role="tab"
            aria-selected={openLabel === l.name}
            onClick={() => setOpenLabel(openLabel === l.name ? null : l.name)}
            className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium ring-1 transition-[transform,box-shadow] duration-200 hover:-translate-y-px ${PALETTE[i].pill} ${
              openLabel === l.name ? 'shadow-sm ring-2' : ''
            }`}
          >
            <kbd className="font-mono opacity-70">{l.shortcut}</kbd>
            {l.name}
          </button>
        ))}
      </div>
      {current ? (
        <p key={current.name} className="mt-3 animate-rise rounded-lg bg-slate-50 px-3 py-2 text-sm leading-relaxed text-slate-600 ring-1 ring-slate-200">
          <b className="font-semibold text-slate-800">{current.name}:</b> {current.description}
        </p>
      ) : (
        <p className="mt-3 text-xs text-slate-500">Select a label to see its definition.</p>
      )}
    </section>
  )
}

export function CaseAnnotator({ caseId, annotator, scheme, onProgress, onBack }) {
  const [detail, setDetail] = useState(null)
  const [cursor, setCursor] = useState(0)
  const [status, setStatus] = useState({ state: 'idle', text: '' })
  const rowRefs = useRef([])
  const byShortcut = useMemo(() => Object.fromEntries(scheme.map((l) => [l.shortcut, l.name])), [scheme])
  const colour = useMemo(() => Object.fromEntries(scheme.map((l, i) => [l.name, PALETTE[i]])), [scheme])

  useEffect(() => {
    let cancelled = false
    setDetail(null)
    annotationApi
      .getCase(caseId, annotator)
      .then((d) => {
        if (cancelled) return
        setDetail(d)
        const firstOpen = d.sentences.findIndex((s) => !s.label)
        setCursor(firstOpen === -1 ? 0 : firstOpen)
      })
      .catch((e) => !cancelled && setStatus({ state: 'error', text: e.message }))
    return () => {
      cancelled = true
    }
  }, [caseId, annotator])

  useEffect(() => {
    rowRefs.current[cursor]?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [cursor])

  const assign = useCallback(
    async (idx, label) => {
      const previous = detail.sentences[idx].label
      setDetail((d) => ({ ...d, sentences: d.sentences.map((s) => (s.idx === idx ? { ...s, label } : s)) }))
      setStatus({ state: 'saving', text: 'Saving…' })
      try {
        const progress = await annotationApi.saveLabels(caseId, annotator, [{ idx, label }])
        setDetail((d) => ({ ...d, progress }))
        onProgress(caseId, progress)
        setStatus({ state: 'saved', text: 'Saved' })
      } catch (e) {
        setDetail((d) => ({ ...d, sentences: d.sentences.map((s) => (s.idx === idx ? { ...s, label: previous } : s)) }))
        setStatus({ state: 'error', text: `Not saved: ${e.message}` })
      }
    },
    [detail, caseId, annotator, onProgress],
  )

  useEffect(() => {
    if (!detail) return
    const onKey = (e) => {
      if (e.target.closest('input, textarea, select') || e.ctrlKey || e.metaKey || e.altKey) return
      const last = detail.sentences.length - 1
      if (byShortcut[e.key]) {
        e.preventDefault()
        assign(cursor, byShortcut[e.key])
        setCursor((c) => Math.min(c + 1, last))
      } else if (e.key === 'ArrowDown' || e.key === 'j') {
        e.preventDefault()
        setCursor((c) => Math.min(c + 1, last))
      } else if (e.key === 'ArrowUp' || e.key === 'k') {
        e.preventDefault()
        setCursor((c) => Math.max(c - 1, 0))
      } else if (e.key === 'n') {
        e.preventDefault()
        const next = detail.sentences.findIndex((s, i) => i > cursor && !s.label)
        const wrapped = next === -1 ? detail.sentences.findIndex((s) => !s.label) : next
        if (wrapped !== -1) setCursor(wrapped)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [detail, cursor, byShortcut, assign])

  if (!detail) {
    return (
      <div className="mx-auto w-full max-w-4xl space-y-2 px-4 py-8">
        {status.state === 'error' ? (
          <p className="rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-200">{status.text}</p>
        ) : (
          Array.from({ length: 8 }, (_, i) => <span key={i} className="block h-10 animate-pulse rounded-xl bg-slate-100" style={{ opacity: 1 - i * 0.1 }} />)
        )}
      </div>
    )
  }

  const done = detail.progress.labelled === detail.progress.sentences
  const statusTone = status.state === 'error' ? 'text-red-600' : status.state === 'saved' ? 'text-emerald-600' : 'text-slate-400'
  return (
    <div className="flex min-h-0 flex-1 animate-rise flex-col">
      <div className="border-b border-slate-200/70 bg-white/80 px-4 py-3 backdrop-blur">
        <div className="mx-auto flex max-w-4xl flex-wrap items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-3">
            <button
              onClick={onBack}
              className="grid size-9 shrink-0 place-items-center rounded-lg text-slate-500 ring-1 ring-slate-200 hover:bg-slate-100 hover:text-slate-900"
              aria-label="Back to all cases"
              title="All cases"
            >
              <svg {...stroke} className="size-4">
                <path d="M15.5 10h-11M9 4.5 3.5 10 9 15.5" />
              </svg>
            </button>
            <div className="min-w-0">
              <h2 className="truncate text-sm font-semibold text-slate-900">{detail.title || detail.case_id}</h2>
              <p className="truncate text-xs text-slate-500">
                {[detail.title && detail.case_id, detail.court, detail.decision_year, detail.sections_cited.length && `IPC ${detail.sections_cited.join(', ')}`]
                  .filter(Boolean)
                  .join(' · ')}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <span className={`inline-flex items-center gap-1.5 text-xs ${statusTone}`} role="status" data-testid="save-status">
              {status.state === 'saving' && <span aria-hidden="true" className="size-3 animate-spin rounded-full border-2 border-current border-r-transparent" />}
              {status.state === 'saved' && (
                <svg {...stroke} strokeWidth={2} className="size-3.5">
                  <path d="m4.5 10.5 3.5 3.5 7.5-8" />
                </svg>
              )}
              {status.text}
            </span>
            <ProgressBar progress={detail.progress} wide />
          </div>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-5">
        <ol className="mx-auto max-w-4xl space-y-1.5" data-testid="sentence-list">
          {detail.sentences.map((s) => {
            const active = s.idx === cursor
            const c = s.label ? colour[s.label] : null
            return (
              <li
                key={s.idx}
                ref={(el) => (rowRefs.current[s.idx] = el)}
                onClick={() => setCursor(s.idx)}
                className={`flex cursor-pointer gap-3 rounded-xl border-l-4 px-3 py-2.5 text-sm leading-relaxed ring-1 transition-[background-color,box-shadow] duration-200 ${
                  c ? c.bar : 'border-l-transparent'
                } ${active ? 'bg-indigo-50 shadow-sm ring-2 ring-indigo-300' : 'bg-white ring-slate-200/70 hover:bg-slate-50'}`}
                aria-current={active ? 'true' : undefined}
                data-testid="sentence-row"
              >
                <span className={`w-8 shrink-0 pt-0.5 text-right font-mono text-xs ${active ? 'text-indigo-600' : 'text-slate-400'}`}>{s.idx + 1}</span>
                <span className="min-w-0 flex-1 break-words text-slate-800">{s.text}</span>
                <span className="w-28 shrink-0 text-right">
                  {s.label && (
                    <span key={s.label} className={`inline-block animate-pop rounded-md px-1.5 py-0.5 text-xs font-medium ring-1 ${c.pill}`} data-testid="sentence-label">
                      {s.label}
                    </span>
                  )}
                </span>
              </li>
            )
          })}
        </ol>
        {done && (
          <div className="mx-auto mt-5 flex max-w-4xl animate-rise items-center gap-3 rounded-2xl bg-emerald-50 px-4 py-3.5 text-sm text-emerald-800 ring-1 ring-emerald-200">
            <span className="grid size-8 shrink-0 animate-pop place-items-center rounded-full bg-emerald-500 text-white">
              <svg {...stroke} strokeWidth={2} className="size-4">
                <path d="m4.5 10.5 3.5 3.5 7.5-8" />
              </svg>
            </span>
            <span>
              <b className="font-semibold">Every sentence in this case is labelled.</b> Labels are saved; you can still change any of them.
            </span>
          </div>
        )}
      </div>

      <div className="px-4 pb-4 pt-2">
        <div className="mx-auto max-w-4xl rounded-2xl bg-white p-3 shadow-[0_1px_2px_rgb(15_23_42/0.04),0_8px_24px_-12px_rgb(15_23_42/0.18)] ring-1 ring-slate-200">
          <div className="flex flex-wrap gap-1.5" role="toolbar" aria-label="Assign label to the selected sentence">
            {scheme.map((l, i) => {
              const current = detail.sentences[cursor]?.label === l.name
              return (
                <button
                  key={l.name}
                  onClick={() => {
                    assign(cursor, l.name)
                    setCursor((c) => Math.min(c + 1, detail.sentences.length - 1))
                  }}
                  title={l.description}
                  className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm font-medium ring-1 transition-[transform,box-shadow] duration-200 hover:-translate-y-0.5 hover:shadow-sm ${PALETTE[i].pill} ${
                    current ? 'ring-2' : ''
                  }`}
                >
                  <kbd className="grid size-5 place-items-center rounded bg-white/60 font-mono text-[11px] ring-1 ring-current/20">{l.shortcut}</kbd>
                  {l.name}
                </button>
              )
            })}
          </div>
          <p className="mt-2.5 flex flex-wrap gap-x-3 gap-y-1 px-0.5 text-xs text-slate-500">
            <span className="font-medium text-slate-700">
              Sentence {cursor + 1} of {detail.sentences.length}
            </span>
            <span>
              <Kbd>1</Kbd>–<Kbd>{scheme.length}</Kbd> label and advance
            </span>
            <span>
              <Kbd>↑</Kbd>
              <Kbd>↓</Kbd> or <Kbd>k</Kbd>
              <Kbd>j</Kbd> move
            </span>
            <span>
              <Kbd>n</Kbd> next unlabelled
            </span>
          </p>
        </div>
      </div>
    </div>
  )
}

const Kbd = ({ children }) => (
  <kbd className="mx-0.5 inline-grid min-w-[1.25rem] place-items-center rounded border border-slate-200 bg-slate-50 px-1 font-mono text-[11px] text-slate-600">{children}</kbd>
)
