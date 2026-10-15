import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { annotationApi } from '../api.js'
import { stagger } from './motion.jsx'

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

function readStored(key) {
  try {
    return localStorage.getItem(key) || ''
  } catch {
    return ''
  }
}

function writeStored(key, value) {
  try {
    localStorage.setItem(key, value)
  } catch {
    /* storage unavailable: the name just isn't remembered */
  }
}

/** Progress of one case: a bar that grows in, and "labelled / sentences". */
function ProgressBar({ progress, wide = false }) {
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

/** When to annotate by hand (instead of the Review tab): a quiet, collapsible card. */
function WhenToAnnotate() {
  const [open, setOpen] = useState(() => readStored('legalgpt.annotateNote') !== 'closed')
  const toggle = () => {
    writeStored('legalgpt.annotateNote', open ? 'closed' : 'open')
    setOpen(!open)
  }
  const items = [
    ['Sent back from Review', 'Cases the Review tab returned because too many spot checks were wrong.'],
    ['Gold sample', 'About 1 case in 10, labelled without seeing suggestions, to keep measuring how accurate the computer is.'],
    ['Double labelling', 'The same case labelled by two people, to check the label guideline is applied consistently.'],
  ]
  return (
    <section className="overflow-hidden rounded-2xl bg-white/70 ring-1 ring-slate-200 backdrop-blur" data-testid="annotate-note">
      <button type="button" onClick={toggle} aria-expanded={open} className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-slate-50/60 sm:px-5">
        <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-indigo-50 text-indigo-600 ring-1 ring-indigo-200">
          <svg {...stroke} className="size-4">
            <circle cx="10" cy="10" r="7.25" />
            <path d="M10 9v4.5M10 6.6v.1" />
          </svg>
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-semibold text-slate-800">When to annotate by hand</span>
          <span className="block text-xs text-slate-500">
            New judgments normally go through the{' '}
            <a href="#review" onClick={(e) => e.stopPropagation()} className="font-medium text-indigo-600 hover:underline">
              Review
            </a>{' '}
            tab, where you only check the computer's suggestions.
          </span>
        </span>
        <svg {...stroke} className={`size-4 shrink-0 text-slate-400 transition-transform duration-300 ${open ? 'rotate-180' : ''}`}>
          <path d="m5 7.5 5 5 5-5" />
        </svg>
      </button>
      <div className={`grid transition-[grid-template-rows] duration-300 ease-[cubic-bezier(0.2,0,0,1)] ${open ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'}`}>
        <div className="overflow-hidden" inert={!open}>
          <ol className="grid gap-3 border-t border-slate-100 px-4 py-4 sm:grid-cols-3 sm:px-5">
            {items.map(([title, text], i) => (
              <li key={title} className="flex gap-3">
                <span className="grid size-6 shrink-0 place-items-center rounded-full bg-slate-100 text-xs font-semibold text-slate-600">{i + 1}</span>
                <span>
                  <span className="block text-sm font-medium text-slate-800">{title}</span>
                  <span className="mt-0.5 block text-xs leading-relaxed text-slate-500">{text}</span>
                </span>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </section>
  )
}

/** The label scheme as compact chips; each one opens its definition. */
function Guideline({ scheme }) {
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

function CaseAnnotator({ caseId, annotator, scheme, onProgress, onBack }) {
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

export default function AnnotateView() {
  const [annotator, setAnnotator] = useState(() => readStored('legalgpt.annotator'))
  const [nameDraft, setNameDraft] = useState(annotator)
  const [scheme, setScheme] = useState([])
  const [cases, setCases] = useState(null)
  const [openCase, setOpenCase] = useState(null)
  const [error, setError] = useState('')
  const [query, setQuery] = useState('')

  useEffect(() => {
    annotationApi.scheme().then(setScheme).catch((e) => setError(e.message))
  }, [])

  useEffect(() => {
    if (!annotator) return
    annotationApi
      .cases(annotator)
      .then(setCases)
      .catch((e) => setError(e.message))
  }, [annotator, openCase])

  const onProgress = useCallback((caseId, progress) => {
    setCases((cs) => cs?.map((c) => (c.case_id === caseId ? { ...c, progress } : c)))
  }, [])

  if (!annotator) {
    return (
      <div className="relative isolate flex min-h-0 flex-1 items-start justify-center overflow-y-auto px-4 pt-16">
        <div aria-hidden="true" className="bg-dots pointer-events-none absolute inset-0 -z-10" />
        <form
          className="w-full max-w-sm animate-rise rounded-2xl bg-white p-6 shadow-[0_1px_2px_rgb(15_23_42/0.04),0_16px_40px_-20px_rgb(15_23_42/0.3)] ring-1 ring-slate-200"
          onSubmit={(e) => {
            e.preventDefault()
            const name = nameDraft.trim()
            if (!name) return
            writeStored('legalgpt.annotator', name)
            setAnnotator(name)
          }}
        >
          <span className="grid size-10 place-items-center rounded-xl bg-indigo-50 text-indigo-600 ring-1 ring-indigo-200">
            <svg {...stroke} className="size-5">
              <circle cx="10" cy="7" r="3.25" />
              <path d="M3.75 16.5a6.25 6.25 0 0 1 12.5 0" />
            </svg>
          </span>
          <h2 className="mt-4 text-lg font-semibold tracking-tight text-slate-900">Who is annotating?</h2>
          <p className="mt-1 text-sm text-slate-500">Your name is saved with every label so two people's annotations can be compared.</p>
          <input
            autoFocus
            value={nameDraft}
            onChange={(e) => setNameDraft(e.target.value)}
            maxLength={40}
            placeholder="e.g. priya"
            aria-label="Annotator name"
            className="mt-5 w-full rounded-xl border border-slate-200 bg-slate-50/60 px-3.5 py-2.5 text-sm text-slate-900 transition-[border-color,box-shadow,background-color] focus:border-indigo-400 focus:bg-white focus:outline-none focus:ring-4 focus:ring-indigo-100"
          />
          <button
            type="submit"
            disabled={!nameDraft.trim()}
            className="mt-3 w-full rounded-xl bg-accent px-4 py-2.5 text-sm font-semibold text-on-accent shadow-sm transition-[transform,background-color] duration-200 hover:-translate-y-0.5 hover:bg-accent-hover disabled:translate-y-0 disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none"
          >
            Start annotating
          </button>
        </form>
      </div>
    )
  }

  if (openCase && scheme.length) {
    return (
      <CaseAnnotator
        caseId={openCase}
        annotator={annotator}
        scheme={scheme}
        onProgress={onProgress}
        onBack={() => setOpenCase(null)}
      />
    )
  }

  const totals = (cases || []).reduce(
    (t, c) => ({
      started: t.started + (c.progress?.labelled ? 1 : 0),
      done: t.done + (c.progress && c.progress.labelled === c.progress.sentences ? 1 : 0),
      labelled: t.labelled + (c.progress?.labelled || 0),
      sentences: t.sentences + (c.progress?.sentences || 0),
    }),
    { started: 0, done: 0, labelled: 0, sentences: 0 },
  )
  const q = query.trim().toLowerCase()
  const shown = (cases || []).filter(
    (c) => !q || [c.title, c.case_id, c.court, String(c.decision_year), ...c.sections_cited].some((x) => x && String(x).toLowerCase().includes(q)),
  )

  return (
    <div className="relative isolate min-h-0 flex-1 overflow-y-auto px-4 py-8 sm:py-10">
      <div aria-hidden="true" className="bg-dots pointer-events-none absolute inset-0 -z-10" />
      <div className="mx-auto max-w-4xl space-y-5">
        <header className="relative flex flex-wrap items-end justify-between gap-4">
          <div aria-hidden="true" className="pointer-events-none absolute -left-24 -top-24 -z-10 h-64 w-[30rem] max-w-[120%] animate-float rounded-full bg-[radial-gradient(closest-side,var(--color-indigo-200),transparent)] opacity-50 blur-2xl" />
          <div>
            <p className="mb-3 inline-flex animate-rise items-center gap-1.5 rounded-full bg-white/70 px-2.5 py-1 text-xs font-medium text-slate-600 ring-1 ring-slate-200 backdrop-blur">
              <span aria-hidden="true" className="size-1.5 rounded-full bg-indigo-500" />
              Sentence roles · {scheme.length || '…'} labels
            </p>
            <h2 className="animate-rise text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl" style={stagger(1)}>
              Judgments to annotate
            </h2>
            <p className="mt-1.5 animate-rise text-sm text-slate-500" style={stagger(2)}>
              Label each sentence of a judgment with its role. Every label is saved as you go.
            </p>
          </div>
          <div className="flex animate-rise items-center gap-2 rounded-xl bg-white/80 py-1.5 pl-1.5 pr-3 ring-1 ring-slate-200 backdrop-blur" style={stagger(2)}>
            <span className="grid size-7 place-items-center rounded-lg bg-accent text-xs font-bold uppercase text-on-accent">{annotator.slice(0, 1)}</span>
            <span className="text-sm">
              <span className="block text-[11px] leading-tight text-slate-500">Annotating as</span>
              <span className="block font-medium leading-tight text-slate-900">{annotator}</span>
            </span>
            <button
              onClick={() => {
                writeStored('legalgpt.annotator', '')
                setNameDraft('')
                setAnnotator('')
              }}
              className="ml-2 rounded-md px-1.5 py-0.5 text-xs font-medium text-indigo-600 hover:bg-indigo-50"
            >
              Change
            </button>
          </div>
        </header>

        <div className="animate-rise" style={stagger(3)}>
          <WhenToAnnotate />
        </div>

        {cases && cases.length > 0 && (
          <div className="grid animate-rise gap-3 sm:grid-cols-3" style={stagger(4)}>
            {[
              ['Cases', cases.length, `${totals.done} done · ${totals.started} started`],
              ['Sentences labelled', totals.labelled.toLocaleString(), `of ${totals.sentences.toLocaleString()} sentences`],
              ['Overall', totals.sentences ? `${Math.round((100 * totals.labelled) / totals.sentences)}%` : '0%', 'of all sentences labelled'],
            ].map(([label, value, hint]) => (
              <div key={label} className="rounded-2xl bg-white px-4 py-3 ring-1 ring-slate-200">
                <p className="text-xs text-slate-500">{label}</p>
                <p className="mt-0.5 text-xl font-semibold tabular-nums tracking-tight text-slate-900">{value}</p>
                <p className="text-xs text-slate-400">{hint}</p>
              </div>
            ))}
          </div>
        )}

        {scheme.length > 0 && (
          <div className="animate-rise" style={stagger(5)}>
            <Guideline scheme={scheme} />
          </div>
        )}
        {error && <p className="rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-200">{error}</p>}
        {cases === null ? (
          <div className="space-y-2">
            {Array.from({ length: 5 }, (_, i) => (
              <span key={i} className="block h-16 animate-pulse rounded-2xl bg-slate-100" />
            ))}
          </div>
        ) : cases.length === 0 ? (
          <div className="rounded-2xl bg-white px-5 py-8 text-center text-sm text-slate-600 ring-1 ring-slate-200">
            <p className="font-medium text-slate-800">No judgments collected yet</p>
            <p className="mt-1">
              Put .txt files in <code className="rounded bg-slate-100 px-1">data/inbox/</code> and run{' '}
              <code className="rounded bg-slate-100 px-1">python backend/scripts/collect_judgments.py data/inbox</code>.
            </p>
          </div>
        ) : (
          <section className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h3 className="text-sm font-semibold text-slate-800">
                All judgments <span className="font-normal text-slate-400">({shown.length})</span>
              </h3>
              <label className="relative w-full sm:w-64">
                <span className="sr-only">Search judgments</span>
                <svg {...stroke} className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-slate-400">
                  <circle cx="8.75" cy="8.75" r="5" />
                  <path d="m12.5 12.5 4 4" />
                </svg>
                <input
                  type="search"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search by name, court, year, section"
                  className="w-full rounded-lg border border-slate-200 bg-white py-1.5 pl-8 pr-2 text-sm text-slate-900 placeholder:text-slate-400 focus:border-indigo-400 focus:outline-none focus:ring-4 focus:ring-indigo-100"
                />
              </label>
            </div>
            <ul className="space-y-2" data-testid="case-list">
              {shown.map((c, i) => (
                <li key={c.case_id} className="animate-rise" style={stagger(Math.min(i, 12), 40, 200)}>
                  <button
                    onClick={() => setOpenCase(c.case_id)}
                    className="group flex w-full flex-wrap items-center justify-between gap-3 rounded-2xl bg-white px-4 py-3.5 text-left ring-1 ring-slate-200 transition-[transform,box-shadow,background-color] duration-200 hover:-translate-y-0.5 hover:shadow-[0_10px_24px_-14px_rgb(15_23_42/0.35)] hover:ring-indigo-300"
                  >
                    <div className="flex min-w-0 items-center gap-3">
                      <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-slate-100 font-serif text-sm font-bold text-slate-500 transition-colors group-hover:bg-indigo-50 group-hover:text-indigo-600">§</span>
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium text-slate-900">{c.title || c.case_id}</p>
                        <p className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate-500">
                          {[c.title && c.case_id, c.court, c.decision_year].filter(Boolean).join(' · ')}
                          {c.sections_cited.map((s) => (
                            <span key={s} className="rounded bg-slate-100 px-1.5 py-px font-mono text-[11px] text-slate-600">
                              {s}
                            </span>
                          ))}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <ProgressBar progress={c.progress} />
                      <svg {...stroke} className="size-4 text-slate-300 transition-[transform,color] duration-200 group-hover:translate-x-0.5 group-hover:text-indigo-500">
                        <path d="m7.5 5 5 5-5 5" />
                      </svg>
                    </div>
                  </button>
                </li>
              ))}
              {shown.length === 0 && <li className="px-1 py-4 text-sm text-slate-500">No judgment matches "{query}".</li>}
            </ul>
          </section>
        )}
      </div>
    </div>
  )
}
