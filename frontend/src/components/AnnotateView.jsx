import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { annotationApi } from '../api.js'

// One colour per label, in LabelScheme order (Facts, Law Applied, Precedent, Argument, Ruling, None).
const PALETTE = [
  { pill: 'bg-sky-100 text-sky-800 ring-sky-300', bar: 'border-l-sky-400' },
  { pill: 'bg-violet-100 text-violet-800 ring-violet-300', bar: 'border-l-violet-400' },
  { pill: 'bg-amber-100 text-amber-800 ring-amber-300', bar: 'border-l-amber-400' },
  { pill: 'bg-rose-100 text-rose-800 ring-rose-300', bar: 'border-l-rose-400' },
  { pill: 'bg-emerald-100 text-emerald-800 ring-emerald-300', bar: 'border-l-emerald-400' },
  { pill: 'bg-slate-200 text-slate-700 ring-slate-300', bar: 'border-l-slate-400' },
]

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

function ProgressBar({ progress }) {
  if (!progress) return <span className="text-xs text-slate-400">not opened</span>
  const pct = progress.sentences ? Math.round((100 * progress.labelled) / progress.sentences) : 0
  return (
    <div className="flex items-center gap-2" title={`${progress.labelled} of ${progress.sentences} sentences labelled`}>
      <div className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-200">
        <div className={`h-full ${pct === 100 ? 'bg-emerald-500' : 'bg-indigo-500'}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs tabular-nums text-slate-500">
        {progress.labelled}/{progress.sentences}
      </span>
    </div>
  )
}

function Guideline({ scheme }) {
  return (
    <details className="rounded-lg bg-white px-4 py-2 text-sm ring-1 ring-slate-200">
      <summary className="cursor-pointer select-none font-medium text-slate-700">Label definitions</summary>
      <dl className="mt-2 grid gap-2 sm:grid-cols-2">
        {scheme.map((l, i) => (
          <div key={l.name}>
            <dt>
              <span className={`rounded px-1.5 py-0.5 text-xs font-semibold ring-1 ${PALETTE[i].pill}`}>
                {l.shortcut} · {l.name}
              </span>
            </dt>
            <dd className="mt-1 text-slate-600">{l.description}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-2 text-xs text-slate-500">Full guideline with examples: docs/annotation_guideline.md</p>
    </details>
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
    rowRefs.current[cursor]?.scrollIntoView({ block: 'nearest' })
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
    return <p className="p-6 text-sm text-slate-500">{status.state === 'error' ? status.text : 'Loading judgment…'}</p>
  }

  const done = detail.progress.labelled === detail.progress.sentences
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="border-b border-slate-200 bg-white px-4 py-3">
        <div className="mx-auto flex max-w-4xl flex-wrap items-center justify-between gap-2">
          <div className="min-w-0">
            <button onClick={onBack} className="text-xs text-indigo-600 hover:underline">
              ← All cases
            </button>
            <h2 className="truncate text-sm font-semibold text-slate-900">{detail.title || detail.case_id}</h2>
            <p className="text-xs text-slate-500">
              {[detail.title && detail.case_id, detail.court, detail.decision_year, detail.sections_cited.length && `IPC ${detail.sections_cited.join(', ')}`]
                .filter(Boolean)
                .join(' · ')}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span
              className={`text-xs ${status.state === 'error' ? 'text-red-600' : 'text-slate-400'}`}
              role="status"
              data-testid="save-status"
            >
              {status.text}
            </span>
            <ProgressBar progress={detail.progress} />
          </div>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <ol className="mx-auto max-w-4xl space-y-1" data-testid="sentence-list">
          {detail.sentences.map((s) => {
            const active = s.idx === cursor
            const c = s.label ? colour[s.label] : null
            return (
              <li
                key={s.idx}
                ref={(el) => (rowRefs.current[s.idx] = el)}
                onClick={() => setCursor(s.idx)}
                className={`flex cursor-pointer gap-3 rounded-md border-l-4 px-3 py-2 text-sm leading-relaxed ${
                  c ? c.bar : 'border-l-transparent'
                } ${active ? 'bg-indigo-50 ring-2 ring-indigo-300' : 'bg-white hover:bg-slate-50'}`}
                aria-current={active ? 'true' : undefined}
                data-testid="sentence-row"
              >
                <span className="w-8 shrink-0 pt-0.5 text-right font-mono text-xs text-slate-400">{s.idx + 1}</span>
                <span className="min-w-0 flex-1 break-words text-slate-800">{s.text}</span>
                <span className="w-24 shrink-0 text-right">
                  {s.label && (
                    <span className={`rounded px-1.5 py-0.5 text-xs font-medium ring-1 ${c.pill}`} data-testid="sentence-label">
                      {s.label}
                    </span>
                  )}
                </span>
              </li>
            )
          })}
        </ol>
        {done && (
          <p className="mx-auto mt-4 max-w-4xl rounded-lg bg-emerald-50 px-4 py-3 text-sm text-emerald-800 ring-1 ring-emerald-200">
            Every sentence in this case is labelled. Labels are saved; you can still change any of them.
          </p>
        )}
      </div>

      <div className="border-t border-slate-200 bg-white px-4 py-3">
        <div className="mx-auto max-w-4xl">
          <div className="flex flex-wrap gap-1.5" role="toolbar" aria-label="Assign label to the selected sentence">
            {scheme.map((l, i) => (
              <button
                key={l.name}
                onClick={() => {
                  assign(cursor, l.name)
                  setCursor((c) => Math.min(c + 1, detail.sentences.length - 1))
                }}
                title={l.description}
                aria-label={`${l.name}, key ${l.shortcut}`}
                className={`rounded-md px-2.5 py-1.5 text-sm font-medium ring-1 hover:brightness-95 ${PALETTE[i].pill}`}
              >
                <kbd className="mr-1 font-mono text-xs opacity-70">{l.shortcut}</kbd>
                {l.name}
              </button>
            ))}
          </div>
          <p className="mt-2 text-xs text-slate-400">
            Sentence {cursor + 1} of {detail.sentences.length} · keys 1–{scheme.length} label and advance · ↑/↓ or k/j move · n next
            unlabelled · hover a label for its definition
          </p>
        </div>
      </div>
    </div>
  )
}

export default function AnnotateView() {
  const [annotator, setAnnotator] = useState(() => readStored('legalgpt.annotator'))
  const [nameDraft, setNameDraft] = useState(annotator)
  const [scheme, setScheme] = useState([])
  const [cases, setCases] = useState(null)
  const [openCase, setOpenCase] = useState(null)
  const [error, setError] = useState('')

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
      <div className="mx-auto mt-12 w-full max-w-sm px-4">
        <form
          className="rounded-xl bg-white p-6 ring-1 ring-slate-200"
          onSubmit={(e) => {
            e.preventDefault()
            const name = nameDraft.trim()
            if (!name) return
            writeStored('legalgpt.annotator', name)
            setAnnotator(name)
          }}
        >
          <h2 className="font-semibold text-slate-900">Who is annotating?</h2>
          <p className="mt-1 text-sm text-slate-500">
            Your name is saved with every label so two people's annotations can be compared.
          </p>
          <input
            autoFocus
            value={nameDraft}
            onChange={(e) => setNameDraft(e.target.value)}
            maxLength={40}
            placeholder="e.g. priya"
            aria-label="Annotator name"
            className="mt-4 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
          />
          <button
            type="submit"
            disabled={!nameDraft.trim()}
            className="mt-3 w-full rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-40"
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

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6">
      <div className="mx-auto max-w-4xl space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-lg font-semibold text-slate-900">Judgments to annotate</h2>
          <p className="text-sm text-slate-500">
            Annotating as <span className="font-medium text-slate-800">{annotator}</span>{' '}
            <button
              onClick={() => {
                writeStored('legalgpt.annotator', '')
                setNameDraft('')
                setAnnotator('')
              }}
              className="text-indigo-600 hover:underline"
            >
              change
            </button>
          </p>
        </div>
        {scheme.length > 0 && <Guideline scheme={scheme} />}
        {error && <p className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-200">{error}</p>}
        {cases === null ? (
          <p className="text-sm text-slate-500">Loading…</p>
        ) : cases.length === 0 ? (
          <div className="rounded-lg bg-white px-4 py-6 text-sm text-slate-600 ring-1 ring-slate-200">
            No judgments collected yet. Put .txt files in <code>data/inbox/</code> and run{' '}
            <code>python backend/scripts/collect_judgments.py data/inbox</code>.
          </div>
        ) : (
          <ul className="divide-y divide-slate-100 rounded-lg bg-white ring-1 ring-slate-200" data-testid="case-list">
            {cases.map((c) => (
              <li key={c.case_id}>
                <button
                  onClick={() => setOpenCase(c.case_id)}
                  className="flex w-full flex-wrap items-center justify-between gap-2 px-4 py-3 text-left hover:bg-slate-50"
                >
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium text-slate-900">{c.title || c.case_id}</p>
                    <p className="text-xs text-slate-500">
                      {[c.title && c.case_id, c.court, c.decision_year, c.sections_cited.length && `IPC ${c.sections_cited.join(', ')}`]
                        .filter(Boolean)
                        .join(' · ')}
                    </p>
                  </div>
                  <ProgressBar progress={c.progress} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
