import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { annotationApi, reviewApi } from '../api.js'
import { HeaderArt, GavelArt } from './LawArt.jsx'
import { stagger } from './motion.jsx'
import RoleExplanation from './RoleExplanation.jsx'

const PALETTE = [
  'bg-sky-100 text-sky-800 ring-sky-300',
  'bg-violet-100 text-violet-800 ring-violet-300',
  'bg-amber-100 text-amber-800 ring-amber-300',
  'bg-rose-100 text-rose-800 ring-rose-300',
  'bg-emerald-100 text-emerald-800 ring-emerald-300',
]

const KIND = {
  mandatory: { text: 'Needs review', hint: 'The model was unsure about this sentence.', cls: 'bg-red-50 text-red-700 ring-red-200' },
  audit: { text: 'Spot check', hint: 'The model was sure. We check a few of these to catch confident mistakes.', cls: 'bg-amber-50 text-amber-800 ring-amber-200' },
}

const STATUS = {
  open: { text: 'To review', cls: 'bg-indigo-50 text-indigo-700 ring-indigo-200' },
  promoted: { text: 'Done — added to corpus', cls: 'bg-emerald-50 text-emerald-700 ring-emerald-200' },
  escalated: { text: 'Sent to full annotation', cls: 'bg-amber-50 text-amber-800 ring-amber-200' },
}

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
    /* storage unavailable */
  }
}

const pct = (x) => `${Math.round(x * 100)}%`

const stroke = { viewBox: '0 0 20 20', fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true }

function Pill({ label, colour }) {
  return <span className={`inline-block rounded-md px-1.5 py-0.5 text-xs font-medium ring-1 ${colour}`}>{label}</span>
}

/** A bar that grows to `value` (0..1) after mounting. */
function Meter({ value, tone = 'bg-indigo-500', className = 'w-24' }) {
  const [shown, setShown] = useState(0)
  useEffect(() => {
    const t = requestAnimationFrame(() => setShown(value))
    return () => cancelAnimationFrame(t)
  }, [value])
  return (
    <div className={`h-1.5 overflow-hidden rounded-full bg-slate-100 ring-1 ring-slate-900/5 ${className}`}>
      <div className={`h-full origin-left rounded-full transition-transform duration-700 ease-[cubic-bezier(0.2,0,0,1)] ${tone}`} style={{ transform: `scaleX(${shown})` }} />
    </div>
  )
}

const Kbd = ({ children }) => (
  <kbd className="mx-0.5 inline-grid min-w-[1.25rem] place-items-center rounded border border-slate-200 bg-slate-50 px-1 font-mono text-[11px] text-slate-600">{children}</kbd>
)

const UPLOAD = {
  queued: { text: 'Waiting', cls: 'bg-slate-50 text-slate-600 ring-slate-200', busy: true },
  reading: { text: 'Reading', cls: 'bg-indigo-50 text-indigo-700 ring-indigo-200', busy: true },
  classifying: { text: 'Annotating', cls: 'bg-indigo-50 text-indigo-700 ring-indigo-200', busy: true },
  done: { text: 'Ready to review', cls: 'bg-emerald-50 text-emerald-700 ring-emerald-200' },
  rejected: { text: 'Not accepted', cls: 'bg-amber-50 text-amber-800 ring-amber-200' },
  failed: { text: 'Failed', cls: 'bg-red-50 text-red-700 ring-red-200' },
}

/** Upload judgments (.txt / .pdf): the server screens each one, InLegalBERT suggests a role for every
 * sentence, and the case joins the list below with the uncertain sentences highlighted. */
function UploadPanel({ uploader, onFinished, onOpen }) {
  const [jobs, setJobs] = useState([])
  const [local, setLocal] = useState([]) // uploads the server refused outright (wrong type, too big)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)
  const seenDone = useRef(new Set())

  const refresh = useCallback(() => {
    reviewApi
      .uploads()
      .then((list) => {
        setJobs(list)
        const newlyDone = list.filter((j) => j.status === 'done' && !seenDone.current.has(j.job_id))
        newlyDone.forEach((j) => seenDone.current.add(j.job_id))
        if (newlyDone.length) onFinished()
      })
      .catch(() => {})
  }, [onFinished])

  useEffect(() => {
    reviewApi
      .uploads()
      .then((list) => {
        list.filter((j) => j.status === 'done').forEach((j) => seenDone.current.add(j.job_id))
        setJobs(list)
      })
      .catch(() => {})
  }, [])

  const busy = jobs.some((j) => UPLOAD[j.status]?.busy)
  useEffect(() => {
    if (!busy) return
    const t = setInterval(refresh, 1500)
    return () => clearInterval(t)
  }, [busy, refresh])

  const send = async (files) => {
    for (const file of files) {
      try {
        await reviewApi.upload(file, uploader)
      } catch (e) {
        setLocal((l) => [{ job_id: `local-${Date.now()}-${file.name}`, filename: file.name, status: 'rejected',
                           message: e.message.replace(/^HTTP \d+: /, '') }, ...l])
      }
    }
    refresh()
  }

  const shown = [...local, ...jobs].slice(0, 8)
  return (
    <section className="rounded-2xl bg-white p-4 ring-1 ring-slate-200 sm:p-5" data-testid="upload-panel">
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          send([...e.dataTransfer.files])
        }}
        className={`flex flex-wrap items-center gap-4 rounded-xl border-2 border-dashed px-4 py-4 transition-[border-color,background-color,transform] duration-200 ${
          dragging ? 'scale-[1.01] border-indigo-400 bg-indigo-50' : 'border-slate-200 hover:border-slate-300'
        }`}
      >
        <span className={`grid size-11 shrink-0 place-items-center rounded-xl ring-1 transition-colors ${dragging ? 'bg-indigo-100 text-indigo-600 ring-indigo-200' : 'bg-slate-50 text-slate-500 ring-slate-200'}`}>
          <svg {...stroke} className={`size-5 transition-transform duration-300 ${dragging ? '-translate-y-0.5' : ''}`}>
            <path d="M10 13V3.75M6.25 7.5 10 3.75l3.75 3.75" />
            <path d="M3.75 12.5v1.75a2 2 0 0 0 2 2h8.5a2 2 0 0 0 2-2V12.5" />
          </svg>
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-slate-800">{dragging ? 'Drop to add these judgments' : 'Add judgments'}</p>
          <p className="mt-0.5 text-xs leading-relaxed text-slate-500">
            Drop .txt or .pdf files here. Each one is checked, annotated by the computer, and added below for you to
            review; nothing enters the corpus until someone signs it off.
          </p>
        </div>
        <button
          onClick={() => inputRef.current?.click()}
          className="rounded-xl bg-accent px-4 py-2 text-sm font-semibold text-on-accent shadow-sm transition-[transform,background-color] duration-200 hover:-translate-y-0.5 hover:bg-accent-hover"
        >
          Choose files
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".txt,.pdf"
          multiple
          className="hidden"
          data-testid="upload-input"
          onChange={(e) => {
            send([...e.target.files])
            e.target.value = ''
          }}
        />
      </div>
      {shown.length > 0 && (
        <ul className="mt-3 space-y-1.5" data-testid="upload-list">
          {shown.map((j) => {
            const st = UPLOAD[j.status] || UPLOAD.failed
            return (
              <li
                key={j.job_id}
                className="flex animate-rise flex-wrap items-center gap-2 rounded-xl bg-slate-50/70 px-3 py-2 text-sm ring-1 ring-slate-200/70"
                data-testid="upload-job"
                data-status={j.status}
              >
                {st.busy && <span aria-hidden="true" className="size-3.5 animate-spin rounded-full border-2 border-indigo-500 border-r-transparent" />}
                <Pill label={st.text} colour={st.cls} />
                <span className="font-medium text-slate-800">{j.filename}</span>
                <span className={`min-w-0 flex-1 text-xs ${st.busy ? 'text-shimmer' : 'text-slate-600'}`}>{j.message}</span>
                {j.status === 'done' && (
                  <button onClick={() => onOpen(j.case_id)} className="rounded-md px-2 py-0.5 text-xs font-semibold text-indigo-600 hover:bg-indigo-50">
                    Open →
                  </button>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}

function CaseReview({ caseId, reviewer, scheme, onBack }) {
  const [detail, setDetail] = useState(null)
  const [showAll, setShowAll] = useState(false)
  const [cursor, setCursor] = useState(null) // sentence_id
  const [status, setStatus] = useState({ state: 'idle', text: '' })
  const [result, setResult] = useState(null)
  const [explainFor, setExplainFor] = useState(null) // sentence_id whose explanation is open
  const rowRefs = useRef({})
  const colour = useMemo(() => Object.fromEntries(scheme.map((l, i) => [l.name, PALETTE[i % PALETTE.length]])), [scheme])
  const byShortcut = useMemo(() => Object.fromEntries(scheme.map((l) => [l.shortcut, l.name])), [scheme])

  const load = useCallback(() => reviewApi.getCase(caseId).then(setDetail), [caseId])

  useEffect(() => {
    load().catch((e) => setStatus({ state: 'error', text: e.message }))
  }, [load])

  const rows = useMemo(() => (detail ? detail.items.filter((it) => showAll || it.kind !== 'auto') : []), [detail, showAll])
  const flagged = useMemo(() => (detail ? detail.items.filter((it) => it.kind !== 'auto') : []), [detail])

  useEffect(() => {
    if (detail && cursor === null) {
      const first = flagged.find((it) => it.status === 'pending') || flagged[0]
      if (first) setCursor(first.sentence_id)
    }
  }, [detail, cursor, flagged])

  useEffect(() => {
    if (cursor) rowRefs.current[cursor]?.scrollIntoView({ block: 'nearest' })
  }, [cursor])

  const readOnly = detail?.status !== 'open'

  // Keystrokes can arrive faster than React re-renders. The handlers below read and write these
  // refs synchronously, so the second of two quick keys already sees the first one's effect
  // (otherwise it would act on the same sentence again and silently skip the next one).
  const detailRef = useRef(detail)
  const cursorRef = useRef(cursor)
  const showAllRef = useRef(showAll)
  detailRef.current = detail
  cursorRef.current = cursor
  showAllRef.current = showAll

  const setCursorNow = (id) => {
    cursorRef.current = id
    setCursor(id)
  }

  const pendingAfter = (items, id) => {
    const flaggedNow = items.filter((it) => it.kind !== 'auto')
    const i = flaggedNow.findIndex((r) => r.sentence_id === id)
    return flaggedNow.slice(i + 1).find((r) => r.status === 'pending') || flaggedNow.find((r) => r.status === 'pending')
  }

  const choose = useCallback(
    async (sentenceId, label, advance = true) => {
      const d = detailRef.current
      const item = d?.items.find((it) => it.sentence_id === sentenceId)
      if (!item || d.status !== 'open' || item.kind === 'auto') return
      const items = d.items.map((it) =>
        it.sentence_id === sentenceId ? { ...it, final_label: label, status: label === it.predicted ? 'accepted' : 'corrected' } : it,
      )
      detailRef.current = { ...d, items }
      setDetail((prev) => ({ ...prev, items }))
      if (advance) {
        const next = pendingAfter(items, sentenceId)
        if (next) setCursorNow(next.sentence_id)
      }
      setStatus({ state: 'saving', text: 'Saving…' })
      try {
        const r = await reviewApi.setLabel(caseId, sentenceId, reviewer, label)
        setDetail((prev) => ({ ...prev, reviewed: r.reviewed, audit_error_rate: r.audit_error_rate, corrected: r.corrected }))
        setStatus({ state: 'saved', text: 'Saved' })
      } catch (e) {
        setStatus({ state: 'error', text: `Not saved: ${e.message}` })
        load()
      }
    },
    [caseId, reviewer, load],
  )

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.closest('input, textarea, select, button') || e.ctrlKey || e.metaKey || e.altKey) return
      const d = detailRef.current
      if (!d) return
      const id = cursorRef.current
      const item = d.items.find((it) => it.sentence_id === id)
      const visible = d.items.filter((it) => showAllRef.current || it.kind !== 'auto')
      if ((e.key === 'Enter' || e.key === 'a') && item) {
        e.preventDefault()
        choose(id, item.predicted)
      } else if (byShortcut[e.key] && item) {
        e.preventDefault()
        choose(id, byShortcut[e.key])
      } else if (e.key === 'ArrowDown' || e.key === 'j' || e.key === 'ArrowUp' || e.key === 'k') {
        e.preventDefault()
        const i = visible.findIndex((r) => r.sentence_id === id)
        const delta = e.key === 'ArrowDown' || e.key === 'j' ? 1 : -1
        const next = visible[Math.min(Math.max(i + delta, 0), visible.length - 1)]
        if (next) setCursorNow(next.sentence_id)
      } else if (e.key === 'w' && id) {
        e.preventDefault()
        setExplainFor((cur) => (cur === id ? null : id))
      } else if (e.key === 'n') {
        e.preventDefault()
        const next = pendingAfter(d.items, id)
        if (next) setCursorNow(next.sentence_id)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [byShortcut, choose])

  const signOff = async () => {
    setStatus({ state: 'saving', text: 'Signing off…' })
    try {
      const r = await reviewApi.signOff(caseId, reviewer)
      setResult(r)
      setStatus({ state: 'idle', text: '' })
      load()
    } catch (e) {
      setStatus({ state: 'error', text: e.message })
    }
  }

  if (!detail) {
    return (
      <div className="mx-auto w-full max-w-4xl space-y-2 px-4 py-8">
        {status.state === 'error' ? (
          <p className="rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-200">{status.text}</p>
        ) : (
          Array.from({ length: 6 }, (_, i) => <span key={i} className="block h-16 animate-pulse rounded-xl bg-slate-100" style={{ opacity: 1 - i * 0.12 }} />)
        )}
      </div>
    )
  }

  const done = detail.reviewed === detail.flagged
  const overLimit = detail.audit_error_rate > detail.max_audit_error
  const current = detail.items.find((it) => it.sentence_id === cursor)

  return (
    <div className="flex min-h-0 flex-1 animate-rise flex-col">
      <div className="border-b border-slate-200/70 bg-white/80 px-4 py-3 backdrop-blur">
        <div className="mx-auto max-w-4xl">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex min-w-0 items-center gap-3">
              <button
                onClick={onBack}
                className="grid size-9 shrink-0 place-items-center rounded-lg text-slate-500 ring-1 ring-slate-200 hover:bg-slate-100 hover:text-slate-900"
                aria-label="Back to all judgments"
                title="All judgments"
              >
                <svg {...stroke} className="size-4">
                  <path d="M15.5 10h-11M9 4.5 3.5 10 9 15.5" />
                </svg>
              </button>
              <h2 className="truncate text-sm font-semibold text-slate-900">{detail.title || detail.case_id}</h2>
            </div>
            <Pill label={STATUS[detail.status].text} colour={STATUS[detail.status].cls} />
          </div>
          {detail.status === 'open' && (
            <details className="group mt-3 rounded-xl bg-slate-50/80 ring-1 ring-slate-200" open={detail.reviewed === 0}>
              <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-xs font-medium text-slate-600">
                <svg {...stroke} className="size-4 text-indigo-500">
                  <circle cx="10" cy="10" r="7.25" />
                  <path d="M10 9v4.5M10 6.6v.1" />
                </svg>
                How to review this judgment
                <svg {...stroke} className="ml-auto size-4 text-slate-400 transition-transform duration-300 group-open:rotate-180">
                  <path d="m5 7.5 5 5 5-5" />
                </svg>
              </summary>
              <p className="px-3 pb-3 text-sm leading-relaxed text-slate-600">
                The computer has suggested a role for every sentence. Please check the{' '}
                <span className="font-medium text-red-700">Needs review</span> sentences (it wasn't sure) and the{' '}
                <span className="font-medium text-amber-800">Spot check</span> sentences (it was sure — we check a few to make
                sure it isn't confidently wrong). Press <Kbd>Enter</Kbd> if the suggestion is right, or a number key to pick the
                correct role. When every highlighted sentence is checked, click <b>Sign off</b>.
              </p>
            </details>
          )}
        </div>
      </div>

      {(result || detail.decision_note) && (
        <div className="px-4 pt-3">
          <div
            role="status"
            data-testid="signoff-result"
            className={`mx-auto flex max-w-4xl animate-rise items-start gap-3 rounded-2xl px-4 py-3 text-sm ring-1 ${
              detail.status === 'promoted' ? 'bg-emerald-50 text-emerald-800 ring-emerald-200' : 'bg-amber-50 text-amber-900 ring-amber-200'
            }`}
          >
            <span className={`grid size-7 shrink-0 animate-pop place-items-center rounded-full text-white ${detail.status === 'promoted' ? 'bg-emerald-500' : 'bg-amber-500'}`}>
              <svg {...stroke} strokeWidth={2} className="size-4">
                {detail.status === 'promoted' ? <path d="m4.5 10.5 3.5 3.5 7.5-8" /> : <path d="M10 5.5v5M10 14v.1" />}
              </svg>
            </span>
            <span className="pt-1">{result?.reason || detail.decision_note}</span>
          </div>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        <div className="mx-auto max-w-4xl">
          <label className="mb-3 flex w-fit cursor-pointer items-center gap-2 rounded-lg px-2 py-1 text-xs text-slate-500 hover:bg-slate-100">
            <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} className="size-3.5 accent-[var(--color-accent)]" />
            Also show the {detail.sentences - detail.flagged} sentences the computer labelled confidently (for context)
          </label>
          <ol className="space-y-1.5" data-testid="review-list">
            {rows.map((it) => {
              const active = it.sentence_id === cursor
              const label = it.final_label || it.predicted
              const kind = KIND[it.kind]
              return (
                <li
                  key={it.sentence_id}
                  ref={(el) => (rowRefs.current[it.sentence_id] = el)}
                  onClick={() => setCursor(it.sentence_id)}
                  data-testid="review-row"
                  className={`cursor-pointer rounded-xl px-3.5 py-2.5 text-sm ring-1 transition-[background-color,box-shadow] duration-200 ${
                    active
                      ? 'bg-indigo-50 shadow-sm ring-2 ring-indigo-300'
                      : it.kind === 'auto'
                        ? 'bg-slate-50 text-slate-500 ring-slate-200/60'
                        : 'bg-white ring-slate-200/70 hover:bg-slate-50'
                  }`}
                >
                  <div className="flex flex-wrap items-center gap-2">
                    {kind && <Pill label={kind.text} colour={kind.cls} />}
                    <span className="text-xs text-slate-500">
                      Suggested <Pill label={it.predicted} colour={colour[it.predicted]} /> · {pct(it.confidence)} sure
                    </span>
                    {it.status === 'accepted' && <span className="inline-flex animate-pop items-center gap-1 text-xs font-medium text-emerald-700">✓ Accepted</span>}
                    {it.status === 'corrected' && (
                      <span className="text-xs font-medium text-indigo-700">
                        → Corrected to <Pill label={label} colour={colour[label]} />
                      </span>
                    )}
                  </div>
                  <p className="mt-1 break-words leading-relaxed text-slate-800">{it.text}</p>
                  {active && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        setExplainFor((cur) => (cur === it.sentence_id ? null : it.sentence_id))
                      }}
                      className="mt-1.5 rounded-md px-1.5 py-0.5 text-xs font-medium text-indigo-600 hover:bg-indigo-100/60"
                      data-testid="explain-toggle"
                    >
                      {explainFor === it.sentence_id ? 'Hide explanation' : 'Why this suggestion? (w)'}
                    </button>
                  )}
                  {explainFor === it.sentence_id && (
                    <RoleExplanation caseId={caseId} item={it} labels={scheme.map((l) => l.name)} colour={colour} />
                  )}
                </li>
              )
            })}
          </ol>
        </div>
      </div>

      <div className="px-4 pb-4 pt-2">
        <div className="mx-auto max-w-4xl space-y-3 rounded-2xl bg-white p-3 shadow-[0_1px_2px_rgb(15_23_42/0.04),0_8px_24px_-12px_rgb(15_23_42/0.18)] ring-1 ring-slate-200">
          {current && current.kind !== 'auto' && !readOnly && (
            <div className="flex flex-wrap gap-1.5" role="toolbar" aria-label="Choose the role for the selected sentence">
              {scheme.map((l) => (
                <button
                  key={l.name}
                  title={l.description}
                  aria-label={`${l.name}${l.name === current.predicted ? " (suggested)" : ""}, key ${l.shortcut}`}
                  onClick={() => {
                    choose(current.sentence_id, l.name)
                  }}
                  className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm font-medium ring-1 transition-[transform,box-shadow] duration-200 hover:-translate-y-0.5 hover:shadow-sm ${colour[l.name]} ${
                    l.name === current.predicted ? 'outline outline-2 outline-offset-2 outline-indigo-500' : ''
                  }`}
                >
                  <kbd className="grid size-5 place-items-center rounded bg-white/60 font-mono text-[11px] ring-1 ring-current/20">{l.shortcut}</kbd>
                  {l.name}
                  {l.name === current.predicted && <span className="text-xs opacity-80">· suggested</span>}
                </button>
              ))}
            </div>
          )}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="min-w-0 flex-1 space-y-1.5 text-xs text-slate-600">
              <div className="flex flex-wrap items-center gap-3">
                <span className="font-semibold text-slate-800" data-testid="review-progress">
                  {detail.reviewed} of {detail.flagged} checked
                </span>
                <Meter value={detail.flagged ? detail.reviewed / detail.flagged : 0} tone={done ? 'bg-emerald-500' : 'bg-indigo-500'} className="w-32" />
                <span className={`inline-flex items-center gap-1.5 ${status.state === 'error' ? 'text-red-600' : status.state === 'saved' ? 'text-emerald-600' : 'text-slate-400'}`}>
                  {status.state === 'saving' && <span aria-hidden="true" className="size-3 animate-spin rounded-full border-2 border-current border-r-transparent" />}
                  {status.text}
                </span>
              </div>
              <p className={overLimit ? 'font-medium text-red-700' : ''}>
                Spot checks needing correction: {pct(detail.audit_error_rate)} (limit {pct(detail.max_audit_error)})
                {overLimit && <span className="block">Over the limit: signing off will send this judgment back for full manual annotation.</span>}
              </p>
            </div>
            {!readOnly && (
              <button
                onClick={signOff}
                disabled={!done}
                title={done ? 'Finish this judgment' : 'Check every highlighted sentence first'}
                aria-label="Sign off"
                className="inline-flex items-center gap-2 rounded-xl bg-accent px-5 py-2.5 text-sm font-semibold text-on-accent shadow-sm transition-[transform,background-color] duration-200 hover:-translate-y-0.5 hover:bg-accent-hover disabled:translate-y-0 disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none"
              >
                <svg {...stroke} strokeWidth={2} className="size-4">
                  <path d="m4.5 10.5 3.5 3.5 7.5-8" />
                </svg>
                Sign off
              </button>
            )}
          </div>
          <p className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-slate-500">
            <span>
              <Kbd>Enter</Kbd> accept
            </span>
            <span>
              <Kbd>1</Kbd>–<Kbd>{scheme.length}</Kbd> choose a role
            </span>
            <span>
              <Kbd>↑</Kbd>
              <Kbd>↓</Kbd> move
            </span>
            <span>
              <Kbd>n</Kbd> next unchecked
            </span>
            <span>
              <Kbd>w</Kbd> why this suggestion
            </span>
          </p>
        </div>
      </div>
    </div>
  )
}

export default function ReviewView() {
  const [reviewer, setReviewer] = useState(() => readStored('legalgpt.annotator'))
  const [draft, setDraft] = useState(reviewer)
  const [scheme, setScheme] = useState([])
  const [cases, setCases] = useState(null)
  const [open, setOpen] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    annotationApi.scheme().then(setScheme).catch((e) => setError(e.message))
  }, [])

  const loadCases = useCallback(() => {
    reviewApi.cases().then(setCases).catch((e) => setError(e.message))
  }, [])

  useEffect(() => {
    if (!open) loadCases()
  }, [open, loadCases])

  if (!reviewer) {
    return (
      <div className="relative isolate flex min-h-0 flex-1 items-start justify-center overflow-y-auto px-4 pt-16">
        <div aria-hidden="true" className="bg-dots pointer-events-none absolute inset-0 -z-10" />
        <form
          className="w-full max-w-sm animate-rise rounded-2xl bg-white p-6 shadow-[0_1px_2px_rgb(15_23_42/0.04),0_16px_40px_-20px_rgb(15_23_42/0.3)] ring-1 ring-slate-200"
          onSubmit={(e) => {
            e.preventDefault()
            if (!draft.trim()) return
            writeStored('legalgpt.annotator', draft.trim())
            setReviewer(draft.trim())
          }}
        >
          <span className="grid size-10 place-items-center rounded-xl bg-indigo-50 text-indigo-600 ring-1 ring-indigo-200">
            <svg {...stroke} className="size-5">
              <path d="m4.5 10.5 3.5 3.5 7.5-8" />
            </svg>
          </span>
          <h2 className="mt-4 text-lg font-semibold tracking-tight text-slate-900">Who is reviewing?</h2>
          <p className="mt-1 text-sm text-slate-500">Your name is saved with every check you make.</p>
          <input
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            maxLength={40}
            aria-label="Reviewer name"
            placeholder="e.g. priya"
            className="mt-5 w-full rounded-xl border border-slate-200 bg-slate-50/60 px-3.5 py-2.5 text-sm text-slate-900 transition-[border-color,box-shadow,background-color] focus:border-indigo-400 focus:bg-white focus:outline-none focus:ring-4 focus:ring-indigo-100"
          />
          <button
            type="submit"
            disabled={!draft.trim()}
            className="mt-3 w-full rounded-xl bg-accent px-4 py-2.5 text-sm font-semibold text-on-accent shadow-sm transition-[transform,background-color] duration-200 hover:-translate-y-0.5 hover:bg-accent-hover disabled:translate-y-0 disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none"
          >
            Start reviewing
          </button>
        </form>
      </div>
    )
  }

  if (open && scheme.length) {
    return <CaseReview caseId={open} reviewer={reviewer} scheme={scheme} onBack={() => setOpen(null)} />
  }

  const counts = (cases || []).reduce((t, c) => ({ ...t, [c.status]: (t[c.status] || 0) + 1 }), {})

  return (
    <div className="relative isolate min-h-0 flex-1 overflow-y-auto px-4 py-8 sm:py-10">
      <div aria-hidden="true" className="bg-dots pointer-events-none absolute inset-0 -z-10" />
      <div className="mx-auto max-w-4xl space-y-5">
        <header className="relative flex flex-wrap items-end justify-between gap-4">
          <div aria-hidden="true" className="pointer-events-none absolute -left-24 -top-24 -z-10 h-64 w-[30rem] max-w-[120%] animate-float rounded-full bg-[radial-gradient(closest-side,var(--color-indigo-200),transparent)] opacity-50 blur-2xl" />
          <HeaderArt>
            <GavelArt />
          </HeaderArt>
          <div>
            <p className="mb-3 inline-flex animate-rise items-center gap-1.5 rounded-full bg-white/70 px-2.5 py-1 text-xs font-medium text-slate-600 ring-1 ring-slate-200 backdrop-blur">
              <span aria-hidden="true" className="size-1.5 rounded-full bg-indigo-500" />
              Human-in-the-loop · InLegalBERT suggestions
            </p>
            <h2 className="animate-rise text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl" style={stagger(1)}>
              Judgments to review
            </h2>
            <p className="mt-1.5 max-w-xl animate-rise text-sm text-slate-500" style={stagger(2)}>
              The computer suggested roles for these judgments. You only need to check the sentences it highlights.
            </p>
          </div>
          <div className="flex animate-rise items-center gap-2 rounded-xl bg-white/80 py-1.5 pl-1.5 pr-3 ring-1 ring-slate-200 backdrop-blur" style={stagger(2)}>
            <span className="grid size-7 place-items-center rounded-lg bg-accent text-xs font-bold uppercase text-on-accent">{reviewer.slice(0, 1)}</span>
            <span className="text-sm">
              <span className="block text-[11px] leading-tight text-slate-500">Reviewing as</span>
              <span className="block font-medium leading-tight text-slate-900">{reviewer}</span>
            </span>
            <button
              className="ml-2 rounded-md px-1.5 py-0.5 text-xs font-medium text-indigo-600 hover:bg-indigo-50"
              onClick={() => {
                writeStored('legalgpt.annotator', '')
                setDraft('')
                setReviewer('')
              }}
            >
              Change
            </button>
          </div>
        </header>

        {cases && cases.length > 0 && (
          <div className="grid animate-rise gap-3 sm:grid-cols-3" style={stagger(3)}>
            {[
              ['To review', counts.open || 0, 'waiting for a check', 'bg-indigo-500'],
              ['Added to corpus', counts.promoted || 0, 'signed off', 'bg-emerald-500'],
              ['Sent to annotation', counts.escalated || 0, 'too many spot-check corrections', 'bg-amber-500'],
            ].map(([label, value, hint, dot]) => (
              <div key={label} className="rounded-2xl bg-white px-4 py-3 ring-1 ring-slate-200">
                <p className="flex items-center gap-1.5 text-xs text-slate-500">
                  <span aria-hidden="true" className={`size-1.5 rounded-full ${dot}`} />
                  {label}
                </p>
                <p className="mt-0.5 text-xl font-semibold tabular-nums tracking-tight text-slate-900">{value}</p>
                <p className="text-xs text-slate-400">{hint}</p>
              </div>
            ))}
          </div>
        )}

        <div className="animate-rise" style={stagger(4)}>
          <UploadPanel uploader={reviewer} onFinished={loadCases} onOpen={setOpen} />
        </div>
        {error && <p className="rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-200">{error}</p>}
        {cases === null ? (
          <div className="space-y-2">
            {Array.from({ length: 4 }, (_, i) => (
              <span key={i} className="block h-16 animate-pulse rounded-2xl bg-slate-100" />
            ))}
          </div>
        ) : cases.length === 0 ? (
          <div className="rounded-2xl bg-white px-5 py-8 text-center text-sm text-slate-600 ring-1 ring-slate-200">
            <p className="font-medium text-slate-800">Nothing to review yet</p>
            <p className="mt-1">Upload a judgment above to get started.</p>
          </div>
        ) : (
          <section className="space-y-3">
            <h3 className="text-sm font-semibold text-slate-800">
              All judgments <span className="font-normal text-slate-400">({cases.length})</span>
            </h3>
            <ul className="space-y-2" data-testid="review-case-list">
              {cases.map((c, i) => {
                const frac = c.flagged ? c.reviewed / c.flagged : 1
                return (
                  <li key={c.case_id} className="animate-rise" style={stagger(Math.min(i, 12), 40, 220)}>
                    <button
                      onClick={() => setOpen(c.case_id)}
                      className="group flex w-full flex-wrap items-center justify-between gap-3 rounded-2xl bg-white px-4 py-3.5 text-left ring-1 ring-slate-200 transition-[transform,box-shadow] duration-200 hover:-translate-y-0.5 hover:shadow-[0_10px_24px_-14px_rgb(15_23_42/0.35)] hover:ring-indigo-300"
                    >
                      <div className="flex min-w-0 items-center gap-3">
                        <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-slate-100 font-serif text-sm font-bold text-slate-500 transition-colors group-hover:bg-indigo-50 group-hover:text-indigo-600">§</span>
                        <div className="min-w-0">
                          <p className="truncate text-sm font-medium text-slate-900">{c.title || c.case_id}</p>
                          <p className="mt-0.5 text-xs text-slate-500">
                            {c.reviewed} of {c.flagged} highlighted sentences checked · {c.sentences} sentences in total
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <Meter value={frac} tone={c.status === 'escalated' ? 'bg-amber-500' : frac === 1 ? 'bg-emerald-500' : 'bg-indigo-500'} className="hidden w-20 sm:block" />
                        <Pill label={STATUS[c.status].text} colour={STATUS[c.status].cls} />
                        <svg {...stroke} className="size-4 text-slate-300 transition-[transform,color] duration-200 group-hover:translate-x-0.5 group-hover:text-indigo-500">
                          <path d="m7.5 5 5 5-5 5" />
                        </svg>
                      </div>
                    </button>
                  </li>
                )
              })}
            </ul>
          </section>
        )}
      </div>
    </div>
  )
}
