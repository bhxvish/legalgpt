import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { annotationApi, reviewApi } from '../api.js'

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

function Pill({ label, colour }) {
  return <span className={`rounded px-1.5 py-0.5 text-xs font-medium ring-1 ${colour}`}>{label}</span>
}

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
    <section className="rounded-lg bg-white p-4 ring-1 ring-slate-200" data-testid="upload-panel">
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
        className={`flex flex-wrap items-center gap-3 rounded-lg border-2 border-dashed px-4 py-3 ${
          dragging ? 'border-indigo-400 bg-indigo-50' : 'border-slate-200'
        }`}
      >
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-slate-800">Add judgments</p>
          <p className="text-xs text-slate-500">
            Drop .txt or .pdf files here. Each one is checked, annotated by the computer, and added below for you to
            review; nothing enters the corpus until someone signs it off.
          </p>
        </div>
        <button
          onClick={() => inputRef.current?.click()}
          className="rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-700"
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
              <li key={j.job_id} className="flex flex-wrap items-center gap-2 text-sm" data-testid="upload-job" data-status={j.status}>
                <Pill label={st.text} colour={st.cls} />
                <span className="font-medium text-slate-800">{j.filename}</span>
                <span className={`min-w-0 flex-1 text-xs ${st.busy ? 'animate-pulse text-slate-500' : 'text-slate-600'}`}>
                  {j.message}
                </span>
                {j.status === 'done' && (
                  <button onClick={() => onOpen(j.case_id)} className="text-xs font-medium text-indigo-600 hover:underline">
                    Open
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
    return <p className="p-6 text-sm text-slate-500">{status.state === 'error' ? status.text : 'Loading…'}</p>
  }

  const done = detail.reviewed === detail.flagged
  const overLimit = detail.audit_error_rate > detail.max_audit_error
  const current = detail.items.find((it) => it.sentence_id === cursor)

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="border-b border-slate-200 bg-white px-4 py-3">
        <div className="mx-auto max-w-4xl">
          <button onClick={onBack} className="text-xs text-indigo-600 hover:underline">
            ← All judgments
          </button>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="truncate text-sm font-semibold text-slate-900">{detail.title || detail.case_id}</h2>
            <Pill label={STATUS[detail.status].text} colour={STATUS[detail.status].cls} />
          </div>
          {detail.status === 'open' && (
            <p className="mt-2 text-sm text-slate-600">
              The computer has suggested a role for every sentence. Please check the{' '}
              <span className="font-medium text-red-700">Needs review</span> sentences (it wasn't sure) and the{' '}
              <span className="font-medium text-amber-800">Spot check</span> sentences (it was sure — we check a few to make
              sure it isn't confidently wrong). Press <kbd className="rounded bg-slate-100 px-1 font-mono text-xs">Enter</kbd> if
              the suggestion is right, or a number key to pick the correct role. When every highlighted sentence is checked,
              click <b>Sign off</b>.
            </p>
          )}
        </div>
      </div>

      {(result || detail.decision_note) && (
        <div className="px-4 pt-3">
          <div
            role="status"
            data-testid="signoff-result"
            className={`mx-auto max-w-4xl rounded-lg px-4 py-3 text-sm ring-1 ${
              detail.status === 'promoted' ? 'bg-emerald-50 text-emerald-800 ring-emerald-200' : 'bg-amber-50 text-amber-900 ring-amber-200'
            }`}
          >
            {result?.reason || detail.decision_note}
          </div>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        <div className="mx-auto max-w-4xl">
          <label className="mb-2 flex items-center gap-2 text-xs text-slate-500">
            <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />
            Also show the {detail.sentences - detail.flagged} sentences the computer labelled confidently (for context)
          </label>
          <ol className="space-y-1" data-testid="review-list">
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
                  className={`cursor-pointer rounded-md px-3 py-2 text-sm ${
                    active ? 'bg-indigo-50 ring-2 ring-indigo-300' : it.kind === 'auto' ? 'bg-slate-50 text-slate-500' : 'bg-white hover:bg-slate-50'
                  }`}
                >
                  <div className="flex flex-wrap items-center gap-2">
                    {kind && <Pill label={kind.text} colour={kind.cls} />}
                    <span className="text-xs text-slate-500">
                      Suggested <Pill label={it.predicted} colour={colour[it.predicted]} /> · {pct(it.confidence)} sure
                    </span>
                    {it.status === 'accepted' && <span className="text-xs font-medium text-emerald-700">✓ Accepted</span>}
                    {it.status === 'corrected' && (
                      <span className="text-xs font-medium text-indigo-700">
                        → Corrected to <Pill label={label} colour={colour[label]} />
                      </span>
                    )}
                  </div>
                  <p className="mt-1 break-words leading-relaxed text-slate-800">{it.text}</p>
                </li>
              )
            })}
          </ol>
        </div>
      </div>

      <div className="border-t border-slate-200 bg-white px-4 py-3">
        <div className="mx-auto max-w-4xl space-y-2">
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
                  className={`rounded-md px-2.5 py-1.5 text-sm font-medium ring-1 hover:brightness-95 ${colour[l.name]} ${
                    l.name === current.predicted ? 'outline outline-2 outline-offset-1 outline-indigo-500' : ''
                  }`}
                >
                  <kbd className="mr-1 font-mono text-xs opacity-70">{l.shortcut}</kbd>
                  {l.name}
                  {l.name === current.predicted && <span className="ml-1 text-xs">(suggested · Enter)</span>}
                </button>
              ))}
            </div>
          )}
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="text-xs text-slate-600">
              <span className="font-medium" data-testid="review-progress">
                {detail.reviewed} of {detail.flagged} checked
              </span>
              {' · '}
              <span className={overLimit ? 'font-medium text-red-700' : ''}>
                spot checks needing correction: {pct(detail.audit_error_rate)} (limit {pct(detail.max_audit_error)})
              </span>
              {overLimit && <span className="block text-red-700">Over the limit: signing off will send this judgment back for full manual annotation.</span>}
              <span className={`ml-2 ${status.state === 'error' ? 'text-red-600' : 'text-slate-400'}`}>{status.text}</span>
            </div>
            {!readOnly && (
              <button
                onClick={signOff}
                disabled={!done}
                title={done ? 'Finish this judgment' : 'Check every highlighted sentence first'}
                aria-label="Sign off"
                className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-40"
              >
                Sign off
              </button>
            )}
          </div>
          <p className="text-xs text-slate-400">Enter accept · 1–{scheme.length} choose a role · ↑/↓ move · n next unchecked · hover a role for its meaning</p>
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
      <div className="min-h-0 flex-1 overflow-y-auto px-4">
      <form
        className="mx-auto mt-12 w-full max-w-sm rounded-xl bg-white p-6 ring-1 ring-slate-200"
        onSubmit={(e) => {
          e.preventDefault()
          if (!draft.trim()) return
          writeStored('legalgpt.annotator', draft.trim())
          setReviewer(draft.trim())
        }}
      >
        <h2 className="font-semibold text-slate-900">Who is reviewing?</h2>
        <p className="mt-1 text-sm text-slate-500">Your name is saved with every check you make.</p>
        <input
          autoFocus
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          maxLength={40}
          aria-label="Reviewer name"
          placeholder="e.g. priya"
          className="mt-4 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
        />
        <button type="submit" disabled={!draft.trim()} className="mt-3 w-full rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40">
          Start reviewing
        </button>
      </form>
      </div>
    )
  }

  if (open && scheme.length) {
    return <CaseReview caseId={open} reviewer={reviewer} scheme={scheme} onBack={() => setOpen(null)} />
  }

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6">
      <div className="mx-auto max-w-4xl space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-lg font-semibold text-slate-900">Judgments to review</h2>
          <p className="text-sm text-slate-500">
            Reviewing as <span className="font-medium text-slate-800">{reviewer}</span>{' '}
            <button
              className="text-indigo-600 hover:underline"
              onClick={() => {
                writeStored('legalgpt.annotator', '')
                setDraft('')
                setReviewer('')
              }}
            >
              change
            </button>
          </p>
        </div>
        <p className="text-sm text-slate-600">
          The computer suggested roles for these judgments. You only need to check the sentences it highlights.
        </p>
        <UploadPanel uploader={reviewer} onFinished={loadCases} onOpen={setOpen} />
        {error && <p className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-200">{error}</p>}
        {cases === null ? (
          <p className="text-sm text-slate-500">Loading…</p>
        ) : cases.length === 0 ? (
          <p className="rounded-lg bg-white px-4 py-6 text-sm text-slate-600 ring-1 ring-slate-200">
            Nothing to review yet. Upload a judgment above to get started.
          </p>
        ) : (
          <ul className="divide-y divide-slate-100 rounded-lg bg-white ring-1 ring-slate-200" data-testid="review-case-list">
            {cases.map((c) => (
              <li key={c.case_id}>
                <button onClick={() => setOpen(c.case_id)} className="flex w-full flex-wrap items-center justify-between gap-2 px-4 py-3 text-left hover:bg-slate-50">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium text-slate-900">{c.title || c.case_id}</p>
                    <p className="text-xs text-slate-500">
                      {c.reviewed} of {c.flagged} highlighted sentences checked · {c.sentences} sentences in total
                    </p>
                  </div>
                  <Pill label={STATUS[c.status].text} colour={STATUS[c.status].cls} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
