import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { annotationApi, reviewApi } from '../api.js'
import { Desk, JudgmentMap, Keys, Line, Pads, Saving, Stamp, Tally, marksOf } from './LabelDesk.jsx'
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

export const STATUS = {
  open: { text: 'To review', cls: 'bg-indigo-50 text-indigo-700 ring-indigo-200' },
  promoted: { text: 'Done — added to corpus', cls: 'bg-emerald-50 text-emerald-700 ring-emerald-200' },
  escalated: { text: 'Sent to full annotation', cls: 'bg-amber-50 text-amber-800 ring-amber-200' },
}

const pct = (x) => `${Math.round(x * 100)}%`

const stroke = { viewBox: '0 0 20 20', fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true }

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
export function UploadPanel({ uploader, onFinished, onOpen }) {
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
    <section data-testid="upload-panel">
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
        className={`flex flex-wrap items-center justify-between gap-x-6 gap-y-3 px-4 py-3 transition-colors duration-200 ${dragging ? 'bg-field-gold text-ink' : 'bg-field-panel text-paper'}`}
      >
        <p className="min-w-0 flex-1 basis-64 text-[15px] leading-snug">
          <b className="font-semibold">{dragging ? 'Drop to file these judgments' : 'File a new judgment'}</b>
          <span className={dragging ? '' : 'text-paper/75'}>. Drop a .txt or .pdf here. InLegalBERT marks it up first; nothing joins the corpus until someone signs it off.</span>
        </p>
        <button onClick={() => inputRef.current?.click()} className="press h-11 shrink-0 bg-accent px-5 text-[15px] font-medium text-on-accent hover:bg-accent-hover">
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
        <ul className="mt-2 space-y-1" data-testid="upload-list">
          {shown.map((j) => {
            const st = UPLOAD[j.status] || UPLOAD.failed
            return (
              <li key={j.job_id} className="flex animate-rise flex-wrap items-baseline gap-x-3 bg-field-panel px-4 py-2 text-[15px] text-paper" data-testid="upload-job" data-status={j.status}>
                <span className="font-medium">{j.filename}</span>
                <span className={st.busy ? 'text-field-gold' : 'text-paper/75'}>{st.text}</span>
                <span className="min-w-0 flex-1 text-[14px] text-paper/70">{j.message}</span>
                {j.status === 'done' && (
                  <button onClick={() => onOpen(j.case_id)} className="press font-medium text-field-gold underline underline-offset-2">Open</button>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}

export function CaseReview({ caseId, reviewer, scheme, onBack }) {
  const [detail, setDetail] = useState(null)
  const [showAll, setShowAll] = useState(false)
  const [cursor, setCursor] = useState(null) // sentence_id
  const [status, setStatus] = useState({ state: 'idle', text: '' })
  const [result, setResult] = useState(null)
  const [explainFor, setExplainFor] = useState(null) // sentence_id whose explanation is open
  const rowRefs = useRef({})
  const colour = useMemo(() => Object.fromEntries(scheme.map((l, i) => [l.name, PALETTE[i % PALETTE.length]])), [scheme])
  const marks = useMemo(() => marksOf(scheme), [scheme])
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
    if (cursor) rowRefs.current[cursor]?.scrollIntoView({ block: 'center', behavior: 'smooth' })
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
      <div className="grid min-h-0 flex-1 place-items-center bg-field-black px-4 text-paper">
        {status.state === 'error' ? <p className="text-[17px]">{status.text}</p> : <p className="text-shimmer text-lg font-medium">Opening the judgment…</p>}
      </div>
    )
  }

  const done = detail.reviewed === detail.flagged
  const overLimit = detail.audit_error_rate > detail.max_audit_error
  const current = detail.items.find((it) => it.sentence_id === cursor)

  const counts = detail.items.reduce((t, it) => {
    const l = it.final_label || it.predicted
    return { ...t, [l]: (t[l] || 0) + 1 }
  }, {})
  const settled = (it) => it.status === 'accepted' || it.status === 'corrected' || it.kind === 'auto'
  const finished = detail.status !== 'open'

  return (
    <Desk
      title={detail.title || detail.case_id}
      meta={finished ? STATUS[detail.status].text : `${detail.flagged} sentences to check. The rest were marked with confidence.`}
      onBack={onBack}
      backLabel="Back to all judgments"
      paper={
        <>
          {(result || detail.decision_note) && (
            <p role="status" data-testid="signoff-result" className="mt-4 flex animate-rise flex-wrap items-center gap-x-4 gap-y-2 text-[16px] leading-snug">
              <Stamp className={detail.status === 'promoted' ? 'text-emerald-700' : 'text-amber-700'}>{detail.status === 'promoted' ? 'Signed off' : 'Sent back'}</Stamp>
              <span className="min-w-0 flex-1 basis-64">{result?.reason || detail.decision_note}</span>
            </p>
          )}
          <label className="mt-4 flex w-fit cursor-pointer items-center gap-2 text-[15px] text-slate-700">
            <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} className="size-4 accent-[var(--color-ink)]" />
            Also show the {detail.sentences - detail.flagged} sentences marked with confidence
          </label>
          <ol className="mt-3" data-testid="review-list">
            {rows.map((it) => {
              const active = it.sentence_id === cursor
              const label = it.final_label || it.predicted
              const kind = KIND[it.kind]
              return (
                <Line
                  key={it.sentence_id}
                  n={it.index != null ? it.index + 1 : rows.indexOf(it) + 1}
                  active={active}
                  mark={settled(it) ? marks[label] : null}
                  dashed={settled(it) ? null : marks[it.predicted]}
                  onClick={() => setCursor(it.sentence_id)}
                  innerRef={(el) => (rowRefs.current[it.sentence_id] = el)}
                  testId="review-row"
                  tag={
                    <p className="mt-0.5 flex flex-wrap items-baseline gap-x-2.5 text-[13px] text-slate-600">
                      {kind && <span className={`font-semibold ${it.kind === 'mandatory' ? 'text-red-700' : 'text-amber-800'}`} title={kind.hint}>{kind.text}</span>}
                      <span>Suggested <b className={`font-semibold ${marks[it.predicted]?.ink || ''}`}>{it.predicted}</b>, {pct(it.confidence)} sure</span>
                      {it.status === 'accepted' && <span className="animate-stamp font-semibold text-emerald-700">Accepted</span>}
                      {it.status === 'corrected' && <span className="animate-stamp font-semibold text-indigo-700">Corrected to {label}</span>}
                    </p>
                  }
                  below={
                    <>
                      {active && (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation()
                            setExplainFor((cur) => (cur === it.sentence_id ? null : it.sentence_id))
                          }}
                          className="press mt-1 text-[14px] font-medium text-indigo-600 underline underline-offset-2"
                          data-testid="explain-toggle"
                        >
                          {explainFor === it.sentence_id ? 'Hide the explanation' : 'Why this suggestion?'}
                        </button>
                      )}
                      {explainFor === it.sentence_id && <RoleExplanation caseId={caseId} item={it} labels={scheme.map((l) => l.name)} colour={colour} />}
                    </>
                  }
                >
                  {it.text}
                </Line>
              )
            })}
          </ol>
        </>
      }
      bench={
        <>
          <div>
            <p className="text-[2.25rem] font-semibold leading-none tracking-[-0.04em] [font-variant-numeric:tabular-nums]" data-testid="review-progress">
              {detail.reviewed}
              <span className="text-[1.125rem] font-medium tracking-normal text-paper/70"> of {detail.flagged} checked</span>
            </p>
            <p className="mt-1 flex flex-wrap items-baseline justify-between gap-x-3 text-[14px] text-paper/75">
              {finished ? STATUS[detail.status].text : 'Accept it, or pick the right role'}
              <Saving status={status} />
            </p>
          </div>
          {current && current.kind !== 'auto' && !readOnly && (
            <>
              <button
                type="button"
                onClick={() => choose(current.sentence_id, current.predicted)}
                className="press flex h-12 items-center justify-between bg-accent px-4 text-[17px] font-medium text-on-accent hover:bg-accent-hover"
              >
                Accept {current.predicted}
                <kbd className="bg-ink/15 px-1.5 font-sans text-[13px]">Enter</kbd>
              </button>
              <Pads scheme={scheme} chosen={current.status === 'pending' ? null : current.final_label || current.predicted} suggested={current.predicted} onPick={(name) => choose(current.sentence_id, name)} label="Choose the role for the selected sentence" />
            </>
          )}
          <JudgmentMap
            title="Sentences to check"
            current={cursor}
            onJump={setCursor}
            cells={flagged.map((it, i) => ({ id: it.sentence_id, cell: settled(it) ? marks[it.final_label || it.predicted]?.cell : '', title: `To check ${i + 1}: ${it.status === 'pending' ? 'not checked yet' : it.final_label || it.predicted}` }))}
          />
          <Tally scheme={scheme} counts={counts} />
          <p className={`text-[14px] leading-snug ${overLimit ? 'font-medium text-field-gold' : 'text-paper/75'}`}>
            Spot checks needing correction: {pct(detail.audit_error_rate)}, limit {pct(detail.max_audit_error)}.
            {overLimit && ' Over the limit: signing off will send this judgment back for full manual labelling.'}
          </p>
          {!readOnly && (
            <button
              onClick={signOff}
              disabled={!done}
              title={done ? 'Finish this judgment' : 'Check every highlighted sentence first'}
              aria-label="Sign off"
              className="press h-12 shrink-0 bg-paper text-[17px] font-medium text-ink hover:bg-field-gold disabled:bg-field-panel disabled:text-paper/50"
            >
              Sign off
            </button>
          )}
          <Keys items={[['Enter', 'accept'], [`1–${scheme.length}`, 'choose a role'], ['↑ ↓', 'move'], ['n', 'next unchecked'], ['w', 'why']]} />
        </>
      }
    />
  )
}
