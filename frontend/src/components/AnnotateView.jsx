import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { annotationApi } from '../api.js'
import { Desk, JudgmentMap, Keys, Line, MARKS, Pads, Saving, Stamp, Tally, marksOf } from './LabelDesk.jsx'

/** The five roles and what each means, with the mark it leaves on the paper. */
export function Guideline({ scheme }) {
  return (
    <dl className="grid gap-x-8 gap-y-3 sm:grid-cols-2">
      {scheme.map((l, i) => (
        <div key={l.name}>
          <dt className="text-[1.0625rem] font-semibold tracking-[-0.01em]">
            <span className={`px-1 ${MARKS[i % MARKS.length].mark}`}>{l.name}</span>
          </dt>
          <dd className="mt-1 text-[15px] leading-snug text-slate-700">{l.description}</dd>
        </div>
      ))}
    </dl>
  )
}

export function CaseAnnotator({ caseId, annotator, scheme, onProgress, onBack }) {
  const [detail, setDetail] = useState(null)
  const [cursor, setCursor] = useState(0)
  const [status, setStatus] = useState({ state: 'idle', text: '' })
  const rowRefs = useRef([])
  const byShortcut = useMemo(() => Object.fromEntries(scheme.map((l) => [l.shortcut, l.name])), [scheme])
  const marks = useMemo(() => marksOf(scheme), [scheme])

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
    rowRefs.current[cursor]?.scrollIntoView({ block: 'center', behavior: 'smooth' })
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
      <div className="grid min-h-0 flex-1 place-items-center bg-field-black px-4 text-paper">
        {status.state === 'error' ? <p className="text-[17px]">{status.text}</p> : <p className="text-shimmer text-lg font-medium">Opening the judgment…</p>}
      </div>
    )
  }

  const total = detail.sentences.length
  const done = detail.progress.labelled === detail.progress.sentences
  const counts = detail.sentences.reduce((t, s) => (s.label ? { ...t, [s.label]: (t[s.label] || 0) + 1 } : t), {})
  const pick = (name) => {
    assign(cursor, name)
    setCursor((c) => Math.min(c + 1, total - 1))
  }

  return (
    <Desk
      title={detail.title || detail.case_id}
      meta={[detail.court, detail.decision_year, detail.sections_cited.length && `IPC ${detail.sections_cited.join(', ')}`].filter(Boolean).join(', ')}
      onBack={onBack}
      backLabel="Back to all cases"
      paper={
        <>
          {done && (
            <p className="mt-4 flex animate-rise flex-wrap items-center gap-x-4 gap-y-2 text-[16px] leading-snug">
              <Stamp className="text-emerald-700">All marked</Stamp>
              Every sentence is labelled and saved. You can still change any of them.
            </p>
          )}
          <ol className="mt-4" data-testid="sentence-list">
            {detail.sentences.map((s) => (
              <Line
                key={s.idx}
                n={s.idx + 1}
                active={s.idx === cursor}
                mark={s.label ? marks[s.label] : null}
                onClick={() => setCursor(s.idx)}
                innerRef={(el) => (rowRefs.current[s.idx] = el)}
                testId="sentence-row"
                tag={
                  s.label && (
                    <span key={s.label} className={`mt-0.5 inline-block animate-stamp text-[13px] font-semibold ${marks[s.label].ink}`} data-testid="sentence-label">
                      {s.label}
                    </span>
                  )
                }
              >
                {s.text}
              </Line>
            ))}
          </ol>
        </>
      }
      bench={
        <>
          <div>
            <p className="text-[2.25rem] font-semibold leading-none tracking-[-0.04em] [font-variant-numeric:tabular-nums]">
              {cursor + 1}
              <span className="text-[1.125rem] font-medium tracking-normal text-paper/70"> of {total}</span>
            </p>
            <p className="mt-1 flex flex-wrap items-baseline justify-between gap-x-3 text-[14px] text-paper/75">
              {detail.progress.labelled} marked
              <Saving status={status} />
            </p>
          </div>
          <Pads scheme={scheme} chosen={detail.sentences[cursor]?.label} onPick={pick} label="Assign label to the selected sentence" />
          <JudgmentMap
            title="The whole judgment"
            current={cursor}
            onJump={setCursor}
            cells={detail.sentences.map((s) => ({ id: s.idx, cell: s.label ? marks[s.label].cell : '', title: `Sentence ${s.idx + 1}${s.label ? `: ${s.label}` : ''}` }))}
          />
          <Tally scheme={scheme} counts={counts} />
          <Keys items={[[`1–${scheme.length}`, 'mark and move on'], ['↑ ↓', 'move'], ['n', 'next unmarked']]} />
        </>
      }
    />
  )
}
