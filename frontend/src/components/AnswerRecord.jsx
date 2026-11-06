import { useEffect, useRef, useState } from 'react'
import { ruleSectionsOnce } from '../api.js'
import { sendTo } from '../handoff.js'
import AnswerTrace from './AnswerTrace.jsx'
import ExplainPanel, { VERDICT } from './ExplainPanel.jsx'
import { WhatToCheck, HowChosen, SourceMatches } from './WhyPanel.jsx'
import { AnswerProof } from './ZkProof.jsx'
import { stagger } from './motion.jsx'

// What sits under an answer. One strip says the thing that matters, how far to rely on it, and always
// says so if something does not check out. Everything else is the record, in four tabs that stay shut
// until asked for: nothing is removed, it is only no longer in the way of reading the answer.

const ScalesMini = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="size-3.5 shrink-0">
    <path d="M12 3.5v16M8 20.5h8M4.5 6.5h15M4.5 6.5 2 13M4.5 6.5 7 13M1.8 13a2.7 2.7 0 0 0 5.4 0ZM19.5 6.5 17 13M19.5 6.5 22 13M16.8 13a2.7 2.7 0 0 0 5.4 0Z" />
  </svg>
)

const citedIn = (msg) => new Set([...(msg.content || '').matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])))
const shortName = (s) => (s.kind === 'judgment' ? s.title : s.section ? `${s.doc_id === 'BNS' ? 'BNS ' : ''}s.${s.section}` : s.citation_path)

/** How far to rely on the answer, in one plain sentence. */
function standing(msg) {
  const ex = msg.explanation
  if (msg.cutOff) return { tone: 'warn', text: 'This answer stopped mid-way, so do not rely on it. Ask again or choose another model.' }
  if (msg.refused) return { tone: 'none', text: 'No section matched closely enough, so it declined to answer.' }
  if (msg.retrieval?.translation?.failed) return { tone: 'warn', text: 'The question could not be translated, so the IPC was not searched.' }
  if (!ex) return { tone: 'none', text: 'Checking the answer against its sources…' }
  const claims = (ex.attributions || []).filter((a) => a.kind !== 'analysis')
  const missing = claims.filter((a) => !a.supported).length
  if (missing) return { tone: 'warn', text: `${missing} of ${claims.length} legal statements ${missing === 1 ? 'has' : 'have'} no source. Read ${missing === 1 ? 'it' : 'them'} before relying on this answer.` }
  if (ex.band === 'Low') return { tone: 'warn', text: 'Every legal statement has a source, but the sources only loosely match your question.' }
  if (!citedIn(msg).size) return { tone: 'warn', text: 'This answer cites no source, so it could not be checked. Do not rely on it.' }
  if (!claims.length) return { tone: 'none', text: 'Nothing here could be matched to a source word for word. Read it against the sections it cites.' }
  return { tone: 'good', text: claims.length === 1 ? 'The legal statement is backed by a source.' : `All ${claims.length} legal statements are backed by a source.` }
}

const TONE = { good: 'border-emerald-500', warn: 'border-amber-500', none: 'border-slate-300' }

function Sections({ msg, onCite }) {
  const cited = citedIn(msg)
  const list = msg.sources || []
  const row = (s, i) => (
    <li key={s.chunk_id} className="min-w-0 animate-rise border-b border-slate-200" style={stagger(i, 30)}>
      <button
        onClick={() => onCite(s)}
        title={`${cited.has(s.marker) ? 'Cited in the answer' : 'Retrieved but not cited'}: open the ${s.kind === 'judgment' ? 'judgment passage' : 'IPC text'}`}
        className={`group flex w-full min-w-0 items-baseline gap-2.5 py-2 text-left text-[15px] leading-snug ${cited.has(s.marker) ? 'text-slate-900' : 'text-slate-600'}`}
        data-testid="source-chip"
        data-kind={s.kind || 'statute'}
        data-cited={cited.has(s.marker)}
      >
        <span className={`w-4 shrink-0 text-right font-semibold ${cited.has(s.marker) ? 'text-indigo-600' : ''}`}>{s.marker}</span>
        {s.kind === 'judgment' ? (
          <span className="flex min-w-0 items-baseline gap-1.5 group-hover:underline">
            <span className="translate-y-0.5 self-center"><ScalesMini /></span>
            <span className="truncate">{s.title}</span>
            <span className="shrink-0 opacity-70">({[s.court?.includes('Supreme') ? 'SC' : '', s.year].filter(Boolean).join(', ')})</span>
          </span>
        ) : (
          <span className="min-w-0 group-hover:underline">
            {s.doc_id === 'BNS' && <span className="mr-1 font-semibold">BNS</span>}
            {s.section ? `s.${s.section}` : s.citation_path}
            {s.title ? ` ${s.title}` : ''}
            {s.ipc?.length > 0 && <span className="ml-2 text-[13px] text-slate-500" title="The IPC section this BNS section replaced">{s.ipc[0] === 'new' ? 'new in BNS' : `IPC ${s.ipc.slice(0, 4).join(', ')}`}</span>}
            {s.bns?.length > 0 && <span className="ml-2 text-[13px] text-slate-500" title="Bharatiya Nyaya Sanhita, 2023 equivalent">{s.bns[0] === 'none' ? 'no BNS' : `BNS ${s.bns.join(', ')}`}</span>}
          </span>
        )}
      </button>
    </li>
  )
  const used = list.filter((s) => cited.has(s.marker))
  const others = list.filter((s) => !cited.has(s.marker))
  return (
    <div className="space-y-4">
      <Correspondence msg={msg} cited={cited} />
      {used.length > 0 && <ul>{used.map(row)}</ul>}
      {others.length > 0 && (
        <Fold title={`${others.length} more found but not cited`} open={used.length === 0}>
          <ul>{others.map(row)}</ul>
        </Fold>
      )}
      {msg.explanation?.sources?.length > 0 && (
        <Fold title="How well each source matched">
          <SourceMatches explanation={msg.explanation} sources={list} onOpenSource={onCite} />
        </Fold>
      )}
      {msg.retrieval && (
        <Fold title="How the sources were chosen">
          <HowChosen r={msg.retrieval} />
        </Fold>
      )}
    </div>
  )
}

/** The cited IPC sections with their BNS numbers (BPRD correspondence table), or the reverse. */
function Correspondence({ msg, cited }) {
  if (msg.refused) return null
  const seen = new Set()
  const bnsRows = (msg.sources || []).filter((s) => s.doc_id === 'BNS' && s.ipc?.length && cited.has(s.marker) && !seen.has(s.section) && seen.add(s.section))
  if (bnsRows.length > 0)
    return (
      <p className="flex flex-wrap gap-x-3 gap-y-1 text-[15px] text-slate-700" data-testid="ipc-line">
        <span className="font-medium text-slate-900">Answered under the BNS; before 1 July 2024:</span>
        {bnsRows.map((s) => (
          <span key={s.section} className="whitespace-nowrap">s.{s.section} BNS ← {s.ipc[0] === 'new' ? <i>new provision</i> : <b className="font-semibold">s.{s.ipc.slice(0, 4).join(', ')} IPC</b>}</span>
        ))}
      </p>
    )
  const rows = (msg.sources || []).filter((s) => s.kind !== 'judgment' && s.section && s.bns?.length && cited.has(s.marker) && !seen.has(s.section) && seen.add(s.section))
  if (!rows.length) return null
  return (
    <p className="flex flex-wrap gap-x-3 gap-y-1 text-[15px] text-slate-700" data-testid="bns-line">
      <span className="font-medium text-slate-900">Under the BNS (since 1 July 2024):</span>
      {rows.map((s) => (
        <span key={s.section} className="whitespace-nowrap">s.{s.section} IPC → {s.bns[0] === 'none' ? <i>no direct equivalent</i> : <b className="font-semibold">s.{s.bns.join(', ')} BNS</b>}</span>
      ))}
    </p>
  )
}

export function Fold({ title, open = false, children }) {
  return (
    <details className="group text-[15px]" open={open}>
      <summary className="press inline-flex cursor-pointer select-none items-center gap-2 font-medium text-slate-800 hover:text-slate-950">
        <span aria-hidden="true" className="text-slate-500 transition-transform duration-200 group-open:rotate-90">›</span>
        {title}
      </summary>
      <div className="mt-2">{children}</div>
    </details>
  )
}

export default function AnswerRecord({ msg, live, onCite, onUpdate }) {
  const [tab, setTab] = useState(null)
  const [rules, setRules] = useState([])
  const panels = useRef(null)
  // a tab opens below the fold of a long answer: bring it up
  useEffect(() => {
    if (tab) requestAnimationFrame(() => panels.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
  }, [tab])
  useEffect(() => {
    let alive = true
    ruleSectionsOnce().then((r) => alive && setRules(r))
    return () => {
      alive = false
    }
  }, [])

  if (msg.mode !== 'legal' || msg.error || live || !msg.content) return null

  const cited = citedIn(msg)
  const statutes = (msg.sources || []).filter((s) => s.kind !== 'judgment')
  const used = (msg.sources || []).filter((s) => cited.has(s.marker))
  const { tone, text } = standing(msg)
  const ex = msg.explanation
  const toCheck = ex ? (ex.attributions || []).filter((a) => (!a.supported && a.kind !== 'analysis') || (a.supported && a.cited_supported === false)).length : 0
  // the first cited IPC section the rule engine encodes: its facts can be tested there
  const ruleSection = (s) => (s.doc_id === 'BNS' ? s.ipc?.find((x) => rules.includes(x)) : rules.includes(s.section) ? s.section : null)
  const checkable = statutes.find((s) => cited.has(s.marker) && ruleSection(s))
  const checkSection = checkable && ruleSection(checkable)
  const canExplain = onUpdate && msg.content && msg.question
  const bias = msg.explain?.sensitive && msg.explain?.[msg.explain.sensitiveAttr === 'gender' ? 'bias' : `bias_${msg.explain.sensitiveAttr}`]
  const biasWords = bias?.verdict && VERDICT[bias.verdict]?.[1]

  const tabs = [
    ['sources', `Sources${msg.sources?.length ? ` (${msg.sources.length})` : ''}`],
    ex && ['check', toCheck ? `What to check (${toCheck})` : 'What to check'],
    canExplain && ['decided', 'How it decided'],
    ['proof', 'How it was made'],
  ].filter(Boolean)

  return (
    <div className="mt-6 animate-rise" data-testid="answer-record">
      <div className={`border-l-4 bg-white py-3 pl-4 pr-3 ${TONE[tone]}`}>
        <div className="flex flex-wrap items-start justify-between gap-x-5 gap-y-2">
          <p className="min-w-0 flex-1 basis-72 text-[16px] font-medium leading-snug text-slate-900" data-testid="answer-standing">{text}</p>
          {checkable && (
            <button
              type="button"
              onClick={() => sendTo('verify', { text: msg.question, section: checkSection })}
              title={`Open Verify with your question and IPC s.${checkSection}: the facts are tested against its elements`}
              className="press shrink-0 text-[15px] font-medium text-indigo-600 underline underline-offset-4 hover:text-indigo-800"
            >
              Test the facts against s.{checkSection}
            </button>
          )}
        </div>
        {used.length > 0 && (
          <p className="mt-2 flex flex-wrap items-center gap-1.5 text-[14px]">
            <span className="text-slate-600">Rests on</span>
            {used.map((s) => (
              <button
                key={s.chunk_id}
                onClick={() => onCite(s)}
                title={s.title || s.citation_path}
                className="press max-w-[16rem] truncate bg-slate-100 px-2 py-0.5 font-medium text-slate-800 hover:bg-slate-200"
                data-testid="cited-section"
              >
                {shortName(s)}
              </button>
            ))}
          </p>
        )}
        {msg.retrieval?.background_ignored?.length > 0 && (
          <p className="mt-2 text-[14px] leading-snug text-slate-600">Left out of the search, because the law does not turn on it: {msg.retrieval.background_ignored.join(' · ')}.</p>
        )}
        {msg.explain?.sensitive && (
          <p className="mt-2 text-[14px] leading-snug text-slate-600">
            This touches on {msg.explain.sensitiveAttr === 'caste' ? 'caste' : 'gender'}, so a bias check runs by itself{biasWords ? `: ${biasWords.toLowerCase()}.` : '.'}{' '}
            <button type="button" onClick={() => setTab('decided')} className="press font-medium text-slate-900 underline underline-offset-2">See it</button>
          </p>
        )}
      </div>

      <div ref={panels} role="tablist" aria-label="The record of this answer" className="mt-1 flex scroll-mt-2 flex-wrap gap-x-1">
        {tabs.map(([id, label]) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={tab === id}
            onClick={() => setTab((t) => (t === id ? null : id))}
            className={`press border-b-2 px-2.5 py-2 text-[15px] font-medium transition-colors duration-150 ${
              tab === id ? 'border-indigo-500 text-slate-950' : 'border-transparent text-slate-600 hover:text-slate-950'
            }`}
            data-testid={`record-${id}`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* the panels stay mounted, so the automatic bias check runs and each keeps its results */}
      <div hidden={tab !== 'sources'} className="animate-rise pt-3" style={{ animationDuration: '280ms' }}><Sections msg={msg} onCite={onCite} /></div>
      {ex && (
        <div hidden={tab !== 'check'} className="animate-rise pt-3" style={{ animationDuration: '280ms' }}>
          <WhatToCheck explanation={ex} sources={msg.sources} onOpenSource={onCite} />
        </div>
      )}
      {canExplain && (
        <div hidden={tab !== 'decided'} className="animate-rise pt-3" style={{ animationDuration: '280ms' }}>
          <ExplainPanel msg={msg} onUpdate={onUpdate} onCite={onCite} embedded active={tab === 'decided'} />
        </div>
      )}
      <div hidden={tab !== 'proof'} className="animate-rise space-y-3 pt-3" style={{ animationDuration: '280ms' }}>
        <AnswerTrace msg={msg} onCite={onCite} goTo={setTab} />
        <AnswerProof msg={msg} embedded />
      </div>
    </div>
  )
}
