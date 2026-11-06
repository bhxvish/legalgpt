import { useEffect, useRef, useState } from 'react'
import { caseApi, explainApi } from '../api.js'
import { ATTRIBUTES, Badge, BiasResult } from './ExplainPanel.jsx'
import { AnswerText } from './ChatWindow.jsx'
import { RedTape, Stamp } from './LegalArt.jsx'
import Select from './Select.jsx'
import { stagger } from './motion.jsx'
import SourceModal from './SourceModal.jsx'
import { VerificationResultView } from './VerifyPanel.jsx'
import WhyPanel from './WhyPanel.jsx'

const stroke = { viewBox: '0 0 20 20', fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true }

const STEPS = [
  // module: the part of the system that does the step (System overview, the paper's module numbers)
  { id: 'facts', label: 'Reading the facts', module: 'LLM', title: 'Facts with quotes, each checked against the file' },
  { id: 'law', label: 'Finding the law', module: 'M0', title: 'IPC retrieval; the section must exist and its title match' },
  { id: 'elements', label: 'Checking each element', module: 'M5', title: 'The Prolog rule engine for encoded sections; the model otherwise' },
  { id: 'similar', label: 'Similar judgments', module: 'M2', title: 'Facts passages of the role-labelled judgment library' },
  { id: 'analysis', label: 'Writing the analysis', module: 'M4', title: 'Cited analysis with its explanation, through the output guard' },
]

const ELEMENT = {
  satisfied: { label: 'shown', cls: 'bg-emerald-100 text-emerald-700 ring-emerald-200', path: 'm4.5 10.5 3.5 3.5 7.5-8' },
  violated: { label: 'contradicted', cls: 'bg-red-100 text-red-700 ring-red-200', path: 'm5.5 5.5 9 9M14.5 5.5l-9 9' },
  missing: { label: 'not established', cls: 'bg-amber-100 text-amber-700 ring-amber-200', path: 'M7.5 7.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .8-1 1.5v.4M10 14.6v.1' },
}

const VERDICT_WORD = {
  CONSISTENT: { text: 'every element shown', cls: 'bg-emerald-50 text-emerald-800 ring-emerald-200' },
  INSUFFICIENT: { text: 'some elements not established', cls: 'bg-amber-50 text-amber-800 ring-amber-200' },
  INCONSISTENT: { text: 'an element is contradicted', cls: 'bg-red-50 text-red-800 ring-red-200' },
}

function readFileBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader()
    r.onload = () => resolve(String(r.result).split(',', 2)[1] || '')
    r.onerror = () => reject(r.error)
    r.readAsDataURL(file)
  })
}

function Section({ n, title, hint, children, delay = 0 }) {
  return (
    <section className="animate-rise border-t border-rule pt-4" style={stagger(delay)}>
      <div className="mb-4 flex items-baseline gap-3">
        <span className="shrink-0 text-[1.3rem] font-semibold leading-tight text-indigo-600">{n}</span>
        <div>
          <h3 className="text-[1.3rem] leading-tight text-slate-900">{title}</h3>
          {hint && <p className="mt-0.5 text-xs text-slate-500">{hint}</p>}
        </div>
      </div>
      {children}
    </section>
  )
}

function Stepper({ steps }) {
  return (
    <ol className="grid gap-2 sm:grid-cols-5" aria-label="Progress">
      {STEPS.map((s) => {
        const st = steps[s.id] || 'waiting'
        return (
          <li key={s.id} title={s.title} className={`flex items-center gap-2 rounded-xl px-3 py-2 text-xs ring-1 transition-colors duration-300 ${
            st === 'done' ? 'bg-emerald-50 text-emerald-800 ring-emerald-200' : st === 'running' ? 'bg-indigo-50 text-indigo-800 ring-indigo-200' : 'bg-white text-slate-400 ring-slate-200'
          }`}>
            {st === 'done' ? (
              <svg {...stroke} strokeWidth={2} className="size-4 shrink-0"><path d="m4.5 10.5 3.5 3.5 7.5-8" /></svg>
            ) : st === 'running' ? (
              <span aria-hidden="true" className="size-3.5 shrink-0 animate-spin rounded-full border-2 border-current border-r-transparent" />
            ) : (
              <span aria-hidden="true" className="size-3.5 shrink-0 rounded-full border-2 border-current opacity-50" />
            )}
            <span className={st === 'running' ? 'text-shimmer font-medium' : 'font-medium'}>{s.label}</span>
            <span className="ml-auto font-mono text-[9.5px] font-semibold opacity-60">{s.module}</span>
          </li>
        )
      })}
    </ol>
  )
}

function SourceChip({ s, onOpen }) {
  const judgment = s.kind === 'judgment'
  return (
    <button
      onClick={() => onOpen(s)}
      className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-left text-xs ring-1 transition-[transform] duration-200 hover:-translate-y-px ${
        judgment ? 'bg-amber-50 text-amber-900 ring-amber-200' : 'bg-indigo-50 text-indigo-900 ring-indigo-200'
      }`}
    >
      <span className="font-semibold">[{s.marker}]</span>
      {judgment ? (
        <span className="max-w-[16rem] truncate">{s.title}</span>
      ) : (
        <span>
          §{s.section} {s.title}
          {s.bns?.length > 0 && <span className="ml-1 font-mono text-[10px] opacity-70">{s.bns[0] === 'none' ? '· no BNS' : `· BNS ${s.bns.join(', ')}`}</span>}
        </span>
      )}
    </button>
  )
}

function ModelCheck({ c, facts }) {
  const byId = Object.fromEntries((facts || []).map((f) => [f.id, f]))
  return (
    <ol className="space-y-2">
      {c.elements.map((e, i) => {
        const st = ELEMENT[e.status] || ELEMENT.missing
        return (
          <li key={i} className="flex animate-rise gap-3 rounded-xl bg-slate-50/60 p-3 ring-1 ring-slate-200" style={stagger(i, 70)}>
            <span className={`mt-0.5 grid size-6 shrink-0 place-items-center rounded-full ring-1 ${st.cls}`} title={st.label}>
              <svg {...stroke} strokeWidth={2} className="size-3.5"><path d={st.path} /></svg>
            </span>
            <div className="min-w-0 text-sm">
              <p className="text-slate-800">
                {e.element} <span className="text-xs font-medium text-slate-500">· {st.label}</span>
              </p>
              {e.explanation && <p className="mt-0.5 text-xs text-slate-600">{e.explanation}</p>}
              {e.facts?.length > 0 && (
                <p className="mt-1 flex flex-wrap gap-1">
                  {e.facts.map((id) => (
                    <span key={id} title={byId[id]?.quote} className="rounded bg-white px-1.5 py-px font-mono text-[11px] text-slate-600 ring-1 ring-slate-200">
                      {id}
                    </span>
                  ))}
                </p>
              )}
            </div>
          </li>
        )
      })}
      {c.elements.length === 0 && <p className="text-sm text-slate-500">{c.note || 'The model returned no elements for this section.'}</p>}
    </ol>
  )
}

const FOLLOW_UPS = [
  'Which facts still need to be proved?',
  'Which element is weakest on these facts?',
  'How did the similar judgments treat facts like these?',
]

// What the server needs to answer a follow-up: it kept nothing, so the analysis goes back with the question.
function analysisContext(run) {
  return {
    summary: run.facts?.summary || '',
    facts: (run.facts?.facts || []).map(({ id, fact, verified }) => ({ id, fact, verified })),
    offences: run.facts?.offences || [],
    checks: (run.elements?.checks || []).map((c) => ({
      section: c.section, title: c.title, status: c.status || '',
      elements: (c.elements || []).map((e) => ({ element: e.element, status: e.status })),
    })),
    analysis: run.answer || '',
    sources: run.sources || [],
  }
}

// Follow-up questions about the finished analysis, answered from its sources (plus any IPC section the
// question needs) and under the same rule as the analysis: what the law requires, never a verdict.
function AskAboutAnalysis({ run, model, localOnly, turns, setTurns, onOpenSource }) {
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const abortRef = useRef(null)
  const endRef = useRef(null)

  const ask = async (q) => {
    q = q.trim()
    if (!q || busy) return
    setQuestion('')
    setBusy(true)
    const history = turns.filter((t) => t.answer && !t.error).slice(-4)
      .flatMap((t) => [{ role: 'user', content: t.question }, { role: 'assistant', content: t.answer.slice(0, 7000) }])
    const i = turns.length
    setTurns((ts) => [...ts, { question: q, answer: '' }])
    const update = (fn) => setTurns((ts) => ts.map((t, j) => (j === i ? fn({ ...t }) : t)))
    requestAnimationFrame(() => endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }))
    const controller = new AbortController()
    abortRef.current = controller
    try {
      await caseApi.ask({ question: q, context: analysisContext(run), history, model, local_only: localOnly }, (event, data) => {
        if (event === 'token') update((t) => ({ ...t, answer: t.answer + data.text }))
        else if (event === 'answer_replace') update((t) => ({ ...t, answer: data.text }))
        else if (event === 'guard') update((t) => ({ ...t, guard: data.message }))
        else if (event === 'error') update((t) => ({ ...t, error: String(data?.message || data) }))
        else update((t) => ({ ...t, [event]: data }))
      }, controller.signal)
    } catch (e) {
      if (e.name !== 'AbortError') update((t) => ({ ...t, error: e.message.replace(/^HTTP \d+: /, '') }))
    } finally {
      setBusy(false)
      abortRef.current = null
    }
  }

  return (
    <div className="space-y-4">
      {turns.map((t, i) => (
        <div key={i} className="space-y-2.5">
          <p className="text-[1.25rem] font-semibold leading-snug tracking-[-0.02em] text-slate-900">{t.question}</p>
          <div className="rounded-xl bg-slate-50/70 px-4 py-3 ring-1 ring-slate-200">
            {t.error ? (
              <p className="text-sm text-red-700">{t.error}</p>
            ) : (
              <div className="text-[0.9375rem] leading-relaxed text-slate-800">
                <AnswerText text={t.answer} sources={t.sources || run.sources} onCite={onOpenSource} caret={busy && i === turns.length - 1} />
              </div>
            )}
            {t.guard && <p className="mt-2 text-xs text-red-700">{t.guard}</p>}
            {(() => {
              // sections retrieved for this question beyond the analysis's own sources: shown only if the answer cites them
              const cited = new Set([...(t.answer || '').matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])))
              const extra = (t.sources || []).slice(run.sources?.length || 0).filter((s) => cited.has(s.marker))
              return extra.length > 0 && (
                <div className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-slate-200 pt-2.5">
                  <span className="text-[11px] text-slate-500">Also used for this question:</span>
                  {extra.map((s) => <SourceChip key={s.chunk_id + s.marker} s={s} onOpen={onOpenSource} />)}
                </div>
              )
            })()}
            {t.done?.model && <p className="mt-2 text-[11px] text-slate-400">Model: {t.done.model}{t.done.model_note ? ` · ${t.done.model_note}` : ''}</p>}
          </div>
        </div>
      ))}
      {turns.length === 0 && (
        <div className="flex flex-wrap gap-2">
          {FOLLOW_UPS.map((q) => (
            <button key={q} onClick={() => ask(q)} className="rounded-xl bg-white px-3 py-1.5 text-left text-sm text-slate-700 ring-1 ring-slate-200 transition-colors hover:bg-slate-50 hover:text-slate-900">
              {q}
            </button>
          ))}
        </div>
      )}
      <form
        ref={endRef}
        onSubmit={(e) => {
          e.preventDefault()
          ask(question)
        }}
        className="flex items-end gap-2"
      >
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              ask(question)
            }
          }}
          rows={1}
          maxLength={2000}
          placeholder="Ask about this analysis, e.g. what evidence would establish intent?"
          aria-label="Question about this analysis"
          className="block min-h-[2.75rem] w-full resize-y rounded-xl border border-slate-200 bg-white px-3.5 py-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:border-indigo-400 focus:outline-none focus:ring-4 focus:ring-indigo-100"
        />
        {busy ? (
          <button type="button" onClick={() => abortRef.current?.abort()} className="shrink-0 rounded-xl bg-slate-200 px-4 py-2.5 text-sm font-semibold text-slate-800 hover:bg-slate-300">
            Stop
          </button>
        ) : (
          <button type="submit" disabled={!question.trim()} className="shrink-0 rounded-xl bg-accent px-4 py-2.5 text-sm font-semibold text-on-accent hover:bg-accent-hover disabled:bg-slate-200 disabled:text-slate-400">
            Ask
          </button>
        )}
      </form>
      <p className="text-[11px] leading-relaxed text-slate-500">
        Answers use this analysis and its sources only. Like the analysis, they explain what the law requires and what the file
        shows; they do not say whether anyone is guilty.
      </p>
    </div>
  )
}

// Bias check for a finished analysis: the same file with one attribute (gender, caste, age...) changed, through facts,
// offences and element checks, compared with this analysis and with the unchanged file run again.
// Runs by itself when the file is gender-sensitive (dowry, cruelty, sexual offences...).
function CaseBias({ run, request, result, setResult }) {
  const [attribute, setAttribute] = useState('gender')
  const [busy, setBusy] = useState(false)
  const [sensitive, setSensitive] = useState(undefined)
  const started = useRef(false)
  const go = async (attr) => {
    setBusy(true)
    try {
      const r = await explainApi.caseBias({
        ...request, attribute: attr, control: true,
        original: { offences: run.facts?.offences || [], checks: (run.elements?.checks || []).map((c) => ({ section: c.section, status: c.status })), analysis: run.answer || '' },
      })
      setResult((x) => ({ ...x, [attr]: r }))
    } catch (e) {
      setResult((x) => ({ ...x, [attr]: { error: e.message.replace(/^HTTP \d+: /, '') } }))
    } finally {
      setBusy(false)
    }
  }
  useEffect(() => {
    if (started.current) return
    started.current = true
    const text = [run.facts?.summary, ...(run.facts?.facts || []).map((f) => f.fact)].join(' ')
    explainApi
      .sensitive({ text, sections: (run.facts?.offences || []).map((o) => o.section) })
      .then((r) => {
        setSensitive(r.reason || null)
        const attr = r.attribute || 'gender'
        if (r.reason && !result?.[attr]) {
          setAttribute(attr)
          go(attr)
        }
      })
      .catch(() => setSensitive(null))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  const data = result?.[attribute]
  return (
    <div className="space-y-3 text-xs">
      {sensitive && (
        <p className="rounded-md bg-amber-50 px-2.5 py-1.5 text-amber-900 ring-1 ring-amber-200">
          This file is gender-sensitive ({sensitive}), so the bias check ran automatically.
        </p>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-slate-600">Swap:</span>
        {ATTRIBUTES.map(([a, label]) => (
          <button key={a} type="button" onClick={() => setAttribute(a)} className={`rounded-md px-2 py-1 ring-1 ${attribute === a ? 'bg-accent text-on-accent ring-indigo-600' : 'bg-white text-slate-600 ring-slate-200'}`}>
            {label}
          </button>
        ))}
        {data?.verdict && <Badge tone={data.verdict === 'possible_bias' ? 'bad' : data.verdict === 'no_difference' ? 'good' : 'info'}>{data.verdict.replace('_', ' ')}</Badge>}
      </div>
      {!data ? (
        <div className="flex flex-wrap items-center gap-2">
          <button type="button" onClick={() => go(attribute)} disabled={busy} className="rounded-lg bg-accent px-3 py-1.5 text-xs font-semibold text-on-accent hover:bg-accent-hover disabled:bg-slate-300">
            {busy ? 'Analysing the swapped file…' : 'Run the bias check'}
          </button>
          <span className="text-[11px] text-slate-500">facts, offences and element checks, with each version run twice (a few minutes)</span>
        </div>
      ) : (
        <BiasResult data={data} attribute={attribute} />
      )}
    </div>
  )
}

export default function CaseAnalysisView() {
  const [info, setInfo] = useState(null)
  const [models, setModels] = useState([])
  const [model, setModel] = useState('')
  const [localOnly, setLocalOnly] = useState(false)
  const [text, setText] = useState('')
  const [file, setFile] = useState(null) // { name, size }
  const [dragging, setDragging] = useState(false)
  const [run, setRun] = useState(null)
  const [busy, setBusy] = useState(false)
  const [openSource, setOpenSource] = useState(null)
  const [followUps, setFollowUps] = useState([]) // questions about the current analysis
  const [bias, setBias] = useState({}) // bias check results for the current analysis, by attribute
  const fileRef = useRef(null)
  const fileData = useRef('')
  const abortRef = useRef(null)
  const reportRef = useRef(null)

  useEffect(() => {
    caseApi.info().then(setInfo).catch(() => setInfo({ error: true }))
    caseApi.models().then((list) => {
      setModels(list.filter((m) => m.available))
      setModel((m) => m || list.find((x) => x.default)?.id || '')
    }).catch(() => {})
  }, [])

  const localModels = info?.local_models || []
  useEffect(() => {
    if (localOnly && !localModels.includes(model)) setModel(localModels[0] || '')
  }, [localOnly, localModels, model])

  const pickFile = async (f) => {
    if (!f) return
    if (!/\.(txt|pdf)$/i.test(f.name)) {
      setRun({ error: 'Choose a .txt or .pdf file.' })
      return
    }
    fileData.current = await readFileBase64(f)
    setFile({ name: f.name, size: f.size })
    setRun(null)
  }

  const start = async () => {
    setBusy(true)
    const r = { steps: {}, answer: '' }
    setRun({ ...r })
    setFollowUps([]) // they belonged to the previous analysis
    setBias({})
    const controller = new AbortController()
    abortRef.current = controller
    requestAnimationFrame(() => reportRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
    const update = (fn) => setRun((prev) => fn({ ...prev }))
    try {
      await caseApi.analyze(
        { text: file ? '' : text, filename: file?.name || '', file_base64: file ? fileData.current : '', model, local_only: localOnly },
        (event, data) => {
          if (event === 'step') {
            update((p) => {
              const steps = { ...p.steps }
              for (const s of STEPS) if (steps[s.id] === 'running') steps[s.id] = 'done'
              steps[data.id] = 'running'
              return { ...p, steps }
            })
          } else if (event === 'token') update((p) => ({ ...p, answer: p.answer + data.text }))
          else if (event === 'done') update((p) => ({ ...p, done: data, steps: Object.fromEntries(STEPS.map((s) => [s.id, 'done'])) }))
          else if (event === 'error') update((p) => ({ ...p, error: String(data?.message || data) }))
          else if (event === 'analysis_replace') update((p) => ({ ...p, answer: data.text, removed: data.removed }))
          else if (event === 'guard') update((p) => ({ ...p, guards: [...(p.guards || []), data] }))
          else update((p) => ({ ...p, [event]: data }))
        },
        controller.signal,
      )
    } catch (e) {
      if (e.name !== 'AbortError') update((p) => ({ ...p, error: e.message.replace(/^HTTP \d+: /, '') }))
    } finally {
      setBusy(false)
      abortRef.current = null
    }
  }

  const chars = text.trim().length
  const ready = (file || chars >= 200) && model && !busy
  const statute = run?.law?.sources || []

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <section className="flex min-h-full flex-col items-center justify-center bg-field-green px-4 py-8 text-paper sm:py-10">
        <h2 className="head animate-rise text-center">Open a case file</h2>
        <p className="standfirst mt-3 animate-rise text-center" style={stagger(1)}>
          Give it an FIR, charge sheet or case note. It reads the facts, finds the sections they may engage, tests every element and cites
          past judgments. It never decides guilt.
        </p>

        <div
          className={`light-scope relative mt-11 w-full max-w-3xl animate-sheet text-ink transition-transform duration-300 ${dragging ? '-translate-y-1.5' : ''}`}
          style={stagger(2, 80)}
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragging(false)
            pickFile(e.dataTransfer.files[0])
          }}
        >
          <span aria-hidden="true" className="absolute -top-7 left-0 flex h-7 items-center bg-[#d9c28f] px-5 text-[15px] font-semibold">Case file</span>
          <div className="relative bg-[#d9c28f] p-3 pr-8 sm:p-5 sm:pr-14">
            <RedTape className="-bottom-1 -top-1 right-2 sm:right-5" />
            <div className="relative bg-white px-4 pb-4 pt-5 sm:px-7 sm:pb-5 sm:pt-6">
              <div className="flex items-start justify-between gap-3">
                <p className="text-[1.25rem] font-semibold tracking-[-0.02em]">Facts of the case</p>
                {(file || chars >= 200) && <Stamp className="text-indigo-600">Received</Stamp>}
              </div>
              {file ? (
                <p className="mt-4 border-y border-slate-200 py-5 text-[17px]">
                  <b className="font-semibold">{file.name}</b>
                  <span className="text-slate-600"> · {(file.size / 1024).toFixed(0)} KB · </span>
                  <button className="font-medium text-indigo-600 underline underline-offset-2" onClick={() => setFile(null)}>remove</button>
                </p>
              ) : (
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  rows={6}
                  maxLength={200000}
                  placeholder={dragging ? 'Drop the case file' : 'Paste the facts here, at least 200 characters, or attach a file.'}
                  aria-label="Case text"
                  className="ruled mt-2 block w-full resize-y bg-transparent font-serif text-[1.125rem] text-slate-900 placeholder:text-slate-500 focus:outline-none"
                />
              )}
              <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                <button onClick={() => fileRef.current?.click()} className="press inline-flex h-10 items-center gap-2 bg-slate-100 px-3.5 text-[15px] font-medium hover:bg-slate-200">
                  <svg {...stroke} className="size-5"><path d="M15.5 9.5 9.8 15.2a3.4 3.4 0 0 1-4.8-4.8l6-6a2.2 2.2 0 0 1 3.2 3.2l-5.8 5.8a1 1 0 0 1-1.5-1.5l5.2-5.2" /></svg>
                  Attach .txt or .pdf
                </button>
                <input ref={fileRef} type="file" accept=".txt,.pdf" className="hidden" onChange={(e) => { pickFile(e.target.files[0]); e.target.value = '' }} />
                {!file && (
                  <span className={`text-[13px] tabular-nums ${chars > 0 && chars < 200 ? 'font-medium text-amber-700' : 'text-slate-500'}`}>
                    {chars > 0 && chars < 200 ? `${200 - chars} more characters needed` : `${chars.toLocaleString()} characters`}
                  </span>
                )}
              </div>
            </div>

            <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 sm:mt-4">
              <Select
                up
                label="Model"
                value={model}
                onChange={setModel}
                buttonClass="h-11 min-w-44 bg-white text-ink"
                menuClass="w-72"
                options={models.filter((m) => !localOnly || localModels.includes(m.id)).map((m) => ({
                  value: m.id,
                  label: m.id === 'groq' ? 'Base (Groq)' : m.id === 'adapter' ? 'Local (LoRA)' : m.id === 'local-heavy' ? 'Local heavy (7B)' : m.id,
                  hint: m.model_id,
                }))}
              />
              <button
                type="button"
                role="switch"
                aria-checked={localOnly}
                disabled={!localModels.length}
                onClick={() => setLocalOnly((v) => !v)}
                className="press flex items-center gap-2 text-left text-[15px] font-medium disabled:opacity-50"
                title={localModels.length ? 'Only a local model reads the file' : 'No local model is set up'}
              >
                <span className={`relative h-6 w-10 shrink-0 ${localOnly ? 'bg-ink' : 'bg-ink/25'}`}>
                  <span className={`absolute left-1 top-1 size-4 bg-white transition-transform duration-200 ${localOnly ? 'translate-x-4' : ''}`} />
                </span>
                Keep on this computer
              </button>
              {busy ? (
                <button onClick={() => abortRef.current?.abort()} className="press ml-auto h-11 bg-ink px-6 text-[15px] font-medium text-paper">
                  Stop
                </button>
              ) : (
                <button
                  onClick={start}
                  disabled={!ready}
                  className="press ml-auto inline-flex h-11 items-center gap-2 bg-ink px-6 text-[15px] font-medium text-paper hover:bg-field-red hover:text-ink disabled:bg-ink/30 disabled:text-ink/60"
                >
                  Analyse the file
                  <svg {...stroke} strokeWidth={2} className="size-4"><path d="M4.5 10h11M11 5.5l4.5 4.5-4.5 4.5" /></svg>
                </button>
              )}
            </div>
          </div>
        </div>
        <p className="mt-4 max-w-xl text-center text-[15px] leading-snug animate-rise" style={stagger(4, 80)}>
          {localOnly
            ? 'Nothing leaves this computer: a local model reads the file, and the searches run here.'
            : 'With a hosted model the text is sent to that provider (Groq or Gemini) for this analysis. Nothing is stored.'}
        </p>
      </section>

      <div className="mx-auto max-w-4xl space-y-6 px-4 empty:hidden [&:has(>div:not(:empty))]:py-8">
        <div ref={reportRef} className="scroll-mt-4 space-y-5" aria-live="polite">
          {run?.steps && <Stepper steps={run.steps} />}
          {run?.error && (
            <p className="animate-rise rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-200">{run.error}</p>
          )}

          {run?.guards?.length > 0 && (
            <div className="animate-rise space-y-1.5 rounded-xl bg-red-50 px-4 py-3 text-sm text-red-800 ring-1 ring-red-200" role="status">
              <p className="flex items-center gap-2 font-semibold">
                <svg {...stroke} className="size-4"><path d="M10 2.75 3.75 5v4.5c0 3.6 2.6 6.4 6.25 7.75 3.65-1.35 6.25-4.15 6.25-7.75V5Z" /></svg>
                Safety checks acted on this file
              </p>
              {run.guards.map((g, i) => <p key={i} className="text-xs">{g.message}</p>)}
            </div>
          )}

          {run?.facts && (
            <Section n={1} title="What the file says" hint={`Facts read by the model; each quote was checked against your file${run.facts.truncated ? ` (only the first ${run.facts.chars_read.toLocaleString()} characters were read)` : ''}.`}>
              {run.facts.summary && <p className="mb-4 rounded-xl bg-slate-50 px-4 py-3 text-sm leading-relaxed text-slate-700 ring-1 ring-slate-200">{run.facts.summary}</p>}
              <ol className="space-y-2">
                {run.facts.facts.map((f, i) => (
                  <li key={f.id} className={`flex animate-rise gap-3 text-sm ${f.verified ? '' : 'opacity-70'}`} style={stagger(i, 40)}>
                    <span className="w-7 shrink-0 font-mono text-xs text-slate-400">{f.id}</span>
                    <div className="min-w-0">
                      <p className="text-slate-800">{f.fact}</p>
                      <p className="mt-0.5 text-xs italic text-slate-500">
                        “{f.quote}”{' '}
                        {f.injection ? (
                          <span className="not-italic font-medium text-red-600">· reads like an instruction to an AI (possible prompt injection), ignored</span>
                        ) : (
                          !f.verified && <span className="not-italic font-medium text-red-600">· quote not found in the file, not used</span>
                        )}
                      </p>
                    </div>
                  </li>
                ))}
              </ol>
            </Section>
          )}

          {run?.law && (
            <Section n={2} title="The law" hint="IPC sections these facts may engage, with their text. Click to read.">
              {run.facts?.offences?.length ? (
                <p className="mb-3 flex flex-wrap gap-1.5">
                  {run.facts.offences.map((o) => (
                    <span key={o.section} className="rounded-lg bg-slate-900 px-2.5 py-1 text-xs font-semibold text-slate-50">
                      s.{o.section} · {o.name}
                      {run.law.bns?.[o.section] && (
                        <span className="ml-1.5 font-normal opacity-75">
                          → {run.law.bns[o.section].length ? `BNS ${run.law.bns[o.section].join(', ')}` : 'no BNS equivalent'}
                        </span>
                      )}
                    </span>
                  ))}
                </p>
              ) : (
                <p className="mb-3 text-sm text-slate-500">The model named no IPC offence for these facts.</p>
              )}
              <div className="flex flex-wrap gap-1.5">{statute.map((s) => <SourceChip key={s.chunk_id} s={s} onOpen={setOpenSource} />)}</div>
            </Section>
          )}

          {run?.elements && (
            <Section n={3} title="Does the file show each element?" hint="What each offence requires, and whether this file shows it.">
              <div className="space-y-5">
                {run.elements.checks.map((c) => (
                  <div key={c.section}>
                    <div className="mb-2 flex flex-wrap items-center gap-2">
                      <span className="text-sm font-semibold text-slate-900">s.{c.section} · {c.title}</span>
                      <span className={`rounded-md px-1.5 py-0.5 text-[11px] font-medium ring-1 ${c.method === 'rules' ? 'bg-indigo-50 text-indigo-700 ring-indigo-200' : 'bg-slate-100 text-slate-600 ring-slate-200'}`}>
                        {c.method === 'rules' ? 'Rule engine (Prolog)' : 'Model-assessed'}
                      </span>
                      {c.status && VERDICT_WORD[c.status] && (
                        <span className={`rounded-md px-1.5 py-0.5 text-[11px] font-medium ring-1 ${VERDICT_WORD[c.status].cls}`}>{VERDICT_WORD[c.status].text}</span>
                      )}
                    </div>
                    {c.error ? (
                      <p className="text-sm text-red-700">{c.error}</p>
                    ) : c.method === 'rules' && c.result ? (
                      <VerificationResultView result={c.result} className="space-y-4 rounded-xl bg-slate-50/60 p-4 ring-1 ring-slate-200" />
                    ) : (
                      <ModelCheck c={c} facts={run.facts?.facts} />
                    )}
                  </div>
                ))}
                {run.elements.checks.length === 0 && <p className="text-sm text-slate-500">No offence to check.</p>}
              </div>
            </Section>
          )}

          {run?.similar && (
            <Section n={4} title="Similar past judgments" hint="Judgments whose facts are closest to this file, and what the court held there.">
              {run.similar.cases.length === 0 ? (
                <p className="text-sm text-slate-500">No sufficiently similar judgment in the library.</p>
              ) : (
                <div className="grid gap-3">
                  {run.similar.cases.map((c, i) => (
                    <article key={c.case_id} className="animate-rise rounded-xl bg-amber-50/40 p-4 ring-1 ring-amber-200/70" style={stagger(i, 80)}>
                      <div className="flex flex-wrap items-baseline justify-between gap-2">
                        <h4 className="text-sm font-semibold text-slate-900">{c.title}</h4>
                        <span className="text-xs text-slate-500">
                          {[c.court, c.year].filter(Boolean).join(' · ')} · {Math.round(c.similarity * 100)}% similar facts
                        </span>
                      </div>
                      <p className="mt-2 text-xs leading-relaxed text-slate-600"><b className="font-medium text-slate-700">Facts:</b> {c.facts_passage.slice(0, 320)}{c.facts_passage.length > 320 ? '…' : ''}</p>
                      {c.source && (
                        <p className="mt-2 text-xs leading-relaxed text-slate-700"><b className="font-medium">The court held:</b> {c.source.text.slice(0, 420)}{c.source.text.length > 420 ? '…' : ''}</p>
                      )}
                    </article>
                  ))}
                </div>
              )}
            </Section>
          )}

          {(run?.answer || run?.steps?.analysis) && run?.sources && (
            <Section n={5} title="Analysis" hint="Written only from the numbered sources; every claim is cited.">
              <div className="text-[0.9375rem] leading-relaxed text-slate-800">
                <AnswerText text={run.answer || ''} sources={run.sources} onCite={setOpenSource} caret={busy} />
              </div>
              {run.sources?.length > 0 && (
                <div className="mt-4 flex flex-wrap gap-1.5 border-t border-slate-100 pt-3">
                  {run.sources.map((s) => <SourceChip key={s.chunk_id + s.marker} s={s} onOpen={setOpenSource} />)}
                </div>
              )}
              {run.explanation && <WhyPanel explanation={run.explanation} sources={run.sources} retrieval={null} onOpenSource={setOpenSource} />}
            </Section>
          )}

          {run?.done && !run.done.out_of_scope && run.facts?.offences?.length > 0 && (
            <Section n={6} title="Bias check" hint="Would the analysis change if only who the people are changed: gender, religion, caste, income, region or age?">
              <CaseBias
                run={run}
                request={{ text: file ? '' : text, filename: file?.name || '', file_base64: file ? fileData.current : '', model, local_only: localOnly }}
                result={bias}
                setResult={setBias}
              />
            </Section>
          )}

          {run?.done && !run.done.out_of_scope && run.sources?.length > 0 && (
            <Section n={7} title="Ask about this analysis" hint="Follow-up questions, answered from this analysis and its sources.">
              <AskAboutAnalysis run={run} model={model} localOnly={localOnly} turns={followUps} setTurns={setFollowUps} onOpenSource={setOpenSource} />
            </Section>
          )}

          {run?.done && (
            <p className="flex animate-rise gap-2 rounded-xl bg-amber-50 px-4 py-3 text-xs leading-relaxed text-amber-900 ring-1 ring-amber-200">
              <svg {...stroke} className="mt-px size-4 shrink-0"><circle cx="10" cy="10" r="7.25" /><path d="M10 6.5v4.5M10 13.6v.1" /></svg>
              {run.done.disclaimer} Model: {run.done.model}{run.done.local_only ? ' (local only)' : ''}.
              {run.done.model_note ? ` Note: ${run.done.model_note}.` : ''}
            </p>
          )}
        </div>
      </div>
      <SourceModal source={openSource} onClose={() => setOpenSource(null)} />
    </div>
  )
}
