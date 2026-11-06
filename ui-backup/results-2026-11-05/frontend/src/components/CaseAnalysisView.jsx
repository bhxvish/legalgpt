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

/** A numbered part of the report, set like the paragraphs of a judgment. */
function Section({ n, title, hint, children, delay = 0 }) {
  return (
    <section className="grid animate-rise gap-x-5 border-t-2 border-ink pt-4 sm:grid-cols-[2.75rem_minmax(0,1fr)]" style={stagger(delay)}>
      <span className="text-[2rem] font-semibold leading-none tracking-[-0.04em] text-indigo-600">{n}</span>
      <div className="min-w-0">
        <h3 className="mt-1 text-[1.625rem] leading-none tracking-[-0.03em] text-slate-900 sm:mt-0">{title}</h3>
        {hint && <p className="mt-1.5 text-[15px] leading-snug text-slate-600">{hint}</p>}
        <div className="mt-4">{children}</div>
      </div>
    </section>
  )
}

/** Five stages along one line: each fills in as it finishes. */
function Stepper({ steps, failed }) {
  return (
    <ol className="grid grid-cols-5 gap-1" aria-label="Progress">
      {STEPS.map((s) => {
        const st = steps[s.id] || 'waiting'
        return (
          <li key={s.id} title={s.title} className="min-w-0">
            <span aria-hidden="true" className="block h-1.5 overflow-hidden bg-slate-200">
              <span className={`block h-full origin-left bg-ink transition-transform duration-500 ease-[cubic-bezier(0.22,1,0.36,1)] ${st === 'done' ? 'scale-x-100' : st === 'running' ? `scale-x-50 ${failed ? 'bg-red-500' : 'animate-pulse bg-indigo-500'}` : 'scale-x-0'}`} />
            </span>
            <span className={`mt-1.5 block truncate text-[13px] font-medium sm:whitespace-normal ${st === 'waiting' ? 'text-slate-500' : 'text-slate-900'}`}>
              <span className="sr-only">{st}: </span>{s.label}
            </span>
          </li>
        )
      })}
    </ol>
  )
}

function SourceChip({ s, onOpen }) {
  const judgment = s.kind === 'judgment'
  return (
    <button onClick={() => onOpen(s)} className="group flex w-full min-w-0 items-baseline gap-2 border-b border-slate-200 py-1.5 text-left text-[15px] leading-snug text-slate-900">
      <span className="w-4 shrink-0 text-right font-semibold text-indigo-600">{s.marker}</span>
      {judgment ? (
        <span className="min-w-0 truncate group-hover:underline">{s.title}</span>
      ) : (
        <span className="min-w-0 group-hover:underline">
          s.{s.section} {s.title}
          {s.bns?.length > 0 && <span className="ml-2 text-[13px] text-slate-500">{s.bns[0] === 'none' ? 'no BNS' : `BNS ${s.bns.join(', ')}`}</span>}
        </span>
      )}
    </button>
  )
}

function ModelCheck({ c, facts }) {
  const byId = Object.fromEntries((facts || []).map((f) => [f.id, f]))
  return (
    <ol>
      {c.elements.map((e, i) => {
        const st = ELEMENT[e.status] || ELEMENT.missing
        return (
          <li key={i} className="flex animate-rise gap-3 border-t border-slate-200 py-3" style={stagger(i, 70)}>
            <span className={`mt-0.5 grid size-6 shrink-0 place-items-center ring-1 ${st.cls}`} title={st.label}>
              <svg {...stroke} strokeWidth={2} className="size-3.5"><path d={st.path} /></svg>
            </span>
            <div className="min-w-0 text-[16px] leading-snug">
              <p className="text-slate-900">
                {e.element} <span className="text-[14px] font-semibold text-slate-600">{st.label}</span>
              </p>
              {e.explanation && <p className="mt-0.5 text-[15px] text-slate-600">{e.explanation}</p>}
              {e.facts?.length > 0 && (
                <p className="mt-1 flex flex-wrap gap-1">
                  {e.facts.map((id) => (
                    <span key={id} title={byId[id]?.quote} className="bg-slate-100 px-1.5 py-px text-[13px] font-medium text-slate-700">
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
          <div>
            {t.error ? (
              <p className="border-l-4 border-red-500 pl-3 text-[15px] text-red-700">{t.error}</p>
            ) : (
              <div className="reading">
                <AnswerText text={t.answer} sources={t.sources || run.sources} onCite={onOpenSource} caret={busy && i === turns.length - 1} />
              </div>
            )}
            {t.guard && <p className="mt-2 text-xs text-red-700">{t.guard}</p>}
            {(() => {
              // sections retrieved for this question beyond the analysis's own sources: shown only if the answer cites them
              const cited = new Set([...(t.answer || '').matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])))
              const extra = (t.sources || []).slice(run.sources?.length || 0).filter((s) => cited.has(s.marker))
              return extra.length > 0 && (
                <div className="mt-3">
                  <p className="text-[14px] text-slate-600">Also used for this question:</p>
                  {extra.map((s) => <SourceChip key={s.chunk_id + s.marker} s={s} onOpen={onOpenSource} />)}
                </div>
              )
            })()}
            {t.done?.model && <p className="mt-2 text-[13px] text-slate-500">Model: {t.done.model}{t.done.model_note ? ` · ${t.done.model_note}` : ''}</p>}
          </div>
        </div>
      ))}
      {turns.length === 0 && (
        <ul>
          {FOLLOW_UPS.map((q) => (
            <li key={q} className="border-b border-slate-200">
              <button onClick={() => ask(q)} className="press group flex w-full items-baseline justify-between gap-4 py-2 text-left text-[17px] font-medium text-slate-900 hover:text-indigo-600">
                {q}
                <span aria-hidden="true" className="transition-transform duration-200 group-hover:translate-x-1">→</span>
              </button>
            </li>
          ))}
        </ul>
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
          className="block min-h-[3rem] w-full resize-y bg-slate-100 px-3.5 py-3 text-[16px] text-slate-900 placeholder:text-slate-500 focus:outline-none"
        />
        {busy ? (
          <button type="button" onClick={() => abortRef.current?.abort()} className="press h-12 shrink-0 bg-slate-200 px-5 text-[15px] font-medium text-slate-900">
            Stop
          </button>
        ) : (
          <button type="submit" disabled={!question.trim()} className="press h-12 shrink-0 bg-ink px-5 text-[15px] font-medium text-paper hover:bg-field-red hover:text-ink disabled:bg-slate-200 disabled:text-slate-500">
            Ask
          </button>
        )}
      </form>
      <p className="text-[14px] leading-snug text-slate-600">
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
    <div className="space-y-3 text-[14px]">
      {sensitive && (
        <p className="border-l-4 border-amber-500 pl-3 text-slate-800">
          This file is gender-sensitive ({sensitive}), so the bias check ran automatically.
        </p>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-slate-600">Swap:</span>
        {ATTRIBUTES.map(([a, label]) => (
          <button key={a} type="button" onClick={() => setAttribute(a)} className={`press h-9 px-3 text-[15px] font-medium ${attribute === a ? 'bg-ink text-paper' : 'bg-slate-100 text-slate-800 hover:bg-slate-200'}`}>
            {label}
          </button>
        ))}
        {data?.verdict && <Badge tone={data.verdict === 'possible_bias' ? 'bad' : data.verdict === 'no_difference' ? 'good' : 'info'}>{data.verdict.replace('_', ' ')}</Badge>}
      </div>
      {!data ? (
        <div className="flex flex-wrap items-center gap-2">
          <button type="button" onClick={() => go(attribute)} disabled={busy} className="press h-11 bg-ink px-5 text-[15px] font-medium text-paper hover:bg-field-red hover:text-ink disabled:bg-slate-300">
            {busy ? 'Analysing the swapped file…' : 'Run the bias check'}
          </button>
          <span className="text-[14px] text-slate-600">Facts, offences and element checks, on each version (a few minutes)</span>
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
          else if (event === 'error') update((p) => ({ ...p, error: String(data?.message || data).replace(/^\w*Error: /, '') }))
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

      <div className={run ? 'bg-field-green px-4 pb-10' : 'hidden'}>
        <div ref={reportRef} className="light-scope mx-auto max-w-4xl scroll-mt-4 animate-sheet space-y-8 bg-white p-5 text-ink sm:p-9" aria-live="polite">
          <header className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 className="head">Case report</h2>
              <p className="mt-2 text-[16px] text-slate-700">{run?.facts?.offences?.length ? `Offences read from the file: ${run.facts.offences.map((o) => `s.${o.section}`).join(', ')}` : 'What the law requires, and what the file shows or leaves open.'}</p>
            </div>
            {run?.done && <Stamp className="text-indigo-600">Not a verdict</Stamp>}
          </header>
          {run?.steps && !run.done && <Stepper steps={run.steps} failed={!!run.error} />}
          {run?.error && <p className="animate-rise border-l-4 border-red-500 pl-3 text-[16px] leading-snug text-red-700">{run.error}</p>}

          {run?.guards?.length > 0 && (
            <div className="animate-rise border-l-4 border-red-500 pl-3 text-[15px] leading-snug" role="status">
              <p className="font-semibold text-slate-900">Safety checks acted on this file</p>
              {run.guards.map((g, i) => <p key={i} className="text-slate-700">{g.message}</p>)}
            </div>
          )}

          {run?.facts?.facts?.length > 0 && (
            <Section n={1} title="What the file says" hint={`Read by the model; each quote was checked against your file${run.facts.truncated ? ` (only the first ${run.facts.chars_read.toLocaleString()} characters were read)` : ''}.`}>
              {run.facts.summary && <p className="reading mb-4">{run.facts.summary}</p>}
              <ol>
                {run.facts.facts.map((f, i) => (
                  <li key={f.id} className={`flex animate-rise gap-3 border-t border-slate-200 py-2 ${f.verified ? '' : 'opacity-70'}`} style={stagger(i, 40)}>
                    <span className="w-7 shrink-0 text-[14px] font-semibold text-indigo-600">{f.id}</span>
                    <div className="min-w-0">
                      <p className="text-[16px] leading-snug text-slate-900">{f.fact}</p>
                      <p className="mt-0.5 font-serif text-[15px] italic leading-snug text-slate-600">
                        “{f.quote}”{' '}
                        {f.injection ? (
                          <span className="font-sans font-medium not-italic text-red-700">Reads like an instruction to an AI (possible prompt injection), ignored.</span>
                        ) : (
                          !f.verified && <span className="font-sans font-medium not-italic text-red-700">Quote not found in the file, not used.</span>
                        )}
                      </p>
                    </div>
                  </li>
                ))}
              </ol>
            </Section>
          )}

          {run?.law && (
            <Section n={2} title="The law" hint="IPC sections these facts may engage. Open one to read its text.">
              {run.facts?.offences?.length ? (
                <ul className="mb-4 space-y-1">
                  {run.facts.offences.map((o) => (
                    <li key={o.section} className="flex flex-wrap items-baseline gap-x-3 text-[1.125rem] font-semibold tracking-[-0.02em] text-slate-900">
                      s.{o.section} {o.name}
                      {run.law.bns?.[o.section] && (
                        <span className="text-[15px] font-normal tracking-normal text-slate-600">
                          {run.law.bns[o.section].length ? `BNS ${run.law.bns[o.section].join(', ')}` : 'no BNS equivalent'}
                        </span>
                      )}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="mb-3 text-[16px] text-slate-600">The model named no IPC offence for these facts.</p>
              )}
              <div className="grid gap-x-8 sm:grid-cols-2">{statute.map((s) => <SourceChip key={s.chunk_id} s={s} onOpen={setOpenSource} />)}</div>
            </Section>
          )}

          {run?.elements && (
            <Section n={3} title="Does the file show each element?" hint="What each offence requires, and whether this file shows it.">
              <div className="space-y-7">
                {run.elements.checks.map((c) => (
                  <div key={c.section}>
                    {c.method !== 'rules' || !c.result ? (
                      <p className="mb-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
                        <span className="text-[1.125rem] font-semibold tracking-[-0.02em] text-slate-900">s.{c.section} {c.title}</span>
                        {c.status && VERDICT_WORD[c.status] && <span className="text-[15px] font-medium text-slate-700">{VERDICT_WORD[c.status].text}</span>}
                      </p>
                    ) : null}
                    <p className="mb-2 text-[14px] text-slate-600">{c.method === 'rules' ? 'Tested by the rule engine (Prolog)' : 'Assessed by the model'}</p>
                    {c.error ? (
                      <p className="border-l-4 border-red-500 pl-3 text-[15px] text-red-700">{c.error}</p>
                    ) : c.method === 'rules' && c.result ? (
                      <VerificationResultView result={c.result} compact />
                    ) : (
                      <ModelCheck c={c} facts={run.facts?.facts} />
                    )}
                  </div>
                ))}
                {run.elements.checks.length === 0 && <p className="text-[16px] text-slate-600">No offence to check.</p>}
              </div>
            </Section>
          )}

          {run?.similar && (
            <Section n={4} title="Similar past judgments" hint="Judgments whose facts are closest to this file, and what the court held there.">
              {run.similar.cases.length === 0 ? (
                <p className="text-[16px] text-slate-600">No sufficiently similar judgment in the library.</p>
              ) : (
                <div>
                  {run.similar.cases.map((c, i) => (
                    <article key={c.case_id} className="animate-rise border-t border-slate-200 py-3" style={stagger(i, 80)}>
                      <div className="flex flex-wrap items-baseline justify-between gap-x-4">
                        <h4 className="text-[1.125rem] font-semibold tracking-[-0.02em] text-slate-900">{c.title}</h4>
                        <span className="text-[14px] text-slate-600">
                          {[c.court, c.year].filter(Boolean).join(', ')}. {Math.round(c.similarity * 100)}% similar facts
                        </span>
                      </div>
                      <p className="mt-1.5 font-serif text-[16px] leading-snug text-slate-700"><b className="font-sans text-[14px] font-semibold text-slate-900">Facts </b>{c.facts_passage.slice(0, 320)}{c.facts_passage.length > 320 ? '…' : ''}</p>
                      {c.source && (
                        <p className="mt-1.5 font-serif text-[16px] leading-snug text-slate-700"><b className="font-sans text-[14px] font-semibold text-slate-900">The court held </b>{c.source.text.slice(0, 420)}{c.source.text.length > 420 ? '…' : ''}</p>
                      )}
                    </article>
                  ))}
                </div>
              )}
            </Section>
          )}

          {(run?.answer || run?.steps?.analysis) && run?.sources && (
            <Section n={5} title="Analysis" hint="Written only from the numbered sources; every claim is cited.">
              <div className="reading">
                <AnswerText text={run.answer || ''} sources={run.sources} onCite={setOpenSource} caret={busy} />
              </div>
              {run.sources?.length > 0 && (
                <div className="mt-5 grid gap-x-8 sm:grid-cols-2">
                  {run.sources.map((s) => <SourceChip key={s.chunk_id + s.marker} s={s} onOpen={setOpenSource} />)}
                </div>
              )}
              {run.explanation && <WhyPanel explanation={run.explanation} sources={run.sources} retrieval={null} onOpenSource={setOpenSource} />}
            </Section>
          )}

          {run?.done && !run.done.out_of_scope && run.facts?.offences?.length > 0 && (
            <Section n={6} title="Bias check" hint="Would the analysis change if only who the people are changed?">
              <CaseBias
                run={run}
                request={{ text: file ? '' : text, filename: file?.name || '', file_base64: file ? fileData.current : '', model, local_only: localOnly }}
                result={bias}
                setResult={setBias}
              />
            </Section>
          )}

          {run?.done && !run.done.out_of_scope && run.sources?.length > 0 && (
            <Section n={7} title="Ask about this analysis" hint="Answered from this analysis and its sources.">
              <AskAboutAnalysis run={run} model={model} localOnly={localOnly} turns={followUps} setTurns={setFollowUps} onOpenSource={setOpenSource} />
            </Section>
          )}

          {run?.done && (
            <p className="animate-rise border-l-4 border-amber-500 pl-3 text-[15px] leading-snug text-slate-800">
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
