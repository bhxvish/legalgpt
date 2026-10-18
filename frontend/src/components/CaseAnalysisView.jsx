import { useEffect, useRef, useState } from 'react'
import WhereItFits from './WhereItFits.jsx'
import { caseApi } from '../api.js'
import { AnswerText } from './ChatWindow.jsx'
import { CaseFileArt, HeaderArt } from './LawArt.jsx'
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
    <section className="animate-rise rounded-2xl bg-white p-5 shadow-[0_1px_2px_rgb(15_23_42/0.04)] ring-1 ring-slate-200 sm:p-6" style={stagger(delay)}>
      <div className="mb-4 flex items-baseline gap-3">
        <span className="grid size-6 shrink-0 place-items-center rounded-full bg-indigo-50 text-xs font-semibold text-indigo-700 ring-1 ring-indigo-200">{n}</span>
        <div>
          <h3 className="text-base font-semibold tracking-tight text-slate-900">{title}</h3>
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
    <div className="relative isolate min-h-0 flex-1 overflow-y-auto px-4 py-8 sm:py-10">
      <div aria-hidden="true" className="bg-dots pointer-events-none absolute inset-0 -z-10" />
      <div className="mx-auto max-w-4xl space-y-6">
        <header className="relative">
          <div aria-hidden="true" className="pointer-events-none absolute -left-24 -top-24 -z-10 h-64 w-[30rem] max-w-[120%] animate-float rounded-full bg-[radial-gradient(closest-side,var(--color-indigo-200),transparent)] opacity-50 blur-2xl" />
          <HeaderArt>
            <CaseFileArt />
          </HeaderArt>
          <p className="mb-3 inline-flex animate-rise items-center gap-1.5 rounded-full bg-white/70 px-2.5 py-1 text-xs font-medium text-slate-600 ring-1 ring-slate-200 backdrop-blur">
            <span aria-hidden="true" className="size-1.5 rounded-full bg-amber-500" />
            Analysis, not a verdict
          </p>
          <h2 className="animate-rise text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl" style={stagger(1)}>
            Analyse a case file
          </h2>
          <p className="mt-2 max-w-2xl animate-rise text-sm leading-relaxed text-slate-500 sm:text-[0.9375rem]" style={stagger(2)}>
            Upload an FIR, charge sheet or case note. LegalGPT reads the facts (each with a quote from your file),
            finds the IPC sections they may engage, checks every element, finds similar past judgments, and writes a
            cited analysis. It shows what the law requires and what the file shows or leaves open; it does not decide guilt.
          </p>
          <WhereItFits part="case" className="mt-4 animate-rise" />
        </header>

        <section className="grid animate-rise gap-5 rounded-2xl bg-white p-5 shadow-[0_1px_2px_rgb(15_23_42/0.04),0_12px_32px_-20px_rgb(15_23_42/0.25)] ring-1 ring-slate-200 sm:p-6" style={stagger(3)}>
          <div
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
            className={`flex flex-wrap items-center gap-4 rounded-xl border-2 border-dashed px-4 py-4 transition-[border-color,background-color,transform] duration-200 ${
              dragging ? 'scale-[1.01] border-indigo-400 bg-indigo-50' : 'border-slate-200 hover:border-slate-300'
            }`}
          >
            <span className="grid size-11 shrink-0 place-items-center rounded-xl bg-slate-50 text-slate-500 ring-1 ring-slate-200">
              <svg {...stroke} className="size-5"><path d="M10 13V3.75M6.25 7.5 10 3.75l3.75 3.75" /><path d="M3.75 12.5v1.75a2 2 0 0 0 2 2h8.5a2 2 0 0 0 2-2V12.5" /></svg>
            </span>
            <div className="min-w-0 flex-1">
              {file ? (
                <>
                  <p className="truncate text-sm font-semibold text-slate-800">{file.name}</p>
                  <p className="text-xs text-slate-500">
                    {(file.size / 1024).toFixed(0)} KB ·{' '}
                    <button className="font-medium text-indigo-600 hover:underline" onClick={() => setFile(null)}>remove</button>
                  </p>
                </>
              ) : (
                <>
                  <p className="text-sm font-semibold text-slate-800">{dragging ? 'Drop the case file' : 'Upload a case file'}</p>
                  <p className="text-xs text-slate-500">.txt or .pdf, up to 8 MB, or paste the text below</p>
                </>
              )}
            </div>
            <button onClick={() => fileRef.current?.click()} className="rounded-xl bg-slate-100 px-3.5 py-2 text-sm font-medium text-slate-700 ring-1 ring-slate-200 hover:bg-slate-200">
              Choose file
            </button>
            <input ref={fileRef} type="file" accept=".txt,.pdf" className="hidden" onChange={(e) => { pickFile(e.target.files[0]); e.target.value = '' }} />
          </div>

          {!file && (
            <div>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={6}
                maxLength={200000}
                placeholder="…or paste the facts of the case here (at least 200 characters)."
                aria-label="Case text"
                className="block w-full resize-y rounded-xl border border-slate-200 bg-slate-50/60 px-3.5 py-3 text-sm leading-relaxed text-slate-900 placeholder:text-slate-400 transition-[border-color,box-shadow,background-color] focus:border-indigo-400 focus:bg-white focus:outline-none focus:ring-4 focus:ring-indigo-100"
              />
              <span className={`mt-1.5 block text-right text-[11px] tabular-nums ${chars > 0 && chars < 200 ? 'text-amber-600' : 'text-slate-400'}`}>
                {chars > 0 && chars < 200 ? `${200 - chars} more characters needed` : `${chars.toLocaleString()} characters`}
              </span>
            </div>
          )}

          <div className="grid gap-4 sm:grid-cols-2">
            <label className="text-sm">
              <span className="mb-1.5 block font-semibold text-slate-800">Model</span>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full rounded-xl border border-slate-200 px-3 py-2 text-sm text-slate-900 focus:border-indigo-400 focus:outline-none focus:ring-4 focus:ring-indigo-100"
              >
                {models.filter((m) => !localOnly || localModels.includes(m.id)).map((m) => (
                  <option key={m.id} value={m.id}>{m.id === 'groq' ? 'Base (Groq)' : m.id === 'adapter' ? 'Local (LoRA)' : m.id === 'local-heavy' ? 'Local heavy (7B)' : m.id}</option>
                ))}
              </select>
            </label>
            <div className="text-sm">
              <span className="mb-1.5 block font-semibold text-slate-800">Privacy</span>
              <button
                type="button"
                role="switch"
                aria-checked={localOnly}
                disabled={!localModels.length}
                onClick={() => setLocalOnly((v) => !v)}
                className="flex w-full items-center gap-3 rounded-xl px-3 py-2 text-left ring-1 ring-slate-200 hover:bg-slate-50 disabled:opacity-50"
              >
                <span className={`relative h-5 w-9 shrink-0 rounded-full transition-colors duration-200 ${localOnly ? 'bg-emerald-500' : 'bg-slate-300'}`}>
                  <span className={`absolute top-0.5 size-4 rounded-full bg-white shadow transition-transform duration-200 ${localOnly ? 'translate-x-[18px]' : 'translate-x-0.5'}`} />
                </span>
                <span className="text-xs text-slate-600">
                  <b className="font-semibold text-slate-800">Keep the file on this computer</b>
                  <br />
                  {localModels.length ? 'only a local model reads it' : 'no local model is set up'}
                </span>
              </button>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 pt-4">
            <p className="max-w-md text-xs leading-relaxed text-slate-500">
              {localOnly
                ? 'Nothing leaves this computer: a local model reads the file, and the IPC and judgment searches run here.'
                : 'With a hosted model the text is sent to that provider (Groq or Gemini) for this analysis. Nothing is stored.'}
            </p>
            {busy ? (
              <button onClick={() => abortRef.current?.abort()} className="rounded-xl bg-slate-200 px-5 py-2.5 text-sm font-semibold text-slate-800 hover:bg-slate-300">
                Stop
              </button>
            ) : (
              <button
                onClick={start}
                disabled={!ready}
                className="inline-flex items-center gap-2 rounded-xl bg-accent px-5 py-2.5 text-sm font-semibold text-on-accent shadow-sm transition-[transform,background-color] duration-200 hover:-translate-y-0.5 hover:bg-accent-hover disabled:translate-y-0 disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none"
              >
                Analyse case
                <svg {...stroke} strokeWidth={2} className="size-4"><path d="M4.5 10h11M11 5.5l4.5 4.5-4.5 4.5" /></svg>
              </button>
            )}
          </div>
        </section>

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
