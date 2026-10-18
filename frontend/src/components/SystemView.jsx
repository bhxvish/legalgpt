import { useEffect, useState } from 'react'
import { systemApi } from '../api.js'
import { TabIcons } from './LawArt.jsx'
import { stagger } from './motion.jsx'

// The whole system on one page: how the knowledge is built (offline) and what every question runs
// through (online), with each stage's module number from the paper, live counts from this server
// and its headline result (docs/reports/system_results.json). Every stage opens the tab that runs it.

const fmt = (n) => (n == null ? '—' : Number(n).toLocaleString('en-IN'))

const BUILD = [
  {
    id: 'sources',
    title: 'Real judgments',
    text: 'Supreme Court criminal judgments, 2020-2023 (CC-BY-4.0), kept only if they cite the IPC.',
    live: (l) => `${fmt(l.library_judgments)} fetched`,
  },
  {
    id: 'annotate',
    module: 'M1 · M2',
    title: 'Label',
    text: 'InLegalBERT suggests a role for every sentence (Facts, Argument, Law Applied, Precedent, Ruling); people check the unsure ones, and 1 judgment in 10 is labelled blind to measure it.',
    live: (l) => `${fmt(l.annotated_judgments)} judgments labelled`,
    tab: 'label',
  },
  {
    id: 'classifier',
    module: 'M2',
    title: 'InLegalBERT ensemble',
    text: 'Fine-tuned InLegalBERT with a sequence model over each judgment; retrained on the checked labels.',
    live: (l) => (l.classifier ? `deployed: ${l.classifier}` : null),
    tab: 'label',
  },
  {
    id: 'library',
    title: 'Judgment library',
    text: 'Every passage indexed with its role, so the chat cites rulings and Case analysis compares facts.',
    live: (l) => `${fmt(l.judgment_passages)} passages`,
  },
]

const ANSWER = [
  {
    id: 'hindi',
    title: 'Question',
    text: 'In English or Hindi; a Hindi question is searched in English translation and answered in Hindi.',
    tab: 'chat',
  },
  {
    id: 'retrieval',
    module: 'M0',
    title: 'Retrieve',
    text: 'IPC sections by meaning, everyday words mapped to legal terms, an LLM reading of the facts, and a scope check that declines civil matters.',
    live: (l) => `${fmt(l.ipc_chunks)} IPC chunks${l.query_rewrite ? ' · rewrite on' : ''}`,
    tab: 'chat',
  },
  {
    id: 'answer',
    module: 'M0 · M3',
    title: 'Answer with citations',
    text: 'Written only from the numbered sources, every claim cited [n]; hosted models or the LoRA-tuned local model.',
    live: (l) => (l.default_model ? `default: ${l.default_model}` : null),
    tab: 'chat',
  },
  {
    id: 'checks',
    module: 'M4 · M5',
    title: 'Check',
    text: 'Each sentence matched to its source; the facts tested against Prolog rules for the offence; BNS numbers added.',
    live: (l) => (l.rule_sections != null ? `${l.rule_sections} sections as rules · ${fmt(l.bns_sections)} mapped to BNS` : null),
    tab: 'verify',
  },
  {
    id: 'zk',
    title: 'Prove',
    text: 'The answer is signed; a zero-knowledge proof shows the cited sources were the closest matches in the index.',
    tab: 'chat',
  },
]

// stage id -> the results file's stage keys
const RESULT_KEYS = {
  sources: [], annotate: ['annotate'], classifier: ['classifier'], library: ['library'],
  hindi: ['hindi'], retrieval: ['retrieval'], answer: ['answer'], checks: ['explain', 'verify', 'bns'], zk: ['zk'],
}

function Results({ entries }) {
  const shown = entries.filter((e) => e.value && e.value !== 'live' && e.value !== 'per answer')
  if (!shown.length) return null
  return (
    <ul className="mt-2.5 space-y-1.5 border-t border-slate-100 pt-2.5">
      {shown.map((e) => (
        <li key={e.label} className="text-xs leading-snug text-slate-600" title={`${e.detail ? `${e.detail}. ` : ''}Source: ${e.source}`}>
          <span className="mr-1.5 font-mono text-[13px] font-semibold tabular-nums text-slate-900">{e.value}</span>
          {e.label}
          {e.provisional && <span className="ml-1 rounded bg-amber-100 px-1 text-[10px] font-medium text-amber-800">provisional</span>}
        </li>
      ))}
    </ul>
  )
}

function Stage({ s, live, results, i, last }) {
  const Tag = s.tab ? 'a' : 'div'
  const entries = (RESULT_KEYS[s.id] || []).flatMap((k) => results?.stages?.[k] || [])
  const liveText = live && s.live ? s.live(live) : null
  return (
    <li className="relative flex min-w-0 animate-rise flex-col lg:flex-1" style={stagger(i, 70)}>
      <Tag
        {...(s.tab ? { href: `#${s.tab}` } : {})}
        className={`group flex h-full flex-col rounded-2xl bg-white p-3.5 ring-1 ring-slate-200 shadow-[0_1px_2px_rgb(15_23_42/0.04)] transition duration-200 ${
          s.tab ? 'hover:-translate-y-0.5 hover:ring-indigo-300 hover:shadow-[0_8px_24px_-12px_rgb(79_70_229/0.35)] focus-visible:ring-2 focus-visible:ring-indigo-500 outline-none' : ''
        }`}
      >
        <div className="mb-1.5 flex items-center gap-2">
          {s.module && <span className="rounded-md bg-indigo-50 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-indigo-700 ring-1 ring-indigo-200">{s.module}</span>}
          <h3 className="text-sm font-semibold tracking-tight text-slate-900">{s.title}</h3>
          {s.tab && (
            <span className="ml-auto text-slate-300 transition group-hover:translate-x-0.5 group-hover:text-indigo-500" aria-hidden="true">
              {TabIcons[s.tab]}
            </span>
          )}
        </div>
        <p className="text-xs leading-relaxed text-slate-600">{s.text}</p>
        {liveText && (
          <p className="mt-2 inline-flex items-center gap-1.5 self-start rounded-md bg-slate-50 px-1.5 py-0.5 font-mono text-[11px] text-slate-500 ring-1 ring-slate-200">
            <span className="size-1.5 rounded-full bg-emerald-500" aria-hidden="true" />
            {liveText}
          </p>
        )}
        <Results entries={entries} />
        {s.tab && <span className="sr-only">Open the {s.tab} tab</span>}
      </Tag>
      {!last && (
        <span aria-hidden="true" className="mx-auto h-4 w-0.5 flow-y lg:absolute lg:-right-3 lg:top-1/2 lg:mx-0 lg:h-0.5 lg:w-3 lg:flow-x" />
      )}
    </li>
  )
}

function Lane({ title, kicker, stages, live, results, offset = 0 }) {
  return (
    <section className="rounded-3xl bg-slate-100/60 p-3 ring-1 ring-slate-200/70 sm:p-4">
      <header className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1 px-1">
        <h2 className="text-base font-semibold tracking-tight text-slate-900">{title}</h2>
        <p className="text-xs text-slate-500">{kicker}</p>
      </header>
      <ol className="flex flex-col gap-0 lg:flex-row lg:gap-3">
        {stages.map((s, i) => (
          <Stage key={s.id} s={s} live={live} results={results} i={i + offset} last={i === stages.length - 1} />
        ))}
      </ol>
    </section>
  )
}

export default function SystemView() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    systemApi.overview().then(setData).catch((e) => setError(e.message))
  }, [])
  const live = data?.live
  const results = data?.results
  const caseResults = results?.stages?.case || []

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 sm:py-8">
        <div className="mb-6 max-w-3xl animate-rise">
          <p className="text-xs font-medium uppercase tracking-[0.14em] text-indigo-600">How LegalGPT fits together</p>
          <h1 className="mt-1.5 text-2xl font-semibold tracking-tight text-balance text-slate-900 sm:text-3xl">
            One system: knowledge built from real judgments, and answers that are cited, checked and provable
          </h1>
          <p className="mt-2 text-sm leading-relaxed text-pretty text-slate-600">
            The top row runs offline and builds what the system knows. The bottom row runs for every question. Case
            analysis uses both. M0-M5 are the modules in the paper; numbers marked live come from this server, the
            others from the evaluations (hover a result for its source).
          </p>
          {error && (
            <p className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800 ring-1 ring-amber-200">
              {error.includes('404')
                ? 'Live counts need a newer backend: restart it to load GET /api/system. The diagram still applies.'
                : `Live counts unavailable (${error}); the diagram still applies.`}
            </p>
          )}
        </div>

        <div className="space-y-3">
          <Lane title="Build the knowledge" kicker="offline, once per batch of judgments" stages={BUILD} live={live} results={results} />

          {/* the library feeds the answering side: judgment citations and similar cases */}
          <div aria-hidden="true" className="flex items-center justify-center gap-3 py-0.5 text-[11px] font-medium text-slate-500">
            <span className="h-6 w-0.5 flow-y" />
            <span className="rounded-full bg-white px-2.5 py-1 ring-1 ring-slate-200">judgment passages with their roles feed every answer</span>
            <span className="h-6 w-0.5 flow-y" />
          </div>

          <Lane title="Answer a question" kicker="online, every question" stages={ANSWER} live={live} results={results} offset={BUILD.length} />

          <a
            href="#case"
            className="group block animate-rise rounded-3xl bg-white p-4 ring-1 ring-slate-200 transition hover:ring-indigo-300 hover:shadow-[0_8px_24px_-12px_rgb(79_70_229/0.35)] sm:p-5"
            style={stagger(BUILD.length + ANSWER.length, 70)}
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-slate-400 group-hover:text-indigo-500">{TabIcons.case}</span>
              <h2 className="text-base font-semibold tracking-tight text-slate-900">Case analysis: the whole pipeline on one case file</h2>
              <span className="ml-auto text-xs font-medium text-indigo-600 group-hover:underline">Open Case analysis →</span>
            </div>
            <p className="mt-1.5 max-w-4xl text-xs leading-relaxed text-slate-600">
              Facts quoted from the file and checked against it → offences with their IPC text (retrieval, M0) → elements tested by the
              rule engine (M5) → similar past judgments from the library (M2) → a cited analysis with its explanation (M4), passed through
              an output guard. It never says whether anyone is guilty.
            </p>
            <Results entries={caseResults} />
          </a>
        </div>
        <p className="mt-5 text-[11px] text-slate-400">
          Results: docs/reports/system_results.json (each names its source in docs/reports/paper_analysis.md or the report it comes from).
        </p>
      </div>
    </div>
  )
}
