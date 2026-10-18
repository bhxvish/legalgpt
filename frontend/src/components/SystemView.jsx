import { useEffect, useState } from 'react'
import { systemApi } from '../api.js'
import { stagger } from './motion.jsx'

// The whole system at a glance, for presenting: two rows of stages (how the knowledge is built,
// what every question runs through), one number each, and four headline results. One-line
// descriptions are behind "Show details"; numbers name their source on hover
// (docs/reports/system_results.json; live counts from GET /api/system). Each stage opens its tab.

const fmt = (n) => (n == null ? '—' : Number(n).toLocaleString('en-IN'))
const svg = { viewBox: '0 0 20 20', fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true, className: 'size-5' }

const ICON = {
  judgments: <svg {...svg}><path d="M5 2.75h7l3 3v11.5H5Z" /><path d="M12 2.75v3h3M7.5 9.5h5M7.5 12h5M7.5 14.5h3" /></svg>,
  label: <svg {...svg}><path d="M12.8 3.6 16.4 7.2 9 14.6 4.5 15.5 5.4 11Z" /><path d="m11 5.4 3.6 3.6" /></svg>,
  model: <svg {...svg}><rect x="5" y="5" width="10" height="10" rx="2" /><path d="M8 2.5V5M12 2.5V5M8 15v2.5M12 15v2.5M2.5 8H5M2.5 12H5M15 8h2.5M15 12h2.5" /></svg>,
  library: <svg {...svg}><path d="M4 3.5v13M8 3.5v13" /><path d="m11.5 4 3.8 12" /><path d="M2.5 16.5h15" /></svg>,
  question: <svg {...svg}><path d="M4 4.5h12a1.5 1.5 0 0 1 1.5 1.5v7a1.5 1.5 0 0 1-1.5 1.5H9l-3.5 2.75V14.5H4A1.5 1.5 0 0 1 2.5 13V6A1.5 1.5 0 0 1 4 4.5Z" /></svg>,
  retrieve: <svg {...svg}><circle cx="8.75" cy="8.75" r="5" /><path d="m12.5 12.5 4 4" /></svg>,
  answer: <svg {...svg}><path d="M4 6h12M4 10h12M4 14h7" /><path d="M14 13.5h3v3" /></svg>,
  check: <svg {...svg}><path d="M10 3v14M6 17h8M3.5 6h13M3.5 6 1.5 11a2.2 2.2 0 0 0 4 0ZM16.5 6l-2 5a2.2 2.2 0 0 0 4 0Z" /></svg>,
  prove: <svg {...svg}><path d="M10 2.5 16 5v5c0 3.5-2.6 6.4-6 7.5-3.4-1.1-6-4-6-7.5V5Z" /><path d="m7.25 10 2 2 3.5-4" /></svg>,
}

const BUILD = [
  { id: 'judgments', title: 'Judgments', value: (l) => fmt(l.library_judgments), unit: 'SC judgments', text: 'Supreme Court criminal judgments, 2020-2023, that cite the IPC.' },
  { id: 'label', module: 'M1', title: 'Label', value: (l) => fmt(l.annotated_judgments), unit: 'labelled', text: 'Each sentence gets a role; people check the model, 1 in 10 blind.', tab: 'label' },
  { id: 'model', module: 'M2', title: 'InLegalBERT', value: () => '0.81', unit: 'macro-F1', text: 'Learns the five roles from the labels; re-trained on the checked labels.', tab: 'label' },
  { id: 'library', title: 'Library', value: (l) => fmt(l.judgment_passages), unit: 'passages', text: 'Every passage indexed with its role, ready to be cited.' },
]

const ANSWER = [
  { id: 'question', title: 'Ask', value: () => 'EN · हिं', unit: 'languages', text: 'Plain words are enough; a Hindi question is searched in translation.', tab: 'chat' },
  { id: 'retrieve', module: 'M0', title: 'Retrieve', value: (l) => fmt(l.ipc_chunks), unit: 'IPC chunks', text: 'The closest IPC sections; civil questions are declined.', tab: 'chat' },
  { id: 'answer', module: 'M0 · M3', title: 'Answer', value: () => '[n]', unit: 'every claim cited', text: 'Written only from the sources, by a hosted or a local fine-tuned model.', tab: 'chat' },
  { id: 'check', module: 'M4 · M5', title: 'Check', value: (l) => (l.rule_sections != null ? String(l.rule_sections) : '—'), unit: 'sections as rules', text: 'Sentences matched to sources; facts tested by Prolog rules; BNS numbers added.', tab: 'verify' },
  { id: 'prove', title: 'Prove', value: () => 'ZK', unit: 'signed answer', text: 'A signature, and a zero-knowledge proof of which sources were closest.', tab: 'chat' },
]

function Stage({ s, live, details, i, last }) {
  const Tag = s.tab ? 'a' : 'div'
  return (
    <li className="relative flex min-w-0 animate-rise flex-col lg:flex-1" style={stagger(i, 50)}>
      <Tag
        {...(s.tab ? { href: `#${s.tab}` } : {})}
        title={details ? undefined : s.text}
        className={`group flex h-full flex-col rounded-2xl bg-white px-4 py-3.5 ring-1 ring-slate-200 transition duration-200 ${
          s.tab ? 'outline-none hover:-translate-y-0.5 hover:ring-indigo-300 hover:shadow-[0_10px_28px_-16px_rgb(79_70_229/0.45)] focus-visible:ring-2 focus-visible:ring-indigo-500' : ''
        }`}
      >
        <div className="flex items-center justify-between">
          <span className="text-slate-400 transition-colors group-hover:text-indigo-500">{ICON[s.id]}</span>
          {s.module && <span className="font-mono text-[10px] font-semibold text-indigo-600/80">{s.module}</span>}
        </div>
        <h3 className="mt-3 text-[15px] font-semibold tracking-tight text-slate-900">{s.title}</h3>
        <p className="mt-0.5 flex items-baseline gap-1.5">
          <span className="whitespace-nowrap font-mono text-lg font-semibold tabular-nums tracking-tight text-slate-800">{live ? s.value(live) : '—'}</span>
          <span className="truncate text-xs text-slate-500">{s.unit}</span>
        </p>
        {details && <p className="mt-2 animate-rise text-xs leading-relaxed text-slate-500">{s.text}</p>}
      </Tag>
      {!last && (
        <span aria-hidden="true" className="mx-auto h-3 w-0.5 flow-y lg:absolute lg:-right-3 lg:top-1/2 lg:mx-0 lg:h-0.5 lg:w-3 lg:flow-x" />
      )}
    </li>
  )
}

function Row({ label, stages, live, details, offset = 0 }) {
  return (
    <section>
      <h2 className="mb-2 text-xs font-medium uppercase tracking-[0.12em] text-slate-400">{label}</h2>
      <ol className="flex flex-col lg:flex-row lg:gap-3">
        {stages.map((s, i) => (
          <Stage key={s.id} s={s} live={live} details={details} i={i + offset} last={i === stages.length - 1} />
        ))}
      </ol>
    </section>
  )
}

export default function SystemView() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [details, setDetails] = useState(false)
  useEffect(() => {
    systemApi.overview().then(setData).catch((e) => setError(e.message))
  }, [])
  // the diagram works without the backend: stages then show "—"
  const live = data?.live || (error ? {} : null)
  const headline = data?.results?.headline || []

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 sm:py-12">
        <header className="flex flex-wrap items-end justify-between gap-4">
          <div className="animate-rise">
            <p className="text-xs font-medium uppercase tracking-[0.14em] text-indigo-600">How it works</p>
            <h1 className="mt-2 max-w-2xl text-2xl font-semibold tracking-tight text-balance text-slate-900 sm:text-[2rem] sm:leading-tight">
              From real judgments to cited, checked answers
            </h1>
          </div>
          <label className="inline-flex cursor-pointer select-none items-center gap-2 text-xs font-medium text-slate-500">
            <input type="checkbox" checked={details} onChange={(e) => setDetails(e.target.checked)} className="peer sr-only" />
            <span aria-hidden="true" className="relative h-5 w-9 rounded-full bg-slate-200 ring-1 ring-slate-300 transition peer-checked:bg-accent peer-focus-visible:ring-2 peer-focus-visible:ring-indigo-500 after:absolute after:left-0.5 after:top-0.5 after:size-4 after:rounded-full after:bg-white after:shadow after:transition-transform peer-checked:after:translate-x-4" />
            Show details
          </label>
        </header>
        {error && (
          <p className="mt-4 text-xs text-amber-700">
            {error.includes('404') ? 'Restart the backend to show live counts.' : `Live counts unavailable (${error}).`}
          </p>
        )}

        <div className="mt-8 space-y-6">
          <Row label="Build the knowledge · offline" stages={BUILD} live={live} details={details} />
          <Row label="Answer a question · every time" stages={ANSWER} live={live} details={details} offset={BUILD.length} />
        </div>

        <a
          href="#case"
          className="group mt-6 flex animate-rise items-center justify-between gap-4 rounded-2xl bg-white px-4 py-3 text-sm ring-1 ring-slate-200 transition hover:ring-indigo-300"
          style={stagger(BUILD.length + ANSWER.length, 50)}
        >
          <span className="text-slate-600">
            <b className="font-semibold text-slate-900">Case analysis</b> runs both rows on one case file
            {details && <span className="text-slate-500"> — facts, offences, element checks, similar judgments; never a verdict</span>}
          </span>
          <span className="shrink-0 text-xs font-medium text-indigo-600 group-hover:underline">Open →</span>
        </a>

        {headline.length > 0 && (
          <section className="mt-10 border-t border-slate-200 pt-6" aria-label="Headline results">
            <dl className="grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-4">
              {headline.map((h, i) => (
                <div key={h.label} className="flex animate-rise flex-col" style={stagger(i, 60, 500)} title={`${h.detail}. Source: ${h.source}`}>
                  <dt className="order-2 mt-1 text-xs leading-snug text-slate-500">{h.label}</dt>
                  <dd className="font-mono text-2xl font-semibold tabular-nums tracking-tight text-slate-900 sm:text-3xl">{h.value}</dd>
                  {details && <dd className="order-3 mt-1 text-[11px] leading-snug text-slate-400">{h.detail}</dd>}
                </div>
              ))}
            </dl>
          </section>
        )}
      </div>
    </div>
  )
}
