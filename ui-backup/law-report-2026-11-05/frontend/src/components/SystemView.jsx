import { useEffect, useState } from 'react'
import { systemApi } from '../api.js'
import { stagger } from './motion.jsx'

// The front page, for presenting. A real statute with the elements the rule engine tests, then the
// system as a table of contents (how the knowledge is built, what every question runs through), then
// the headline results. Numbers name their source on hover (docs/reports/system_results.json; live
// counts from GET /api/system). Each stage opens its tab.

const fmt = (n) => (n == null ? '—' : Number(n).toLocaleString('en-IN'))

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

// Section 378 as in the corpus, split where the elements of theft fall. The element text is the rule
// engine's own wording for s.379 (backend Prolog rules), in the order the clauses come.
const STATUTE = [
  'Whoever, ',
  { n: 1, text: 'intending to take dishonestly' },
  ' any ',
  { n: 2, text: 'movable property' },
  ' ',
  { n: 3, text: 'out of the possession of any person' },
  ' ',
  { n: 4, text: 'without that person’s consent' },
  ', ',
  { n: 5, text: 'moves that property' },
  ' in order to such taking, is said to commit theft.',
]
const ELEMENTS = [
  'The accused intended to take it dishonestly',
  'The property was movable property',
  'It was taken out of the possession of another person',
  'It was taken without that person’s consent',
  'The property was moved in order to take it',
]

function Leaf() {
  const [hot, setHot] = useState(null)
  const hover = (n) => ({ onPointerEnter: () => setHot(n), onPointerLeave: () => setHot(null) })
  return (
    <figure className="animate-rise bg-white px-6 pb-5 pt-4 ring-1 ring-slate-200 sm:px-8 sm:pb-6 sm:pt-5" style={stagger(3, 80)}>
      <figcaption className="flex items-baseline justify-between border-b border-slate-200 pb-2 font-serif text-[13px] italic text-slate-500">
        <span>The Indian Penal Code, 1860</span>
        <span>Of offences against property</span>
      </figcaption>
      <blockquote className="mt-4 font-serif text-[1.3rem] leading-[1.6] text-slate-800 sm:text-[1.4rem]">
        <span className="font-semibold text-indigo-600">378.</span> <span className="font-semibold text-slate-900">Theft.</span>{' '}
        {STATUTE.map((part, i) =>
          typeof part === 'string' ? (
            part
          ) : (
            <span
              key={i}
              {...hover(part.n)}
              className={`animate-rule bg-[linear-gradient(var(--color-indigo-500),var(--color-indigo-500))] bg-[length:100%_1.5px] bg-[position:0_92%] bg-no-repeat transition-colors duration-200 ${
                hot === part.n ? 'bg-indigo-100 text-slate-950' : ''
              }`}
              style={{ animationDelay: `${500 + part.n * 170}ms`, boxDecorationBreak: 'clone', WebkitBoxDecorationBreak: 'clone' }}
            >
              {part.text}
              <sup className="ml-px font-sans text-[0.6em] font-semibold text-indigo-600">{part.n}</sup>
            </span>
          ),
        )}
      </blockquote>
      <div className="mt-5 border-t border-slate-200 pt-3">
        <p className="text-[13px] font-medium text-slate-900">What the rule engine tests for theft</p>
        <ol className="mt-1.5 grid gap-x-5 text-[13px] leading-snug text-slate-600 sm:grid-cols-2">
          {ELEMENTS.map((e, i) => (
            <li
              key={e}
              {...hover(i + 1)}
              className={`flex gap-2 py-[3px] transition-colors duration-200 ${hot === i + 1 ? 'text-slate-950' : ''}`}
            >
              <span className="w-2.5 shrink-0 text-right font-semibold text-indigo-600">{i + 1}</span>
              {e}
            </li>
          ))}
        </ol>
      </div>
    </figure>
  )
}

function Stage({ s, live, n, i }) {
  const Tag = s.tab ? 'a' : 'div'
  return (
    <li className="animate-rise border-b border-slate-200" style={stagger(i, 40, 200)}>
      <Tag {...(s.tab ? { href: `#${s.tab}` } : {})} className="group grid grid-cols-[1.75rem_minmax(0,1fr)] py-3.5 outline-none focus-visible:bg-slate-100">
        <span className="pt-0.5 font-serif text-[15px] italic text-slate-400">{n}</span>
        <div className="min-w-0">
          <div className="flex items-baseline gap-2">
            <h3 className={`shrink-0 text-[1.3rem] leading-tight text-slate-900 ${s.tab ? 'transition-colors group-hover:text-indigo-600' : ''}`}>{s.title}</h3>
            {s.module && <span className="hidden shrink-0 font-mono text-[10px] text-slate-400 sm:inline">{s.module}</span>}
            <span aria-hidden="true" className="leader h-[0.7em] min-w-4 flex-1" />
            <span className="shrink-0 whitespace-nowrap font-serif text-[1.3rem] leading-tight text-slate-900 [font-variant-numeric:lining-nums_tabular-nums]">{live ? s.value(live) : '—'}</span>
            <span className="w-[6.75rem] shrink-0 truncate text-xs text-slate-500">{s.unit}</span>
          </div>
          <p className="mt-1 text-sm leading-snug text-slate-500">{s.text}</p>
        </div>
      </Tag>
    </li>
  )
}

function Part({ number, title, note, stages, live, offset = 0 }) {
  return (
    <section>
      <header className="border-b border-rule pb-2">
        <p className="kicker">{number}</p>
        <h2 className="mt-0.5 flex flex-wrap items-baseline justify-between gap-x-3 text-[1.625rem] leading-tight text-slate-900">
          {title}
          <span className="font-sans text-xs font-normal tracking-normal text-slate-500">{note}</span>
        </h2>
      </header>
      <ol>
        {stages.map((s, i) => (
          <Stage key={s.id} s={s} live={live} n={i + 1 + offset} i={i + offset} />
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
  // the page works without the backend: stages then show "—"
  const live = data?.live || (error ? {} : null)
  const headline = data?.results?.headline || []

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-6xl px-4 pb-14 pt-10 sm:px-6 sm:pt-16">
        <header className="grid items-start gap-x-14 gap-y-10 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
          <div>
            <p className="kicker animate-rise">Legal research on the Indian Penal Code</p>
            <h1 className="mt-3 animate-rise text-balance font-serif text-[clamp(2.75rem,1.9rem+3.6vw,4.75rem)] font-normal leading-[0.98] tracking-[-0.028em] text-slate-900" style={stagger(1, 80)}>
              From real judgments to cited, checked answers
            </h1>
            <p className="standfirst mt-6 animate-rise" style={stagger(2, 80)}>
              Answers come from the statute text, with a source for every claim, fact checks and verifiable proofs. It is legal
              information, not legal advice.
            </p>
            <p className="mt-8 flex animate-rise flex-wrap items-center gap-x-6 gap-y-3" style={stagger(3, 80)}>
              <a href="#chat" className="inline-flex h-11 items-center rounded-lg bg-accent px-5 text-[15px] font-medium text-on-accent hover:bg-accent-hover">
                Ask a question
              </a>
              <a href="#case" className="group text-[15px] font-medium text-slate-900 underline decoration-slate-300 decoration-1 underline-offset-4 hover:decoration-indigo-600">
                Analyse a case file <span aria-hidden="true" className="inline-block transition-transform duration-200 group-hover:translate-x-0.5">→</span>
              </a>
            </p>
          </div>
          <Leaf />
        </header>

        {error && (
          <p className="mt-6 text-xs text-amber-700">
            {error.includes('404') ? 'Restart the backend to show live counts.' : `Live counts unavailable (${error}).`}
          </p>
        )}

        <div className="mt-16 grid gap-x-14 gap-y-12 sm:mt-20 lg:grid-cols-2">
          <Part number="Part I" title="Building the knowledge" note="offline" stages={BUILD} live={live} />
          <Part number="Part II" title="Answering a question" note="every time" stages={ANSWER} live={live} offset={BUILD.length} />
        </div>

        <a href="#case" className="group mt-10 flex animate-rise flex-wrap items-baseline justify-between gap-x-6 gap-y-1 border-y border-rule py-4" style={stagger(9, 40, 200)}>
          <span className="font-serif text-[1.25rem] leading-snug text-slate-800">
            <span className="text-slate-900 transition-colors group-hover:text-indigo-600">Case analysis</span> runs both parts on one case file
            <span className="text-slate-500">: facts, offences, element checks, similar judgments; never a verdict.</span>
          </span>
          <span aria-hidden="true" className="text-slate-900 transition-transform duration-200 group-hover:translate-x-1">→</span>
        </a>

        {headline.length > 0 && (
          <section className="mt-12" aria-label="Headline results">
            <p className="kicker">Results</p>
            <dl className="mt-3 grid grid-cols-2 gap-y-8 lg:grid-cols-4">
              {headline.map((h, i) => (
                <div key={h.label} className={`flex animate-rise flex-col pr-5 ${i % 2 ? 'border-l border-slate-200 pl-5' : ''} lg:border-l lg:border-slate-200 lg:pl-5 lg:first:border-l-0 lg:first:pl-0`} style={stagger(i, 60, 400)} title={`${h.detail}. Source: ${h.source}`}>
                  <dd className="whitespace-nowrap font-serif text-[clamp(2.25rem,1.8rem+1.6vw,3.25rem)] font-normal leading-none tracking-[-0.02em] text-slate-900 [font-variant-numeric:lining-nums_tabular-nums]">{h.value}</dd>
                  <dt className="mt-2 text-sm font-medium leading-snug text-slate-700">{h.label}</dt>
                  <dd className="mt-1 text-xs leading-snug text-slate-500">{h.detail}</dd>
                </div>
              ))}
            </dl>
          </section>
        )}
      </div>
    </div>
  )
}
