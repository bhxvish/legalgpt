import { useEffect, useRef, useState } from 'react'
import { systemApi } from '../api.js'
import { sendTo } from '../handoff.js'
import ArchitectureMap from './ArchitectureMap.jsx'
import { CourtFront, Scales } from './LegalArt.jsx'
import { Reveal, useCount, useSeen } from './motion.jsx'

// The front page, in chapters. Red: what this is. Black: the live architecture map. Gold: change one
// fact of the Code's own illustration and watch the law answer. Green: what it covers. Bone: the
// results, with their sources on hover (docs/reports/system_results.json; live counts from GET
// /api/system). Blue: ask. The page works without the backend: counts then show "—".

// Built on section 378's own illustration (a tree on Z's ground). Each fact is an element the rule
// engine tests for theft, in its own wording, and the sentence of the story when it is there or not.
const FACTS = [
  { id: 'dishonest', element: 'The accused intended to take it dishonestly', yes: 'A means to take it dishonestly.', no: 'A honestly believes it is his own.', missing: 'A had no dishonest intention' },
  { id: 'movable', element: 'The property was movable property', yes: 'He cuts it free of the earth.', no: 'It stays rooted in the earth.', missing: 'a tree attached to the earth is not movable property' },
  { id: 'possession', element: 'It was taken out of the possession of another person', yes: 'It is in Z’s possession.', no: 'It is in no one’s possession.', missing: 'it was in no one’s possession' },
  { id: 'consent', element: 'It was taken without that person’s consent', yes: 'Z has not agreed.', no: 'Z has agreed.', missing: 'Z consented' },
  { id: 'moved', element: 'The property was moved in order to take it', yes: 'A carries it away.', no: 'A never moves it.', missing: 'the property was never moved' },
]

function OneFact() {
  const [on, setOn] = useState(() => Object.fromEntries(FACTS.map((f) => [f.id, true])))
  const gaps = FACTS.filter((f) => !on[f.id])
  const met = gaps.length === 0
  const story = `There is a tree on Z’s ground. ${FACTS.map((f) => (on[f.id] ? f.yes : f.no)).join(' ')}`
  return (
    <section className="bg-field-gold px-4 py-16 text-ink sm:px-8 sm:py-24" aria-label="Change one fact">
      <div className="mx-auto grid max-w-[84rem] gap-x-14 gap-y-10 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <div>
          <Reveal as="h2" className="chapter-title">Change<br />one fact</Reveal>
          <Reveal as="p" delay={160} className="reading mt-8 text-[clamp(1.5rem,1.1rem+1.5vw,2.25rem)] leading-[1.28] text-ink" aria-live="polite">
            There is a tree on Z’s ground.{' '}
            {FACTS.map((f) => (
              <span key={f.id + on[f.id]} className={`animate-fade ${on[f.id] ? '' : 'bg-ink px-1.5 text-paper [box-decoration-break:clone]'}`}>
                {on[f.id] ? f.yes : f.no}{' '}
              </span>
            ))}
          </Reveal>
        </div>
        <Reveal delay={120}>
          <Scales tilt={gaps.length ? -Math.min(14, 6 + gaps.length * 3) : 0} arm={120} drop={70} className="mb-4 h-32 text-ink" />
          <p className="standfirst">Section 378 sets five elements, and the rule engine tests each. Switch a fact off and watch the law answer.</p>
          <ul data-ui className="mt-5 grid gap-1.5">
            {FACTS.map((f) => (
              <li key={f.id}>
                <button
                  type="button"
                  role="switch"
                  aria-checked={on[f.id]}
                  onClick={() => setOn((o) => ({ ...o, [f.id]: !o[f.id] }))}
                  className={`press flex min-h-14 w-full items-center gap-4 px-4 py-2.5 text-left text-[17px] font-medium leading-snug ${on[f.id] ? 'bg-ink text-paper' : 'bg-transparent text-ink shadow-[inset_0_0_0_2px_var(--color-ink)]'}`}
                  data-testid={`fact-${f.id}`}
                >
                  <span aria-hidden="true" className={`relative h-6 w-10 shrink-0 ${on[f.id] ? 'bg-field-gold' : 'bg-ink'}`}>
                    <span className={`absolute left-1 top-1 size-4 transition-transform duration-200 ease-[cubic-bezier(0.22,1,0.36,1)] ${on[f.id] ? 'translate-x-4 bg-ink' : 'bg-field-gold'}`} />
                  </span>
                  {f.element}
                </button>
              </li>
            ))}
          </ul>
          <div className="mt-6" aria-live="polite">
            <p key={met ? 'met' : gaps.map((g) => g.id).join()} className="head animate-rise" style={{ animationDuration: '420ms' }}>
              {met ? 'All five elements met' : `Not theft: ${gaps.length} missing`}
            </p>
            <p className="mt-2 text-[17px] leading-snug">
              {met ? 'Every element of the section is shown by the facts.' : `Because ${gaps[0].missing}.`} This shows how the rules work; it is not legal advice.
            </p>
          </div>
          <button
            type="button"
            onClick={() => sendTo('verify', { text: story, section: '379' })}
            className="press mt-5 inline-flex h-12 items-center bg-ink px-5 text-[17px] font-medium text-paper hover:bg-field-green"
          >
            Check these facts in the rule engine
          </button>
        </Reveal>
      </div>
    </section>
  )
}

const OFFENCES = ['Theft', 'Robbery', 'Cheating', 'Dowry death', 'Criminal breach of trust', 'Causing death by negligence', 'Extortion', 'Kidnapping', 'Forgery', 'Criminal intimidation', 'Mischief', 'Culpable homicide']

function Offences() {
  const row = OFFENCES.map((o) => (
    <span key={o} className="flex items-center">
      {o}
      <span aria-hidden="true" className="mx-[0.35em] inline-block size-[0.22em] rounded-full bg-paper" />
    </span>
  ))
  return (
    <section className="overflow-hidden bg-field-green py-9 text-paper sm:py-12" aria-label="Offences it can read">
      <div className="marquee flex w-max whitespace-nowrap text-[clamp(3.25rem,8vw,7.5rem)] font-semibold leading-none tracking-[-0.04em]" aria-hidden="true">
        {row}
        {row}
      </div>
      <p className="sr-only">{OFFENCES.join(', ')}</p>
      <p className="mt-6 max-w-2xl px-4 text-[17px] leading-snug sm:px-8">The whole Indian Penal Code and the Bharatiya Nyaya Sanhita, with every IPC section matched to its BNS number.</p>
    </section>
  )
}

function Figure({ h, big = false }) {
  const ref = useRef(null)
  const seen = useSeen(ref)
  const value = useCount(h.value, seen)
  return (
    <div ref={ref} title={`${h.detail}. Source: ${h.source}`} className="min-w-0">
      <p className={`whitespace-nowrap font-semibold leading-[0.9] tracking-[-0.05em] [font-variant-numeric:tabular-nums] ${big ? 'text-[clamp(5rem,calc((100vw-var(--rail))*0.2),17rem)]' : 'text-[clamp(3.25rem,calc((100vw-var(--rail))*0.075),6.5rem)]'}`}>
        {value}
      </p>
      <p className={`mt-3 font-medium leading-tight ${big ? 'text-[clamp(1.25rem,1.1rem+0.7vw,1.625rem)]' : 'text-[17px]'}`}>{h.label}</p>
      <p className="mt-1 text-[15px] leading-snug">{h.detail}</p>
    </div>
  )
}

function Results({ headline }) {
  const [first, ...rest] = headline
  return (
    <section className="bg-field-bone px-4 py-16 text-ink sm:px-8 sm:py-24" aria-label="Results">
      <div className="mx-auto max-w-[84rem]">
        <Reveal as="h2" className="head">What was measured</Reveal>
        <Reveal delay={100} className="mt-10"><Figure h={first} big /></Reveal>
        <div className="mt-14 grid gap-x-10 gap-y-10 sm:grid-cols-3">
          {rest.map((h, i) => (
            <Reveal key={h.label} delay={i * 90}><Figure h={h} /></Reveal>
          ))}
        </div>
      </div>
    </section>
  )
}

function Ask() {
  const [q, setQ] = useState('')
  const go = (text) => {
    if (!text.trim()) return
    window.location.hash = 'chat'
    window.dispatchEvent(new CustomEvent('legalgpt:ask', { detail: text.trim() }))
  }
  return (
    <section className="bg-field-blue px-4 py-16 text-paper sm:px-8 sm:py-24" aria-label="Ask a question">
      <div className="mx-auto grid max-w-[84rem] items-end gap-x-14 gap-y-10 lg:grid-cols-2">
        <div>
          <Reveal as="h2" className="chapter-title">Ask it<br />yourself</Reveal>
          <Reveal as="p" delay={100} className="standfirst mt-6">Describe what happened in plain words. You do not need section numbers.</Reveal>
        </div>
        <Reveal delay={160}>
          <form
            onSubmit={(e) => {
              e.preventDefault()
              go(q)
            }}
            className="bg-white p-4 text-slate-900 sm:p-5"
          >
            <label htmlFor="ask-now" className="text-[15px] font-medium">Your question</label>
            <textarea
              id="ask-now"
              rows={3}
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="My neighbour broke into my shed at night and took my bicycle."
              className="mt-2 block w-full resize-none bg-transparent text-[19px] leading-snug placeholder:text-slate-400 focus:outline-none"
            />
            <button type="submit" disabled={!q.trim()} className="press mt-3 h-12 w-full bg-accent text-[17px] font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50">
              Ask LegalGPT
            </button>
          </form>
        </Reveal>
      </div>
    </section>
  )
}

export default function SystemView() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    systemApi.overview().then(setData).catch((e) => setError(e.message))
  }, [])
  const live = data?.live || (error ? {} : null)
  const headline = data?.results?.headline || []
  // the scales on the first screen never quite settle
  const [sway, setSway] = useState(-4)
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return setSway(0)
    const t = setInterval(() => setSway((s) => -s), 2800)
    return () => clearInterval(t)
  }, [])

  return (
    <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden">
      <section className="relative flex min-h-full flex-col justify-between overflow-hidden bg-field-red px-4 pb-8 pt-8 text-ink sm:px-8 sm:pb-10 sm:pt-12">
        <Scales tilt={sway} arm={96} drop={150} className="pointer-events-none absolute -right-[10%] bottom-[25%] w-[74vw] text-field-red-deep md:-right-[6%] md:bottom-auto md:top-[7%] md:h-[66%] md:w-auto" />
        <h1 className="relative font-semibold leading-[0.88] tracking-[-0.05em] text-[20vw] md:text-[clamp(5.25rem,calc((100vw-var(--rail))*0.158),16rem)]" aria-label="The law, checked.">
          <span aria-hidden="true" className="mask-line"><span className="animate-line-up text-paper" style={{ animationDelay: '120ms' }}>The law,</span></span>
          <span aria-hidden="true" className="mask-line"><span className="animate-line-up" style={{ animationDelay: '240ms' }}>checked.</span></span>
        </h1>
        <div className="relative mt-12 flex flex-wrap items-end justify-between gap-x-10 gap-y-6">
          <p className="standfirst animate-rise font-medium" style={{ animationDelay: '480ms' }}>
            Answers on the Indian Penal Code from the statute text, with a source for every claim, fact checks and verifiable proofs. Legal
            information, not legal advice.
          </p>
          <p className="flex animate-rise flex-wrap items-center gap-x-6 gap-y-3" style={{ animationDelay: '600ms' }}>
            <a href="#chat" className="press inline-flex h-14 items-center bg-ink px-7 text-lg font-medium text-paper hover:bg-field-blue">Ask a question</a>
            <a href="#case" className="text-lg font-medium underline decoration-2 underline-offset-4">Analyse a case file</a>
          </p>
        </div>
      </section>

      <ArchitectureMap live={live} />
      {error && (
        <p className="bg-field-black px-4 pb-6 text-[15px] text-field-gold sm:px-8">
          {error.includes('404') ? 'Restart the backend to show live counts.' : `Live counts unavailable (${error}).`}
        </p>
      )}
      <OneFact />
      <Offences />
      {headline.length > 0 && <Results headline={headline} />}
      <Ask />

      <footer className="overflow-hidden bg-field-black px-4 pt-12 text-paper sm:px-8">
        <CourtFront className="mx-auto mb-10 w-full max-w-4xl text-field-panel-2" />
        <p className="flex flex-wrap justify-between gap-x-8 gap-y-2 text-[15px]">
          <span>Legal information, not legal advice. Check answers against the cited sections.</span>
          <a href="#explain" className="underline underline-offset-4">How it explains itself</a>
        </p>
        <p aria-hidden="true" className="mt-6 -mb-[0.2em] whitespace-nowrap font-semibold leading-[0.9] tracking-[-0.055em] text-[calc((100vw-var(--rail))*0.222)]">LegalGPT</p>
      </footer>
    </div>
  )
}

