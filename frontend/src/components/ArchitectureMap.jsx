import { useEffect, useRef, useState } from 'react'
import { useCount, useSeen } from './motion.jsx'

// The architecture as a live map: how the knowledge is built (offline), how every question runs
// through it, and how the two meet. Lines draw themselves as the map scrolls in, then data keeps
// moving along them, and a walk-through visits each stage until you pick one yourself. Wide screens
// get the map, drawn at 1200 x 560 and scaled to fit; narrow ones get the same stages as one track.

const W = 1200
const H = 560
const fmt = (n) => (n == null ? '—' : Number(n).toLocaleString('en-IN'))
const svg = { viewBox: '0 0 20 20', fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true, className: 'size-6' }

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
  explain: <svg {...svg}><circle cx="8.5" cy="8.5" r="5.25" /><path d="m12.5 12.5 4.75 4.75" /><circle cx="8.5" cy="8.5" r="1.4" /></svg>,
  case: <svg {...svg}><path d="M2.75 6.25A1.5 1.5 0 0 1 4.25 4.75h3.5l1.5 1.75h6.5a1.5 1.5 0 0 1 1.5 1.5v6.75a1.5 1.5 0 0 1-1.5 1.5H4.25a1.5 1.5 0 0 1-1.5-1.5Z" /><path d="M7 10.25h6M7 12.75h4" /></svg>,
}

// x, y, w: the tile on the 1200 x 560 map. lane: the colour it takes when it is the current stage.
const STAGES = [
  { id: 'judgments', lane: 'build', x: 40, y: 44, w: 190, title: 'Judgments', value: (l) => fmt(l.library_judgments), unit: 'SC judgments', text: 'Supreme Court criminal judgments, 2020-2023, that cite the IPC.' },
  { id: 'label', lane: 'build', x: 270, y: 44, w: 190, module: 'M1', title: 'Label', value: (l) => fmt(l.annotated_judgments), unit: 'labelled', text: 'Each sentence gets a role; people check the model, 1 in 10 blind.', tab: 'label' },
  { id: 'model', lane: 'build', x: 500, y: 44, w: 190, module: 'M2', title: 'InLegalBERT', value: () => '0.81', unit: 'macro-F1', text: 'Learns the five roles from the labels; re-trained on the checked labels.', tab: 'label' },
  { id: 'library', lane: 'build', x: 730, y: 44, w: 190, title: 'Library', value: (l) => fmt(l.judgment_passages), unit: 'passages', text: 'Every passage indexed with its role, ready to be cited. This is what Retrieve searches.' },
  { id: 'question', lane: 'answer', x: 40, y: 262, w: 190, title: 'Ask', value: () => 'EN · हिं', unit: 'languages', text: 'Plain words are enough; a Hindi question is searched in translation.', tab: 'chat' },
  { id: 'retrieve', lane: 'answer', x: 270, y: 262, w: 190, module: 'M0', title: 'Retrieve', value: (l) => fmt(l.ipc_chunks), unit: 'IPC chunks', text: 'The closest IPC sections and judgment passages; civil questions are declined.', tab: 'chat' },
  { id: 'answer', lane: 'answer', x: 500, y: 262, w: 190, module: 'M0 · M3', title: 'Answer', value: () => '[n]', unit: 'every claim cited', text: 'Written only from the sources, by a hosted or a local fine-tuned model.', tab: 'chat' },
  { id: 'check', lane: 'answer', x: 730, y: 262, w: 190, module: 'M4 · M5', title: 'Check', value: (l) => (l.rule_sections != null ? String(l.rule_sections) : '—'), unit: 'sections as rules', text: 'Sentences matched to sources; facts tested by Prolog rules; BNS numbers added.', tab: 'verify' },
  { id: 'prove', lane: 'answer', x: 960, y: 262, w: 200, title: 'Prove', value: () => 'ZK', unit: 'signed answer', text: 'A signature, and a zero-knowledge proof of which sources were closest.', tab: 'chat' },
  { id: 'explain', lane: 'open', x: 500, y: 452, w: 190, module: 'M4', title: 'Explain', value: () => '6', unit: 'bias attributes', text: 'Reasoning, a faithfulness test, key words, source influence and a bias check on any answer.', tab: 'explain' },
  { id: 'case', lane: 'open', x: 40, y: 452, w: 420, title: 'Case analysis', value: () => 'FIR', unit: 'in, never a verdict', text: 'Runs both rows on one case file: facts, offences, element checks, similar judgments.', tab: 'case' },
]

const LANE = { build: 'bg-field-bone text-ink', answer: 'bg-field-gold text-ink', open: 'bg-field-red text-ink' }
const LANE_STROKE = { build: '#e7e7e5', answer: '#f5b83d', open: '#e8432e' }

const PATHS = [
  { lane: 'build', d: 'M135 94 H825 V216 H365 V262', secs: 7 }, // Judgments to Library, and the library feeds Retrieve
  { lane: 'answer', d: 'M135 312 H1060 V452', secs: 6.5 }, // Ask to Prove, then to you
  { lane: 'open', d: 'M595 362 V452', secs: 2 }, // any answer can be opened
  { lane: 'open', d: 'M595 502 H730', secs: 2 },
  { lane: 'open', d: 'M135 452 V362', secs: 2 }, // a case file enters as a question
]

function Tile({ s, live, index, on, pick, seen, walking }) {
  const value = useCount(live ? s.value(live) : '—', seen)
  return (
    <button
      type="button"
      onClick={() => pick(index)}
      onPointerEnter={() => pick(index)}
      onFocus={() => pick(index)}
      aria-pressed={on}
      className={`map-node press absolute flex h-[100px] flex-col justify-between p-3 text-left ${on ? LANE[s.lane] : 'bg-field-panel text-paper hover:bg-field-panel-2'}`}
      style={{ left: s.x, top: s.y, width: s.w, transitionDelay: seen ? `${index * 90}ms, ${index * 90}ms, 0ms, 0ms` : undefined }}
      data-testid={`stage-${s.id}`}
    >
      <span className="flex items-center justify-between gap-2">
        <span className="text-[1.0625rem] font-semibold leading-none tracking-[-0.02em]">{s.title}</span>
        <span className={on ? '' : 'text-paper/55'}>{ICON[s.id]}</span>
      </span>
      <span className="flex items-baseline gap-1.5">
        <span className="whitespace-nowrap text-[1.625rem] font-semibold leading-none tracking-[-0.03em] [font-variant-numeric:tabular-nums]">{value}</span>
        <span className={`truncate text-[13px] ${on ? '' : 'text-paper/60'}`}>{s.unit}</span>
      </span>
      {on && walking && <span aria-hidden="true" className="absolute inset-x-0 bottom-0 h-1 origin-left animate-[fill_2600ms_linear_both] bg-ink" />}
    </button>
  )
}

function Lines({ calm }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="absolute inset-0 size-full" aria-hidden="true" fill="none">
      {PATHS.map((p, i) => (
        <g key={p.d}>
          <path d={p.d} stroke="#2c2c31" strokeWidth="2" pathLength="1" className="map-line" style={{ transitionDelay: `${i * 140}ms` }} />
          <path d={p.d} stroke={LANE_STROKE[p.lane]} strokeWidth="2" strokeLinecap="round" className="map-flow" />
          {!calm &&
            [0, 1 / 2].map((at) => (
              <rect key={at} x="-5" y="-5" width="10" height="10" fill={LANE_STROKE[p.lane]}>
                <animateMotion dur={`${p.secs}s`} begin={`${at * p.secs}s`} repeatCount="indefinite" path={p.d} />
              </rect>
            ))}
        </g>
      ))}
    </svg>
  )
}

function Map({ live, active, pick, seen, walking }) {
  const box = useRef(null)
  const [k, setK] = useState(1)
  const calm = typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setK(e.contentRect.width / W))
    ro.observe(box.current)
    return () => ro.disconnect()
  }, [])
  return (
    <div ref={box} className="relative mx-auto w-full" style={{ height: H * k, maxWidth: 'max(780px, calc((100svh - 214px) * 2.1428))' }}>
      <div className={`absolute left-0 top-0 origin-top-left ${seen ? 'map-in' : ''}`} style={{ width: W, height: H, transform: `scale(${k})` }}>
        <Lines calm={calm} />
        <p className="absolute left-10 top-3 text-[15px] font-medium text-paper/70">Building the knowledge, offline</p>
        <p className="absolute left-10 top-[231px] text-[15px] font-medium text-paper/70">Answering a question, every time</p>
        <p className="absolute left-[480px] top-[182px] text-[15px] font-medium text-field-bone">The library feeds every search</p>
        {STAGES.map((s, i) => (
          <Tile key={s.id} s={s} live={live} index={i} on={active === i} pick={pick} seen={seen} walking={walking} />
        ))}
        <a
          href="#chat"
          className="map-node press absolute flex h-[100px] flex-col justify-between bg-paper p-3 text-ink hover:bg-field-gold"
          style={{ left: 730, top: 452, width: 430, transitionDelay: seen ? `${STAGES.length * 90}ms, ${STAGES.length * 90}ms, 0ms, 0ms` : undefined }}
        >
          <span className="text-[1.0625rem] font-semibold leading-none tracking-[-0.02em]">You get an answer you can open</span>
          <span className="flex items-baseline justify-between text-[15px]">
            Cited, checked, signed, explained
            <span aria-hidden="true" className="text-xl leading-none">→</span>
          </span>
        </a>
      </div>
    </div>
  )
}

function Track({ live, active, pick }) {
  const groups = [['build', 'Building the knowledge, offline'], ['answer', 'Answering a question, every time'], ['open', 'Opening it up']]
  return (
    <div className="space-y-8">
      {groups.map(([lane, title]) => (
        <section key={lane}>
          <h3 className="mb-3 text-[15px] font-medium text-paper/70">{title}</h3>
          <ol>
            {STAGES.map((s, i) =>
              s.lane !== lane ? null : (
                <li key={s.id}>
                  <button
                    type="button"
                    onClick={() => pick(i)}
                    aria-expanded={active === i}
                    className={`press block w-full p-3.5 text-left ${active === i ? LANE[lane] : 'bg-field-panel text-paper'}`}
                  >
                    <span className="flex items-baseline justify-between gap-3">
                      <span className="text-lg font-semibold tracking-[-0.02em]">{s.title}</span>
                      <span className="flex items-baseline gap-1.5">
                        <span className="text-lg font-semibold">{live ? s.value(live) : '—'}</span>
                        <span className={`text-[13px] ${active === i ? '' : 'text-paper/60'}`}>{s.unit}</span>
                      </span>
                    </span>
                    {active === i && (
                      <span className="mt-1.5 block animate-fade text-[15px] leading-snug">
                        {s.text}
                        {s.tab && <a href={`#${s.tab}`} className="mt-1 block font-medium underline underline-offset-2">Open {s.tab}</a>}
                      </span>
                    )}
                  </button>
                  <span aria-hidden="true" className="mx-auto block h-3 w-0.5 flow-y text-paper/40" />
                </li>
              ),
            )}
          </ol>
        </section>
      ))}
    </div>
  )
}

export default function ArchitectureMap({ live }) {
  const root = useRef(null)
  const seen = useSeen(root, '0px 0px -25% 0px')
  const [active, setActive] = useState(0)
  const [held, setHeld] = useState(false) // a stage was picked: the walk-through stops
  const [wide, setWide] = useState(() => window.matchMedia('(min-width: 900px)').matches)

  useEffect(() => {
    const mq = window.matchMedia('(min-width: 900px)')
    const on = () => setWide(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])

  useEffect(() => {
    if (!seen || held || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const t = setInterval(() => setActive((i) => (i + 1) % STAGES.length), 2600)
    return () => clearInterval(t)
  }, [seen, held])

  const pick = (i) => {
    setActive(i)
    setHeld(true)
  }
  const s = STAGES[active]

  return (
    <section ref={root} data-ui className="flex min-h-full flex-col justify-center bg-field-black px-4 py-12 text-paper sm:px-8 sm:py-7" aria-label="System architecture">
      <div className="mx-auto w-full max-w-[84rem]">
        <div className="grid items-start gap-x-10 gap-y-4 lg:min-h-28 lg:grid-cols-[minmax(0,6fr)_minmax(0,7fr)]">
          <h2 className={`reveal text-[clamp(2rem,1.2rem+2.6vw,3.25rem)] leading-none tracking-[-0.04em] ${seen ? 'is-in' : ''}`}>How an answer gets made</h2>
          {wide && (
            <div aria-live="polite">
              <p key={s.id} className="animate-rise text-[clamp(1.25rem,1.1rem+0.7vw,1.625rem)] leading-tight tracking-[-0.02em]" style={{ animationDuration: '420ms' }}>
                <b className="font-semibold text-field-gold">{s.title}.</b> {s.text}
              </p>
              <p className="mt-2 flex flex-wrap items-center gap-x-6 gap-y-1 text-[15px]">
                {s.tab && (
                  <a href={`#${s.tab}`} className="font-medium text-paper underline underline-offset-4 transition-colors duration-200 hover:text-field-gold">
                    Open {s.tab}
                  </a>
                )}
                <button type="button" onClick={() => setHeld((h) => !h)} className="press text-paper/70 hover:text-paper">
                  {held ? 'Resume the walk-through' : 'Pause the walk-through'}
                </button>
              </p>
            </div>
          )}
        </div>
        <div className="mt-3">
          {wide ? <Map live={live} active={active} pick={pick} seen={seen} walking={!held} /> : <Track live={live} active={active} pick={pick} />}
        </div>
      </div>
    </section>
  )
}
