import { CourtFront } from './LegalArt.jsx'
import { useCallback, useEffect, useState } from 'react'
import { annotationApi, reviewApi } from '../api.js'
import { CaseAnnotator, Guideline } from './AnnotateView.jsx'
import { stagger } from './motion.jsx'
import { CaseReview, UploadPanel } from './ReviewView.jsx'
import { Stamp } from './LegalArt.jsx'

// One place to label judgments. Every uploaded judgment is labelled by InLegalBERT first; a person
// then either checks its suggestions (the normal way) or labels it blind, suggestions hidden:
//   - blind samples: about 1 judgment in 10, picked by a hash of its id, labelled blind first, so
//     the model's accuracy is measured on labels it did not influence;
//   - judgments sent back from checking (too many spot checks were wrong);
//   - any judgment a second person labels independently (agreement between people).

const stroke = { viewBox: '0 0 20 20', fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true }

function readStored(key) {
  try {
    return localStorage.getItem(key) || ''
  } catch {
    return ''
  }
}

function writeStored(key, value) {
  try {
    localStorage.setItem(key, value)
  } catch {
    /* storage unavailable */
  }
}

const done = (p) => p && p.sentences > 0 && p.labelled === p.sentences

// the colour of a volume's spine says what it needs
const SPINE = {
  check: 'bg-field-gold text-ink',
  blind: 'bg-field-blue text-paper',
  sent: 'bg-field-red text-ink',
  corpus: 'bg-field-green text-paper',
  hand: 'bg-field-bone text-ink',
  unlabelled: 'bg-field-panel-2 text-paper',
}
const ACTION = { check: 'Check the suggestions', blind: 'Label it blind', sent: 'Label it by hand', corpus: 'Read the signed-off file', hand: 'Open and label', unlabelled: 'Start labelling' }

/** What fits on a spine: no date, no shouting, the first party against the second. */
function spineTitle(title) {
  const t = title.replace(/\s*\(.*$/, '').replace(/^\d+\s+/, '').trim()
  const calm = t === t.toUpperCase() ? t.toLowerCase().replace(/\b[a-z]/g, (m) => m.toUpperCase()).replace(/\b(Versus|Vs|V)\b/g, 'v') : t
  return calm.replace(/\s*@.*?(?=\s+(v|vs|versus)\b)/i, '').replace(/\s+(versus|vs\.?)\s+/i, ' v ').replace(/\s*&\s*Ors\.?/i, '').slice(0, 30)
}

/** One judgment as a volume on the shelf: taller for a longer judgment, lifted when pulled out. */
function Volume({ c, on, onPick, i }) {
  const size = c.queue?.sentences || c.progress?.sentences || 90
  const height = 10.5 + Math.min(1, size / 260) * 5
  return (
    <li className="relative flex min-w-[1.65rem] max-w-12 flex-1 animate-rise items-end" style={stagger(Math.min(i, 30), 18, 120)}>
      <button
        type="button"
        onClick={() => onPick(c.case_id)}
        aria-pressed={on}
        title={c.title || c.case_id}
        aria-label={`${c.title || c.case_id}. ${c.state.text}`}
        className={`group flex w-full flex-col items-center justify-between overflow-hidden px-0.5 pb-1.5 pt-2.5 outline-offset-2 transition-transform duration-300 ease-[cubic-bezier(0.22,1,0.36,1)] ${SPINE[c.state.key]} ${on ? '-translate-y-5' : 'hover:-translate-y-2.5'}`}
        style={{ height: `${height}rem` }}
      >
        <span aria-hidden="true" className="h-[3px] w-full shrink-0 bg-current opacity-50" />
        <span className="min-h-0 flex-1 overflow-hidden whitespace-nowrap py-1.5 text-[13px] font-medium leading-none [writing-mode:vertical-rl]">{spineTitle(c.title || c.case_id)}</span>
        <span className="shrink-0 text-[11px] font-semibold tabular-nums">{String(c.decision_year || '').slice(-2)}</span>
      </button>
    </li>
  )
}

/** The volume pulled from the shelf, open on paper. */
function OpenVolume({ c, onOpen, onBlind }) {
  const qd = c.queue
  const parts = qd?.flagged ? [qd.reviewed, qd.flagged, 'checked'] : c.progress?.sentences ? [c.progress.labelled, c.progress.sentences, 'labelled by you'] : null
  const by = (c.labelled_by || []).map((n) => (n.startsWith('model:') ? 'InLegalBERT' : n)).filter((n, i, a) => a.indexOf(n) === i)
  return (
    <article key={c.case_id} className="light-scope grid animate-sheet items-center gap-x-8 gap-y-4 bg-white p-5 text-ink sm:p-7 lg:grid-cols-[minmax(0,1fr)_auto]" data-testid="open-volume" style={{ animationDuration: '380ms' }}>
      <div className="min-w-0">
        <h3 className="text-[clamp(1.5rem,1.2rem+1.3vw,2.25rem)] leading-[1.05] tracking-[-0.035em]">{c.title || c.case_id}</h3>
        <p className="mt-2 text-[16px] leading-snug text-slate-700">
          {[c.court, c.decision_year, c.sections_cited?.length ? `ss. ${c.sections_cited.slice(0, 6).join(', ')}` : ''].filter(Boolean).join(', ') || c.case_id}
        </p>
        {by.length > 0 && <p className="mt-0.5 text-[15px] text-slate-600">Labelled by {by.join(', ')}</p>}
        {parts && (
          <p className="mt-3 flex items-center gap-3 text-[15px] font-medium">
            <span aria-hidden="true" className="block h-2 w-40 bg-slate-200">
              <span className="block h-full origin-left bg-ink transition-transform duration-700 ease-[cubic-bezier(0.22,1,0.36,1)]" style={{ transform: `scaleX(${parts[1] ? parts[0] / parts[1] : 0})` }} />
            </span>
            {parts[0]} of {parts[1]} {parts[2]}
          </p>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-3 lg:flex-col lg:items-end">
        <Stamp className={{ check: 'text-amber-700', blind: 'text-indigo-700', sent: 'text-red-700', corpus: 'text-emerald-700', hand: 'text-slate-700', unlabelled: 'text-slate-700' }[c.state.key]}>{c.state.text}</Stamp>
        <button type="button" onClick={onOpen} className="press h-12 bg-ink px-6 text-[17px] font-medium text-paper hover:bg-field-red hover:text-ink" data-testid="open-judgment">
          {ACTION[c.state.key]} <span aria-hidden="true">→</span>
        </button>
        {c.state.mode === 'review' && (
          <button
            type="button"
            onClick={onBlind}
            title="Label this judgment yourself with the suggestions hidden: for agreement between two people"
            className="press text-[15px] font-medium text-slate-700 underline underline-offset-2 hover:text-slate-950"
          >
            Label it blind instead
          </button>
        )}
      </div>
    </article>
  )
}

/** What a judgment needs from this person, and how it opens. */
function stateOf(c) {
  const q = c.queue
  // not through InLegalBERT's queue: the original labelled collection (or not labelled at all)
  // an older backend sends no labelled_by: say nothing rather than "Not labelled"
  if (!q && c.labelled_by === undefined) return { key: 'hand', text: 'In the label store', mode: 'blind', tone: 'bg-slate-50 text-slate-600 ring-slate-200' }
  if (!q) return c.labelled_by.length
    ? { key: 'hand', text: 'Labelled', mode: 'blind', tone: 'bg-emerald-50 text-emerald-700 ring-emerald-200' }
    : { key: 'unlabelled', text: 'Not labelled', mode: 'blind', tone: 'bg-amber-50 text-amber-800 ring-amber-200' }
  if (q.status === 'escalated') return { key: 'sent', text: 'Sent back · label by hand', mode: 'blind', tone: 'bg-amber-50 text-amber-800 ring-amber-200' }
  if (q.status === 'promoted') return { key: 'corpus', text: 'Checked · in the corpus', mode: 'review', tone: 'bg-emerald-50 text-emerald-700 ring-emerald-200' }
  if (q.blind_sample && !done(c.progress))
    return { key: 'blind', text: 'Blind sample · label first', mode: 'blind', tone: 'bg-violet-50 text-violet-700 ring-violet-200' }
  return { key: 'check', text: q.blind_sample ? 'Blind labels done · now check' : 'Check suggestions', mode: 'review', tone: 'bg-indigo-50 text-indigo-700 ring-indigo-200' }
}

function HowItWorks() {
  const steps = [
    ['Upload', 'A judgment (.txt or .pdf). InLegalBERT labels every sentence with its role.'],
    ['Check', 'You look only at the sentences it is unsure about, plus a few spot checks of the ones it is sure about.'],
    ['Blind sample', 'About 1 judgment in 10 is labelled with the suggestions hidden first, to measure how accurate the model really is.'],
    ['Corpus', 'Checked judgments join the training data and the judgment library the chat cites.'],
  ]
  return (
    <span className="inline-flex flex-wrap items-center justify-center gap-x-2 font-medium" aria-label="How labelling works">
      {steps.map(([title, text], i) => (
        <span key={title} title={text} className="flex items-center gap-2">
          {i > 0 && <span aria-hidden="true" className="text-paper/50">→</span>}
          {title}
        </span>
      ))}
    </span>
  )
}

export default function LabelView() {
  const [name, setName] = useState(() => readStored('legalgpt.annotator'))
  const [draft, setDraft] = useState(name)
  const [scheme, setScheme] = useState([])
  const [review, setReview] = useState(null) // judgments InLegalBERT labelled (review queue)
  const [collected, setCollected] = useState(null) // every collected judgment, with this person's progress
  const [open, setOpen] = useState(null) // { id, mode: 'review' | 'blind' }
  const [filter, setFilter] = useState('all')
  const [query, setQuery] = useState('')
  const [picked, setPicked] = useState(null) // the volume pulled from the shelf
  const [error, setError] = useState('')

  useEffect(() => {
    annotationApi.scheme().then(setScheme).catch((e) => setError(e.message))
  }, [])

  const load = useCallback(() => {
    if (!name) return
    reviewApi.cases().then(setReview).catch((e) => setError(e.message))
    annotationApi.cases(name).then(setCollected).catch((e) => setError(e.message))
  }, [name])

  useEffect(() => {
    if (!open) load()
  }, [open, load])

  if (!name) {
    return (
      <div className="flex min-h-0 flex-1 flex-col items-center justify-center overflow-y-auto bg-field-black px-4 py-8 text-paper">
        <CourtFront className="mb-6 h-20 w-auto animate-rise text-field-gold" columns={8} />
        <form
          className="light-scope w-full max-w-sm animate-sheet bg-white p-6 text-ink"
          onSubmit={(e) => {
            e.preventDefault()
            if (!draft.trim()) return
            writeStored('legalgpt.annotator', draft.trim())
            setName(draft.trim())
          }}
        >
          <h2 className="head">Who is labelling?</h2>
          <p className="mt-2 text-[15px] leading-snug text-slate-600">Your name is saved with every label and check, so two people's work can be compared.</p>
          <input
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            maxLength={40}
            aria-label="Your name"
            placeholder="e.g. priya"
            className="mt-5 h-12 w-full bg-slate-100 px-3.5 text-[17px] text-slate-900 placeholder:text-slate-500 focus:outline-none"
          />
          <button
            type="submit"
            disabled={!draft.trim()}
            className="press mt-2 h-12 w-full bg-ink text-[17px] font-medium text-paper hover:bg-field-red hover:text-ink disabled:bg-slate-200 disabled:text-slate-500"
          >
            Start labelling
          </button>
        </form>
      </div>
    )
  }

  if (open && scheme.length) {
    return open.mode === 'review' ? (
      <CaseReview caseId={open.id} reviewer={name} scheme={scheme} onBack={() => setOpen(null)} />
    ) : (
      <CaseAnnotator caseId={open.id} annotator={name} scheme={scheme} onProgress={() => {}} onBack={() => setOpen(null)} />
    )
  }

  // one list: every collected judgment, joined with its review-queue entry if InLegalBERT labelled it
  const queue = new Map((review || []).map((c) => [c.case_id, c]))
  const known = new Set((collected || []).map((c) => c.case_id))
  const cases = [
    ...(collected || []).map((c) => ({ ...c, queue: queue.get(c.case_id) })),
    ...[...queue.values()].filter((qc) => !known.has(qc.case_id)).map((qc) => ({ case_id: qc.case_id, title: qc.title, progress: null, queue: qc })),
  ].map((c) => ({ ...c, state: stateOf(c) }))
  const counts = cases.reduce((t, c) => ({ ...t, [c.state.key]: (t[c.state.key] || 0) + 1 }), {})
  const openCase = (id) => {
    const c = cases.find((x) => x.case_id === id)
    setOpen({ id, mode: c ? c.state.mode : queue.get(id)?.blind_sample ? 'blind' : 'review' })
  }
  const FILTERS = [
    ['all', 'All', cases.length],
    ['check', 'To check', counts.check || 0],
    ['blind', 'Blind samples', counts.blind || 0],
    ['sent', 'Sent back', counts.sent || 0],
    ['corpus', 'In the corpus', counts.corpus || 0],
    ['hand', 'Labelled', counts.hand || 0],
    ['unlabelled', 'Not labelled', counts.unlabelled || 0],
  ]
  const q = query.trim().toLowerCase()
  const shown = cases
    .filter((c) => filter === 'all' || c.state.key === filter)
    .filter((c) => !q || [c.title, c.case_id, c.court, String(c.decision_year), ...(c.sections_cited || [])].some((x) => x && String(x).toLowerCase().includes(q)))
  const loading = review === null || collected === null

  const chosen = shown.find((c) => c.case_id === picked) || shown.find((c) => ['check', 'blind', 'sent'].includes(c.state.key)) || shown[0]

  return (
    <div className="min-h-0 flex-1 overflow-y-auto bg-field-black text-paper">
      <div className="mx-auto max-w-[84rem] px-4 py-7 sm:px-8 sm:py-8">
        <header className="text-center">
          <CourtFront className="mx-auto h-14 w-auto animate-rise text-field-gold sm:h-16" columns={8} />
          <h2 className="head mt-3 animate-rise" style={stagger(1)}>The cause list</h2>
          <p className="standfirst mx-auto mt-2 animate-rise" style={stagger(2)}>
            Every judgment on the shelf. InLegalBERT suggests a role for each sentence; you decide.
          </p>
          <p className="mt-2 flex animate-rise flex-wrap items-center justify-center gap-x-5 gap-y-1 text-[15px]" style={stagger(3)}>
            <HowItWorks />
            <span>
              Labelling as <b className="font-semibold">{name}</b>{' '}
              <button
                className="press font-medium text-field-gold underline underline-offset-4"
                onClick={() => {
                  writeStored('legalgpt.annotator', '')
                  setDraft('')
                  setName('')
                }}
              >
                Change
              </button>
            </span>
          </p>
        </header>

        {error && <p className="mt-5 text-center text-[16px] text-field-gold">{error}</p>}

        {loading ? (
          <p className="text-shimmer mt-10 text-center text-lg font-medium">Taking the volumes down…</p>
        ) : cases.length === 0 ? (
          <p className="mt-10 text-center text-[17px]">The shelf is empty. File a judgment below to get started.</p>
        ) : (
          <section className="mt-6" aria-label="Judgments">
            <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3">
              <div className="flex flex-wrap gap-x-1 gap-y-1" role="tablist" aria-label="Show">
                {FILTERS.filter(([k, , n]) => k === 'all' || n > 0).map(([k, label, n]) => (
                  <button
                    key={k}
                    role="tab"
                    aria-selected={filter === k}
                    onClick={() => setFilter(k)}
                    className={`press flex h-10 items-center gap-2 border-b-2 px-2.5 text-[15px] font-medium transition-colors duration-150 ${filter === k ? 'border-field-gold text-paper' : 'border-transparent text-paper/70 hover:text-paper'}`}
                  >
                    {k !== 'all' && <span aria-hidden="true" className={`size-3 ${SPINE[k].split(' ')[0]}`} />}
                    {label}
                    <span className="tabular-nums text-paper/60">{n}</span>
                  </button>
                ))}
              </div>
              <label className="relative w-full sm:w-72">
                <span className="sr-only">Search judgments</span>
                <svg {...stroke} className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-paper/60">
                  <circle cx="8.75" cy="8.75" r="5" />
                  <path d="m12.5 12.5 4 4" />
                </svg>
                <input
                  type="search"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search name, court, year, section"
                  style={{ backgroundColor: 'var(--color-field-panel)', color: 'var(--color-paper)' }}
                  className="h-10 w-full pl-9 pr-2 text-[15px] placeholder:text-paper/60 focus:outline-none"
                />
              </label>
            </div>

            <div className="mt-3 overflow-x-auto pt-5">
              <ol className="flex min-h-[16rem] items-end gap-[3px]" data-testid="label-case-list">
                {shown.map((c, i) => (
                  <Volume key={c.case_id} c={c} i={i} on={chosen?.case_id === c.case_id} onPick={setPicked} />
                ))}
                {shown.length === 0 && <li className="w-full self-center text-center text-[17px]">Nothing here{q ? ` matches "${query}"` : ''}.</li>}
              </ol>
              <div aria-hidden="true" className="h-2.5 min-w-full bg-field-gold" />
              <div aria-hidden="true" className="h-1.5 min-w-full bg-[#b98a2c]" />
            </div>

            {chosen && (
              <div className="mt-5">
                <OpenVolume c={chosen} onOpen={() => setOpen({ id: chosen.case_id, mode: chosen.state.mode })} onBlind={() => setOpen({ id: chosen.case_id, mode: 'blind' })} />
              </div>
            )}
          </section>
        )}

        <div className="mt-5 space-y-3">
          <UploadPanel uploader={name} onFinished={load} onOpen={openCase} />
          {scheme.length > 0 && (
            <details className="group light-scope bg-white p-4 text-ink sm:px-6">
              <summary className="press inline-flex cursor-pointer select-none items-center gap-2 text-[16px] font-medium">
                <span aria-hidden="true" className="text-slate-500 transition-transform duration-200 group-open:rotate-90">›</span>
                What the five labels mean
              </summary>
              <div className="mt-3"><Guideline scheme={scheme} /></div>
            </details>
          )}
        </div>
      </div>
    </div>
  )
}
