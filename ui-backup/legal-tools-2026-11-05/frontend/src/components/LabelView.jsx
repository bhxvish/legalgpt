import { CourtFront } from './LegalArt.jsx'
import { useCallback, useEffect, useState } from 'react'
import { annotationApi, reviewApi } from '../api.js'
import { CaseAnnotator, Guideline, ProgressBar } from './AnnotateView.jsx'
import { stagger } from './motion.jsx'
import { CaseReview, Meter, Pill, UploadPanel } from './ReviewView.jsx'

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
    <ol className="flex flex-wrap items-center justify-center gap-x-2 gap-y-1 text-[15px] font-medium" aria-label="How labelling works">
      {steps.map(([title, text], i) => (
        <li key={title} title={text} className="flex animate-rise items-center gap-2" style={stagger(i, 60, 120)}>
          {i > 0 && <span aria-hidden="true" className="text-paper/50">→</span>}
          {title}
        </li>
      ))}
    </ol>
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
      <div className="relative isolate flex min-h-0 flex-1 items-start justify-center overflow-y-auto px-4 pt-16">
        <form
          className="w-full max-w-sm animate-rise rounded-2xl bg-white p-6  ring-1 ring-slate-200"
          onSubmit={(e) => {
            e.preventDefault()
            if (!draft.trim()) return
            writeStored('legalgpt.annotator', draft.trim())
            setName(draft.trim())
          }}
        >
          <span className="grid size-10 place-items-center rounded-xl bg-indigo-50 text-indigo-600 ring-1 ring-indigo-200">
            <svg {...stroke} className="size-5">
              <circle cx="10" cy="7" r="3.25" />
              <path d="M3.75 16.5a6.25 6.25 0 0 1 12.5 0" />
            </svg>
          </span>
          <h2 className="mt-4 text-lg font-semibold tracking-tight text-slate-900">Who is labelling?</h2>
          <p className="mt-1 text-sm text-slate-500">Your name is saved with every label and check, so two people's work can be compared.</p>
          <input
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            maxLength={40}
            aria-label="Your name"
            placeholder="e.g. priya"
            className="mt-5 w-full rounded-xl border border-slate-200 bg-slate-50/60 px-3.5 py-2.5 text-sm text-slate-900 transition-[border-color,box-shadow,background-color] focus:border-indigo-400 focus:bg-white focus:outline-none focus:ring-4 focus:ring-indigo-100"
          />
          <button
            type="submit"
            disabled={!draft.trim()}
            className="mt-3 w-full rounded-xl bg-accent px-4 py-2.5 text-sm font-semibold text-on-accent shadow-sm transition-[transform,background-color] duration-200 hover:-translate-y-0.5 hover:bg-accent-hover disabled:translate-y-0 disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none"
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

  return (
    <div className="min-h-0 flex-1 overflow-y-auto bg-field-black text-paper">
      <div className="relative mx-auto max-w-4xl px-4 py-8 sm:py-10">
        <CourtFront className="mx-auto h-16 w-auto animate-rise text-field-gold sm:h-20" columns={8} />
        <h2 className="head mt-4 animate-rise text-center" style={stagger(1)}>The cause list</h2>
        <p className="standfirst mx-auto mt-3 animate-rise text-center" style={stagger(2)}>
          Judgments waiting for a person. InLegalBERT suggests a role for every sentence; you decide.
        </p>
        <div className="mt-3"><HowItWorks /></div>
        <p className="mt-2 flex animate-rise items-center justify-center gap-2 text-[15px]" style={stagger(3)}>
          Labelling as <b className="font-semibold">{name}</b>
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
        </p>

        <div className="light-scope mt-7 animate-sheet space-y-4 bg-white p-4 text-ink sm:p-6" style={stagger(4, 80)}>
          <UploadPanel uploader={name} onFinished={load} onOpen={openCase} />
          {scheme.length > 0 && (
            <details className="group text-[15px]">
              <summary className="press inline-flex cursor-pointer select-none items-center gap-2 font-medium underline underline-offset-4">
                What the five labels mean
                <span aria-hidden="true" className="transition-transform duration-200 group-open:rotate-90">→</span>
              </summary>
              <div className="mt-3"><Guideline scheme={scheme} /></div>
            </details>
          )}
        {error && <p className="rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-200">{error}</p>}

        {loading ? (
          <div className="space-y-2">
            {Array.from({ length: 4 }, (_, i) => (
              <span key={i} className="block h-14 animate-pulse bg-slate-100" />
            ))}
          </div>
        ) : cases.length === 0 ? (
          <div className="rounded-2xl bg-white px-5 py-8 text-center text-sm text-slate-600 ring-1 ring-slate-200">
            <p className="font-medium text-slate-800">No judgments yet</p>
            <p className="mt-1">Upload one above to get started.</p>
          </div>
        ) : (
          <section className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Show">
                {FILTERS.filter(([k, , n]) => k === 'all' || n > 0).map(([k, label, n]) => (
                  <button
                    key={k}
                    role="tab"
                    aria-selected={filter === k}
                    onClick={() => setFilter(k)}
                    className={`rounded-full px-2.5 py-1 text-xs font-medium ring-1 transition ${
                      filter === k ? 'bg-accent text-on-accent ring-transparent' : 'bg-white text-slate-600 ring-slate-200 hover:text-slate-900'
                    }`}
                  >
                    {label} <span className="tabular-nums opacity-70">{n}</span>
                  </button>
                ))}
              </div>
              <label className="relative w-full sm:w-56">
                <span className="sr-only">Search judgments</span>
                <svg {...stroke} className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-slate-400">
                  <circle cx="8.75" cy="8.75" r="5" />
                  <path d="m12.5 12.5 4 4" />
                </svg>
                <input
                  type="search"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search name, court, year, section"
                  className="w-full rounded-lg border border-slate-200 bg-white py-1.5 pl-8 pr-2 text-sm text-slate-900 placeholder:text-slate-400 focus:border-indigo-400 focus:outline-none focus:ring-4 focus:ring-indigo-100"
                />
              </label>
            </div>
            <ol data-testid="label-case-list">
              {shown.map((c, i) => {
                const qd = c.queue
                const frac = qd?.flagged ? qd.reviewed / qd.flagged : 1
                return (
                  <li key={c.case_id} className="group/row relative animate-rise" style={stagger(Math.min(i, 12), 40, 200)}>
                    <button
                      onClick={() => setOpen({ id: c.case_id, mode: c.state.mode })}
                      className="group flex w-full flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-1 py-3 text-left transition-colors duration-150 hover:bg-slate-100"
                    >
                      <div className="flex min-w-0 items-center gap-3">
                        <span className="w-8 shrink-0 text-right text-[15px] font-semibold tabular-nums text-indigo-600">{i + 1}.</span>
                        <div className="min-w-0">
                          <p className="truncate text-[17px] font-medium tracking-[-0.01em] text-slate-900">{c.title || qd?.title || c.case_id}</p>
                          <p className="mt-0.5 text-[13px] text-slate-600">
                            {c.state.mode === 'review' && qd
                              ? `${qd.reviewed} of ${qd.flagged} highlighted sentences checked · ${qd.sentences} sentences`
                              : [c.court, c.decision_year, c.sections_cited?.length ? `ss. ${c.sections_cited.slice(0, 4).join(', ')}` : '',
                                  c.labelled_by?.length ? `labelled by ${c.labelled_by.map((n) => (n.startsWith('model:') ? 'InLegalBERT' : n)).filter((n, i, a) => a.indexOf(n) === i).join(', ')}` : '']
                                  .filter(Boolean).join(' · ') || c.case_id}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        {c.state.mode === 'review' && qd ? (
                          <Meter value={frac} tone={frac === 1 ? 'bg-emerald-500' : 'bg-indigo-500'} className="hidden w-20 sm:block" />
                        ) : (
                          c.progress?.labelled > 0 && (
                            <span className="hidden sm:block" title="Your own labels on this judgment">
                              <ProgressBar progress={c.progress} />
                            </span>
                          )
                        )}
                        <Pill label={c.state.text} colour={c.state.tone} />
                        <svg {...stroke} className="size-4 text-slate-300 transition-[transform,color] duration-200 group-hover:translate-x-0.5 group-hover:text-indigo-500">
                          <path d="m7.5 5 5 5-5 5" />
                        </svg>
                      </div>
                    </button>
                    {c.state.mode === 'review' && (
                      // a second, independent labelling of the same judgment (agreement between people)
                      <button
                        onClick={() => setOpen({ id: c.case_id, mode: 'blind' })}
                        title="Label this judgment yourself with the suggestions hidden: for agreement between two people"
                        className="absolute -bottom-2 right-12 rounded-md bg-white px-2 py-0.5 text-[11px] font-medium text-slate-500 opacity-0 ring-1 ring-slate-200 transition hover:text-indigo-700 focus-visible:opacity-100 group-hover/row:opacity-100"
                      >
                        Label blind instead
                      </button>
                    )}
                  </li>
                )
              })}
              {shown.length === 0 && <li className="px-1 py-4 text-sm text-slate-500">Nothing here{q ? ` matches "${query}"` : ''}.</li>}
            </ol>
          </section>
        )}
        </div>
      </div>
    </div>
  )
}
