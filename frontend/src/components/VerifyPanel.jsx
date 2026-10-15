import { useEffect, useRef, useState } from 'react'
import { verifyApi } from '../api.js'
import { stagger } from './motion.jsx'
import { VerdictProof } from './ZkProof.jsx'

const STATUS = {
  CONSISTENT: {
    cls: 'bg-emerald-100 text-emerald-800 ring-emerald-300',
    card: 'bg-emerald-50 ring-emerald-200',
    bar: 'bg-emerald-500',
    icon: 'bg-emerald-500',
    label: 'Consistent',
    text: (s) => `The extracted facts satisfy every encoded element of s.${s}.`,
  },
  INCONSISTENT: {
    cls: 'bg-red-100 text-red-800 ring-red-300',
    card: 'bg-red-50 ring-red-200',
    bar: 'bg-red-500',
    icon: 'bg-red-500',
    label: 'Inconsistent',
    text: (s) => `At least one element of s.${s} is contradicted by the facts.`,
  },
  INSUFFICIENT: {
    cls: 'bg-amber-100 text-amber-800 ring-amber-300',
    card: 'bg-amber-50 ring-amber-200',
    bar: 'bg-amber-500',
    icon: 'bg-amber-500',
    label: 'Insufficient facts',
    text: (s) => `Some elements of s.${s} are not established by the facts (nothing contradicts them either).`,
  },
}

const ELEMENT = {
  satisfied: { cls: 'text-emerald-700', dot: 'bg-emerald-100 text-emerald-700 ring-emerald-200', label: 'satisfied' },
  violated: { cls: 'text-red-700', dot: 'bg-red-100 text-red-700 ring-red-200', label: 'contradicted' },
  missing: { cls: 'text-amber-700', dot: 'bg-amber-100 text-amber-700 ring-amber-200', label: 'not established' },
}

const stroke = { viewBox: '0 0 20 20', fill: 'none', stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true }

function StatusGlyph({ status, className = 'size-3.5' }) {
  if (status === 'satisfied' || status === 'CONSISTENT')
    return (
      <svg {...stroke} className={className}>
        <path d="m4.5 10.5 3.5 3.5 7.5-8" />
      </svg>
    )
  if (status === 'violated' || status === 'INCONSISTENT')
    return (
      <svg {...stroke} className={className}>
        <path d="m5.5 5.5 9 9M14.5 5.5l-9 9" />
      </svg>
    )
  return (
    <svg {...stroke} className={className}>
      <path d="M7.5 7.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .8-1 1.5v.4M10 14.6v.1" />
    </svg>
  )
}

// Illustrative facts written for the demo (not a real case).
const EXAMPLE = {
  section: '304A',
  text:
    'On 3 March 2021 the accused was driving a bus on the Jaipur–Ajmer highway at high speed while talking on his ' +
    'mobile phone. He failed to notice a pedestrian crossing at a marked crossing and struck him; the pedestrian died ' +
    'in hospital the same night. Witnesses said the accused did not know the pedestrian, and the accused immediately ' +
    'stopped and took him to hospital.',
}

const MIN_CHARS = 40

function Pill({ status, section }) {
  const s = STATUS[status]
  return (
    <span className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-semibold ring-1 ${s.cls}`}>
      <StatusGlyph status={status} className="size-3" />
      {section ? `s.${section} · ` : ''}
      {s.label}
    </span>
  )
}

/** Verdict card with a score bar, element-by-element breakdown with evidence quotes, other fitting sections. */
export function VerificationResultView({ result: r, className = 'space-y-5 rounded-2xl bg-white p-5 shadow-[0_1px_2px_rgb(15_23_42/0.04)] ring-1 ring-slate-200 sm:p-6' }) {
  const st = STATUS[r.status]
  const satisfied = r.elements.filter((e) => e.status === 'satisfied').length
  const [shown, setShown] = useState(0) // score bar grows in after mount
  useEffect(() => {
    const t = requestAnimationFrame(() => setShown(satisfied / Math.max(1, r.elements.length)))
    return () => cancelAnimationFrame(t)
  }, [satisfied, r.elements.length])

  return (
    <section className={`animate-rise ${className}`} data-testid="verify-result">
      <div className={`flex flex-wrap items-start gap-4 rounded-xl p-4 ring-1 ${st.card}`}>
        <span className={`grid size-10 shrink-0 animate-pop place-items-center rounded-full text-white shadow-sm ${st.icon}`} aria-hidden="true">
          <StatusGlyph status={r.status} className="size-5" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className={`rounded-md px-2 py-0.5 text-sm font-bold ring-1 ${st.cls}`} data-testid="verify-status">
              {st.label}
            </span>
            <span className="text-sm font-medium text-slate-700">
              s.{r.cited_section} · {r.title}
            </span>
          </div>
          <p className="mt-1.5 text-sm text-slate-700">{st.text(r.cited_section)}</p>
          {r.model_note && (
            <p className="mt-1 text-xs text-slate-500" data-testid="verify-model-note">
              Facts extracted by {r.model_id}: {r.model_note.split(';')[0]}.
            </p>
          )}
          <div className="mt-3 flex items-center gap-3">
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/70 ring-1 ring-slate-900/5">
              <div className={`h-full origin-left rounded-full transition-transform duration-1000 ease-[cubic-bezier(0.2,0,0,1)] ${st.bar}`} style={{ transform: `scaleX(${shown})` }} />
            </div>
            <span className="shrink-0 text-xs font-medium tabular-nums text-slate-600">
              {satisfied} of {r.elements.length} elements satisfied
            </span>
          </div>
        </div>
      </div>

      <div>
        <h3 className="mb-2 text-xs font-medium text-slate-500">Element by element</h3>
        <ol className="space-y-2">
          {r.elements.map((el, i) => (
            <li
              key={el.predicate}
              data-testid="verify-element"
              data-status={el.status}
              className="flex animate-rise gap-3 rounded-xl bg-white p-3 ring-1 ring-slate-200"
              style={stagger(i, 90, 150)}
            >
              <span className={`mt-0.5 grid size-6 shrink-0 place-items-center rounded-full ring-1 ${ELEMENT[el.status].dot}`} title={ELEMENT[el.status].label}>
                <StatusGlyph status={el.status} />
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-sm text-slate-800">
                  {el.description}
                  <span className={`ml-1.5 text-xs font-medium ${ELEMENT[el.status].cls}`}>{ELEMENT[el.status].label}</span>
                </p>
                <p className="mt-1 flex flex-wrap items-baseline gap-x-2 gap-y-0.5 text-xs">
                  <span className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-slate-700">{el.found}</span>
                  {el.evidence ? <span className="italic text-slate-500">“{el.evidence}”</span> : <span className="text-slate-400">no quote in the facts</span>}
                </p>
              </div>
            </li>
          ))}
        </ol>
      </div>

      {r.what_if && (
        <div className="rounded-xl bg-slate-50 px-4 py-3 text-sm ring-1 ring-slate-200" data-testid="verify-what-if">
          <p className="text-xs font-medium text-slate-500">What would change the verdict</p>
          <p className="mt-1 text-slate-700">{r.what_if.summary}</p>
          {r.what_if.missing.length > 0 && (
            <ul className="mt-1.5 space-y-1 text-xs text-slate-600">
              {r.what_if.missing.map((m) => (
                <li key={m.predicate} className="flex gap-2">
                  <span aria-hidden="true" className="mt-1.5 size-1 shrink-0 rounded-full bg-slate-400" />
                  <span>
                    {m.description}: if the facts established it, the verdict would be <b>{m.verdict_if_satisfied}</b>.
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {r.alternatives.some((a) => a.status === 'CONSISTENT') && (
        <p className="text-sm text-slate-700">
          These facts also satisfy:{' '}
          {r.alternatives
            .filter((a) => a.status === 'CONSISTENT')
            .map((a) => `s.${a.section} (${a.title})`)
            .join('; ')}
        </p>
      )}
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-xs text-slate-500">Other encoded sections</span>
        {r.alternatives.map((a) => (
          <Pill key={a.section} status={a.status} section={a.section} />
        ))}
      </div>

      {r.warnings.length > 0 && (
        <details className="text-xs text-amber-800">
          <summary className="cursor-pointer font-medium">{r.warnings.length} extraction note(s)</summary>
          <ul className="mt-1 list-disc pl-5">
            {r.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </details>
      )}
      <details className="text-xs text-slate-600">
        <summary className="cursor-pointer font-medium">All facts extracted by {r.model_id}</summary>
        <ul className="mt-1.5 space-y-0.5">
          {Object.entries(r.facts).map(([k, v]) => (
            <li key={k}>
              <span className="font-mono">{k}</span> = <b>{v.value}</b>
              {v.evidence && <span className="italic"> — “{v.evidence}”</span>}
            </li>
          ))}
        </ul>
      </details>
    </section>
  )
}

/** Shown while the server extracts and checks: an honest "working" state, not a fake progress bar. */
function Checking({ section }) {
  return (
    <div className="animate-rise overflow-hidden rounded-2xl bg-white ring-1 ring-slate-200" role="status" data-testid="verify-busy">
      <div className="h-1 w-full overflow-hidden bg-slate-100">
        <div className="h-full w-1/3 animate-[slide_1.4s_cubic-bezier(0.4,0,0.2,1)_infinite] rounded-full bg-indigo-500" />
      </div>
      <div className="flex items-center gap-4 p-5">
        <span aria-hidden="true" className="relative grid size-10 place-items-center">
          <span className="absolute inset-0 animate-ping-soft rounded-full bg-indigo-400" />
          <span className="relative grid size-10 place-items-center rounded-full bg-indigo-50 font-serif text-lg font-bold text-indigo-700 ring-1 ring-indigo-200">§</span>
        </span>
        <div>
          <p className="text-shimmer text-sm font-semibold">Checking the facts against s.{section}…</p>
          <p className="mt-0.5 text-xs text-slate-500">
            The facts are extracted twice (only what is stated consistently is kept), then the rules are applied. Usually
            10–20 seconds.
          </p>
        </div>
      </div>
    </div>
  )
}

export default function VerifyPanel() {
  const [meta, setMeta] = useState(null)
  const [text, setText] = useState('')
  const [section, setSection] = useState('')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const resultRef = useRef(null)

  useEffect(() => {
    verifyApi
      .sections()
      .then((m) => {
        setMeta(m)
        setSection((s) => s || m.sections.find((x) => x.section === '304A')?.section || m.sections[0]?.section || '')
      })
      .catch((e) => setError(e.message.includes('404') ? 'Verification is not available: the server could not start SWI-Prolog.' : e.message))
  }, [])

  const run = async (e) => {
    e.preventDefault()
    setBusy(true)
    setError('')
    setResult(null)
    requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }))
    try {
      setResult(await verifyApi.verify(text, section))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const sectionDef = meta?.sections.find((s) => s.section === section)
  const chars = text.trim().length

  return (
    <div className="relative isolate min-h-0 flex-1 overflow-y-auto px-4 py-8 sm:py-10">
      <div aria-hidden="true" className="bg-dots pointer-events-none absolute inset-0 -z-10" />
      <div className="mx-auto max-w-4xl space-y-6">
        <header className="relative">
          <div aria-hidden="true" className="pointer-events-none absolute -left-24 -top-24 -z-10 h-64 w-[30rem] max-w-[120%] animate-float rounded-full bg-[radial-gradient(closest-side,var(--color-indigo-200),transparent)] opacity-50 blur-2xl" />
          <p className="mb-3 inline-flex animate-rise items-center gap-1.5 rounded-full bg-white/70 px-2.5 py-1 text-xs font-medium text-slate-600 ring-1 ring-slate-200 backdrop-blur">
            <span aria-hidden="true" className="size-1.5 rounded-full bg-indigo-500" />
            Rule engine · {meta ? meta.sections.length : '…'} IPC sections encoded
          </p>
          <h2 className="animate-rise text-2xl font-semibold tracking-tight text-slate-900 sm:text-3xl" style={stagger(1)}>
            Check facts against an IPC section
          </h2>
          <p className="mt-2 max-w-2xl animate-rise text-sm leading-relaxed text-slate-500 sm:text-[0.9375rem]" style={stagger(2)}>
            Paste the facts of a case and choose the section it is charged under. An AI model extracts the facts the section
            depends on, each with a quote from your text, and a rule engine checks them against the section's elements.
          </p>
          {meta && (
            <p className="mt-3 flex max-w-2xl animate-rise gap-2 text-xs leading-relaxed text-slate-500" style={stagger(3)}>
              <svg {...stroke} strokeWidth={1.5} className="mt-px size-4 shrink-0 text-slate-400">
                <circle cx="10" cy="10" r="7.25" />
                <path d="M10 9v4.5M10 6.6v.1" />
              </svg>
              {meta.disclaimer}
            </p>
          )}
        </header>

        <form onSubmit={run} className="grid animate-rise gap-5 lg:grid-cols-[1fr_17rem]" style={stagger(4)}>
          <div className="space-y-5 rounded-2xl bg-white p-5 shadow-[0_1px_2px_rgb(15_23_42/0.04),0_12px_32px_-20px_rgb(15_23_42/0.25)] ring-1 ring-slate-200 sm:p-6">
            <fieldset>
              <legend className="mb-2 text-sm font-semibold text-slate-800">1. Section charged</legend>
              <div role="radiogroup" aria-label="Section charged" className="grid gap-2 sm:grid-cols-2">
                {meta?.sections.map((s) => {
                  const on = s.section === section
                  return (
                    <button
                      key={s.section}
                      type="button"
                      role="radio"
                      aria-checked={on}
                      onClick={() => setSection(s.section)}
                      className={`group flex items-start gap-2.5 rounded-xl px-3 py-2.5 text-left ring-1 transition-[background-color,box-shadow,transform] duration-200 ${
                        on ? 'bg-indigo-50 ring-2 ring-indigo-400' : 'bg-white ring-slate-200 hover:-translate-y-px hover:bg-slate-50 hover:ring-slate-300'
                      }`}
                      data-testid="verify-section"
                    >
                      <span
                        className={`mt-0.5 shrink-0 rounded-md px-1.5 py-0.5 font-mono text-xs font-semibold transition-colors ${
                          on ? 'bg-accent text-on-accent' : 'bg-slate-100 text-slate-600 group-hover:bg-slate-200'
                        }`}
                      >
                        {s.section}
                      </span>
                      <span className={`text-sm leading-snug ${on ? 'font-medium text-indigo-900' : 'text-slate-700'}`}>{s.title}</span>
                    </button>
                  )
                })}
                {!meta && !error && Array.from({ length: 4 }, (_, i) => <span key={i} className="h-12 animate-pulse rounded-xl bg-slate-100" />)}
              </div>
            </fieldset>

            <div>
              <div className="mb-2 flex items-center justify-between gap-3">
                <label className="text-sm font-semibold text-slate-800" htmlFor="verify-text">
                  2. Case facts
                </label>
                <button
                  type="button"
                  onClick={() => {
                    setText(EXAMPLE.text)
                    setSection(EXAMPLE.section)
                  }}
                  className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-indigo-600 hover:bg-indigo-50"
                >
                  <svg {...stroke} strokeWidth={1.5} className="size-3.5">
                    <path d="M10 3.5v2M10 14.5v2M3.5 10h2M14.5 10h2M5.4 5.4l1.4 1.4M13.2 13.2l1.4 1.4M5.4 14.6l1.4-1.4M13.2 6.8l1.4-1.4" />
                  </svg>
                  Try an example
                </button>
              </div>
              <div>
                <textarea
                  id="verify-text"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  rows={7}
                  maxLength={20000}
                  placeholder="e.g. The accused was driving a truck on the highway when…"
                  className="block w-full resize-y rounded-xl border border-slate-200 bg-slate-50/60 px-3.5 py-3 text-sm leading-relaxed text-slate-900 placeholder:text-slate-400 transition-[border-color,box-shadow,background-color] focus:border-indigo-400 focus:bg-white focus:outline-none focus:ring-4 focus:ring-indigo-100"
                />
                <span
                  className={`mt-1.5 block text-right text-[11px] tabular-nums ${chars > 0 && chars < MIN_CHARS ? 'text-amber-600' : 'text-slate-400'}`}
                >
                  {chars > 0 && chars < MIN_CHARS ? `${MIN_CHARS - chars} more characters needed` : `${text.length.toLocaleString()} / 20,000`}
                </span>
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 pt-4">
              <p className="text-xs text-slate-500">The check runs on this server; nothing is saved.</p>
              <button
                type="submit"
                disabled={busy || chars < MIN_CHARS || !section}
                className="inline-flex items-center gap-2 rounded-xl bg-accent px-5 py-2.5 text-sm font-semibold text-on-accent shadow-sm transition-[transform,background-color,box-shadow] duration-200 hover:-translate-y-0.5 hover:bg-accent-hover hover:shadow-md disabled:translate-y-0 disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none"
              >
                {busy ? (
                  <>
                    <span aria-hidden="true" className="size-4 animate-spin rounded-full border-2 border-current border-r-transparent" />
                    Checking…
                  </>
                ) : (
                  <>
                    Check{section ? ` against s.${section}` : ''}
                    <svg {...stroke} className="size-4">
                      <path d="M4.5 10h11M11 5.5l4.5 4.5-4.5 4.5" />
                    </svg>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* what the chosen section requires: always visible, re-animates when the section changes */}
          <aside className="h-fit rounded-2xl bg-white/70 p-5 ring-1 ring-slate-200 backdrop-blur lg:sticky lg:top-0" aria-live="polite">
            {sectionDef ? (
              <div key={sectionDef.section}>
                <p className="text-xs font-medium text-slate-500">What s.{sectionDef.section} requires</p>
                <p className="mt-0.5 animate-rise text-sm font-semibold text-slate-900">{sectionDef.title}</p>
                <ol className="mt-3 space-y-2.5">
                  {sectionDef.elements.map((el, i) => (
                    <li key={el.predicate} className="flex animate-rise gap-2.5 text-sm text-slate-700" style={stagger(i, 70, 80)}>
                      <span className="grid size-5 shrink-0 place-items-center rounded-full bg-indigo-50 text-[11px] font-semibold text-indigo-700 ring-1 ring-indigo-200">
                        {i + 1}
                      </span>
                      <span className="leading-snug">{el.description}</span>
                    </li>
                  ))}
                </ol>
                <p className="mt-4 text-xs leading-relaxed text-slate-500">
                  All {sectionDef.elements.length} must be established by the facts for a <b>Consistent</b> result.
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                <span className="block h-3 w-24 animate-pulse rounded bg-slate-100" />
                <span className="block h-4 w-40 animate-pulse rounded bg-slate-100" />
                <span className="block h-16 animate-pulse rounded bg-slate-100" />
              </div>
            )}
          </aside>
        </form>

        <div ref={resultRef} className="scroll-mt-4 space-y-4">
          {error && (
            <p className="flex animate-rise items-start gap-2 rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-200">
              <StatusGlyph status="violated" className="mt-0.5 size-4 shrink-0" />
              {error}
            </p>
          )}
          {busy && <Checking section={section} />}
          {result && <VerificationResultView result={result} />}
          {result && <VerdictProof result={result} />}
        </div>
      </div>
    </div>
  )
}
