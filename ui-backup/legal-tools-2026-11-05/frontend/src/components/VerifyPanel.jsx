import { Scales, Stamp } from './LegalArt.jsx'
import { useEffect, useRef, useState } from 'react'
import { verifyApi } from '../api.js'
import { takeHandoff } from '../handoff.js'
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
export function VerificationResultView({ result: r, className = 'space-y-5 rounded-2xl bg-white p-5 ring-1 ring-slate-200 sm:p-6' }) {
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
            <Stamp className={r.status === 'CONSISTENT' ? 'text-emerald-700' : r.status === 'INCONSISTENT' ? 'text-red-700' : 'text-amber-700'} data-testid="verify-status">{st.label}</Stamp>
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
  // sent from a chat answer ("Check the facts against s.X"): the question and the section
  const [handoff] = useState(() => takeHandoff('verify'))
  const [meta, setMeta] = useState(null)
  const [text, setText] = useState(handoff?.text || '')
  const [section, setSection] = useState('')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const resultRef = useRef(null)
  // the scales: the law outweighs an empty pan, the facts bring it level, the result settles it
  const tilt = result ? { CONSISTENT: 0, INSUFFICIENT: -3, INCONSISTENT: -5 }[result.status] ?? 0 : -5 * (1 - Math.min(1, text.trim().length / 240))

  useEffect(() => {
    verifyApi
      .sections()
      .then((m) => {
        setMeta(m)
        const sent = m.sections.find((x) => x.section === handoff?.section)?.section
        setSection((s) => sent || s || m.sections.find((x) => x.section === '304A')?.section || m.sections[0]?.section || '')
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
    <div className="min-h-0 flex-1 overflow-y-auto">
      <section className="min-h-full bg-field-blue px-4 py-7 text-paper sm:py-8">
        <form onSubmit={run} className="mx-auto flex max-w-5xl flex-col items-center">
          <h2 className="head animate-rise text-center">Weigh the facts against the law</h2>
          {handoff && (
            <p className="mt-3 animate-rise bg-field-gold px-3 py-1.5 text-[15px] font-medium text-ink" data-testid="verify-handoff">
              The facts are filled in below{handoff.section ? `, with s.${handoff.section}` : ''}. Check they read as the facts of a case, then weigh them.
            </p>
          )}

          <fieldset className="mt-5 animate-rise" style={stagger(1)}>
            <legend className="sr-only">Section charged</legend>
            <div role="radiogroup" aria-label="Section charged" className="flex flex-wrap justify-center gap-1.5">
              {meta?.sections.map((s) => {
                const on = s.section === section
                return (
                  <button
                    key={s.section}
                    type="button"
                    role="radio"
                    aria-checked={on}
                    title={s.title}
                    onClick={() => setSection(s.section)}
                    className={`press h-11 min-w-16 px-3 text-[17px] font-semibold tabular-nums transition-[background-color,color,transform] duration-200 ${
                      on ? '-translate-y-0.5 bg-field-gold text-ink' : 'bg-white/10 text-paper hover:bg-white/20'
                    }`}
                    data-testid="verify-section"
                  >
                    {s.section}
                  </button>
                )
              })}
              {!meta && !error && Array.from({ length: 7 }, (_, i) => <span key={i} className="h-11 w-16 animate-pulse bg-white/10" />)}
            </div>
          </fieldset>

          <Scales arm={225} drop={44} base={false} busy={busy} tilt={tilt} className="mt-2 hidden w-[63.4%] text-paper md:block" />

          <div className="light-scope relative grid w-full gap-4 text-ink md:grid-cols-2 md:gap-8">
            <span aria-hidden="true" className="absolute -bottom-5 left-1/2 top-0 hidden -translate-x-1/2 bg-paper md:block" style={{ width: '13.7px' }} />
            <div className="animate-sheet bg-white p-5" aria-live="polite" style={stagger(2, 80)}>
              <p className="text-[15px] font-medium text-slate-600">The law requires</p>
              {sectionDef ? (
                <div key={sectionDef.section}>
                  <p className="mt-0.5 animate-rise text-[1.25rem] font-semibold leading-tight tracking-[-0.02em]">s.{sectionDef.section} {sectionDef.title}</p>
                  <ol className="mt-3 space-y-2">
                    {sectionDef.elements.map((el, i) => {
                      const found = result?.cited_section === sectionDef.section ? result.elements.find((x) => x.predicate === el.predicate) : null
                      return (
                        <li key={el.predicate} className="flex animate-rise gap-2.5 text-[15px] leading-snug" style={stagger(i, 70, 80)}>
                          {found ? (
                            <span className={`mt-px grid size-5 shrink-0 place-items-center rounded-full ring-1 ${ELEMENT[found.status].dot}`} title={ELEMENT[found.status].label}>
                              <StatusGlyph status={found.status} className="size-3" />
                            </span>
                          ) : (
                            <span className="w-5 shrink-0 text-center font-semibold text-indigo-600">{i + 1}</span>
                          )}
                          {el.description}
                        </li>
                      )
                    })}
                  </ol>
                </div>
              ) : (
                <span className="mt-2 block h-24 animate-pulse bg-slate-100" />
              )}
            </div>

            <div className="flex animate-sheet flex-col bg-white p-5" style={stagger(3, 80)}>
              <div className="flex items-baseline justify-between gap-3">
                <label className="text-[15px] font-medium text-slate-600" htmlFor="verify-text">The facts show</label>
                <button
                  type="button"
                  onClick={() => {
                    setText(EXAMPLE.text)
                    setSection(EXAMPLE.section)
                  }}
                  className="press text-[15px] font-medium text-indigo-600 underline underline-offset-2"
                >
                  Try an example
                </button>
              </div>
              <textarea
                id="verify-text"
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={6}
                maxLength={20000}
                placeholder="The accused was driving a truck on the highway when…"
                className="ruled mt-1 block w-full flex-1 resize-y bg-transparent font-serif text-[1.125rem] text-slate-900 placeholder:text-slate-500 focus:outline-none"
              />
              <span className={`mt-1 block text-right text-[13px] tabular-nums ${chars > 0 && chars < MIN_CHARS ? 'font-medium text-amber-700' : 'text-slate-500'}`}>
                {chars > 0 && chars < MIN_CHARS ? `${MIN_CHARS - chars} more characters needed` : `${text.length.toLocaleString()} / 20,000`}
              </span>
            </div>
          </div>

          <button
            type="submit"
            disabled={busy || chars < MIN_CHARS || !section}
            className="press mt-5 inline-flex h-12 items-center gap-2 bg-accent px-7 text-[17px] font-medium text-on-accent hover:bg-accent-hover disabled:bg-white/15 disabled:text-paper/60"
          >
            {busy ? (
              <>
                <span aria-hidden="true" className="size-4 animate-spin rounded-full border-2 border-current border-r-transparent" />
                Weighing…
              </>
            ) : (
              <>Weigh against{section ? ` s.${section}` : ' the section'}</>
            )}
          </button>
          {meta && <p className="mt-3 max-w-2xl text-center text-[15px] leading-snug">{meta.disclaimer}</p>}
        </form>
      </section>

      <div className="mx-auto max-w-4xl space-y-6 px-4 empty:hidden [&:has(>div:not(:empty))]:py-8">
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
