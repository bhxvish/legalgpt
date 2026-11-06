import { Scales, Stamp } from './LegalArt.jsx'
import { useEffect, useRef, useState } from 'react'
import { Fold } from './AnswerRecord.jsx'
import { caseApi, verifyApi } from '../api.js'
import { modelLabel, modelNote } from '../modelNames.js'
import Select from './Select.jsx'
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

const FIELD = {
  CONSISTENT: 'bg-field-green text-paper',
  INCONSISTENT: 'bg-field-red text-ink',
  INSUFFICIENT: 'bg-field-gold text-ink',
}

/** The finding: a field of its colour with the stamp, then the law and the facts side by side. */
export function VerificationResultView({ result: r, compact = false }) {
  const st = STATUS[r.status]
  const satisfied = r.elements.filter((e) => e.status === 'satisfied').length

  return (
    <section className="animate-rise space-y-5" data-testid="verify-result">
      <div className={`flex flex-wrap items-center justify-between gap-x-8 gap-y-4 ${FIELD[r.status]} ${compact ? 'px-4 py-3.5' : 'px-5 py-6 sm:px-7'}`}>
        <div className="min-w-0 flex-1 basis-64">
          <p className="text-[15px] font-medium">s.{r.cited_section} {r.title}</p>
          <p className={`mt-1 leading-snug ${compact ? 'text-[15px]' : 'text-[17px]'}`}>{st.text(r.cited_section)}</p>
          {r.model_note && (
            <p className="mt-1 text-[14px]" data-testid="verify-model-note">Facts extracted by {r.model_id}: {r.model_note.split(';')[0]}.</p>
          )}
          <p className="mt-3 flex items-center gap-1.5 text-[15px] font-medium">
            {r.elements.map((el, i) => (
              <span
                key={el.predicate}
                aria-hidden="true"
                className={`size-3.5 animate-pop border-2 border-current ${el.status === 'satisfied' ? 'bg-current' : ''}`}
                style={stagger(i, 110, 200)}
              />
            ))}
            <span className="ml-1.5">{satisfied} of {r.elements.length} elements</span>
          </p>
        </div>
        <Stamp className={compact ? 'text-[1.125rem]' : 'text-[clamp(1.375rem,1rem+1.4vw,2rem)]'} data-testid="verify-status">{st.label}</Stamp>
      </div>

      <div>
        <div className="hidden grid-cols-[1.75rem_minmax(0,1fr)_minmax(0,1fr)] gap-x-4 pb-1.5 text-[14px] font-medium text-slate-600 sm:grid">
          <span />
          <span>The law requires</span>
          <span>The facts show</span>
        </div>
        <ol>
          {r.elements.map((el, i) => (
            <li
              key={el.predicate}
              data-testid="verify-element"
              data-status={el.status}
              className="grid animate-rise grid-cols-[1.75rem_minmax(0,1fr)] gap-x-4 gap-y-1 border-t border-slate-200 py-3 sm:grid-cols-[1.75rem_minmax(0,1fr)_minmax(0,1fr)]"
              style={stagger(i, 90, 150)}
            >
              <span className={`mt-0.5 grid size-6 place-items-center ring-1 ${ELEMENT[el.status].dot}`} title={ELEMENT[el.status].label}>
                <StatusGlyph status={el.status} />
              </span>
              <p className="text-[16px] leading-snug text-slate-900">{el.description}</p>
              <div className="col-start-2 sm:col-start-3">
                <p className={`text-[15px] font-semibold ${ELEMENT[el.status].cls}`}>{ELEMENT[el.status].label}</p>
                {el.evidence ? (
                  <p className="mt-0.5 font-serif text-[16px] italic leading-snug text-slate-700">“{el.evidence}”</p>
                ) : (
                  <p className="mt-0.5 text-[14px] text-slate-500">No quote in the facts</p>
                )}
              </div>
            </li>
          ))}
        </ol>
      </div>

      {r.what_if && (
        <div className="border-l-4 border-indigo-500 pl-4" data-testid="verify-what-if">
          <p className="text-[1.125rem] font-semibold tracking-[-0.02em] text-slate-900">What would change this</p>
          <p className="mt-1 text-[16px] leading-snug text-slate-700">{r.what_if.summary}</p>
          {r.what_if.missing.length > 0 && (
            <ul className="mt-2 space-y-1 text-[15px] leading-snug text-slate-700">
              {r.what_if.missing.map((m) => (
                <li key={m.predicate}>{m.description}: if the facts established it, the verdict would be <b className="font-semibold">{m.verdict_if_satisfied}</b>.</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {r.alternatives.some((a) => a.status === 'CONSISTENT') && (
        <p className="text-[16px] leading-snug text-slate-900">
          These facts also satisfy{' '}
          {r.alternatives.filter((a) => a.status === 'CONSISTENT').map((a) => `s.${a.section} (${a.title})`).join('; ')}.
        </p>
      )}

      <div className="space-y-2">
        {r.alternatives.length > 0 && (
          <Fold title="The other encoded sections">
            <ul className="grid gap-x-8 sm:grid-cols-2">
              {r.alternatives.map((a) => (
                <li key={a.section} className="flex items-center justify-between gap-3 border-b border-slate-200 py-1.5 text-[15px]">
                  <span className="font-medium text-slate-900">s.{a.section}</span>
                  <span className="text-slate-600">{STATUS[a.status].label}</span>
                </li>
              ))}
            </ul>
          </Fold>
        )}
        {r.warnings.length > 0 && (
          <Fold title={`Notes from reading the facts (${r.warnings.length})`}>
            <ul className="space-y-1 text-[14px] leading-snug text-slate-700">
              {r.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          </Fold>
        )}
        {r.facts && (
          <Fold title={`All facts extracted by ${r.model_id}`}>
            <ul className="space-y-1 text-[14px] leading-snug text-slate-700">
              {Object.entries(r.facts).map(([k, v]) => (
                <li key={k}>
                  {k.replace(/_/g, ' ')}: <b className="font-semibold">{v.value}</b>
                  {v.evidence && <span className="font-serif italic"> “{v.evidence}”</span>}
                </li>
              ))}
            </ul>
          </Fold>
        )}
      </div>
    </section>
  )
}

export default function VerifyPanel({ active = true }) {
  // sent from a chat answer ("Check the facts against s.X") or the Overview: the facts and the section
  const [handoff, setHandoff] = useState(null)
  const [meta, setMeta] = useState(null)
  const [text, setText] = useState('')
  const [section, setSection] = useState('')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [models, setModels] = useState([])
  const [model, setModel] = useState('') // '': automatic, the free models in turn
  const resultRef = useRef(null)
  // the scales: the law outweighs an empty pan, the facts bring it level, the result settles it
  const tilt = result ? { CONSISTENT: 0, INSUFFICIENT: -3, INCONSISTENT: -5 }[result.status] ?? 0 : -5 * (1 - Math.min(1, text.trim().length / 240))

  useEffect(() => {
    caseApi.models().then((list) => setModels(list)).catch(() => setModels([]))
    verifyApi
      .sections()
      .then((m) => {
        setMeta(m)
        setSection((s) => s || m.sections.find((x) => x.section === '304A')?.section || m.sections[0]?.section || '')
      })
      .catch((e) => setError(e.message.includes('404') ? 'Verification is not available: the server could not start SWI-Prolog.' : e.message))
  }, [])

  // this page stays mounted, so work handed over is picked up each time the tab opens
  useEffect(() => {
    const sent = active && takeHandoff('verify')
    if (!sent) return
    setHandoff(sent)
    setText(sent.text || '')
    if (sent.section) setSection(sent.section)
    setResult(null)
    setError('')
  }, [active])

  useEffect(() => {
    if (result) requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
  }, [result])

  const run = async (e) => {
    e.preventDefault()
    setBusy(true)
    setError('')
    setResult(null)
    try {
      setResult(await verifyApi.verify(text, section, model))
    } catch (err) {
      const down = err.message.includes('503')
      setError(`${err.message.replace(/^HTTP \d+: /, '')}${down && model ? ' Choose another model and weigh again.' : ''}`)
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
              <div className="mt-1 flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
                <span className="flex min-w-0 items-center gap-1 text-[14px] text-slate-600">
                  Read by
                  <Select
                    up
                    label="The model that reads the facts"
                    value={model}
                    onChange={setModel}
                    disabled={busy}
                    menuClass="w-72"
                    testId="verify-model"
                    options={[
                      { value: '', label: 'Automatic', hint: 'Free models in turn, if one is down' },
                      ...models.map((m) => ({ value: m.id, label: modelLabel(m.id), hint: m.available ? modelNote(m) : m.note || 'Not available', disabled: !m.available })),
                    ]}
                  />
                </span>
                <span className={`text-[13px] tabular-nums ${chars > 0 && chars < MIN_CHARS ? 'font-medium text-amber-700' : 'text-slate-500'}`}>
                  {chars > 0 && chars < MIN_CHARS ? `${MIN_CHARS - chars} more characters needed` : `${text.length.toLocaleString()} / 20,000`}
                </span>
              </div>
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
          {busy && (
            <p className="mt-3 animate-rise text-center text-[15px] font-medium leading-snug" role="status" data-testid="verify-busy">
              The facts are read twice and only what is stated both times counts, then the rules are applied.
            </p>
          )}
          {meta && <p className="mt-3 max-w-2xl text-center text-[15px] leading-snug">{meta.disclaimer}</p>}
        </form>
      </section>

      <div ref={resultRef} className={result || error ? 'scroll-mt-0 bg-field-blue px-4 pb-10 pt-2' : 'hidden'}>
        <div className="light-scope mx-auto max-w-4xl animate-sheet space-y-7 bg-white p-5 text-ink sm:p-8">
          {error && <p className="border-l-4 border-red-500 pl-3 text-[16px] text-red-700">{error}</p>}
          {result && <VerificationResultView result={result} />}
          {result && <VerdictProof result={result} />}
        </div>
      </div>
    </div>
  )
}
