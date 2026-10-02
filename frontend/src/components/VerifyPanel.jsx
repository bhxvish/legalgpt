import { useEffect, useState } from 'react'
import { verifyApi } from '../api.js'

const STATUS = {
  CONSISTENT: {
    cls: 'bg-emerald-100 text-emerald-800 ring-emerald-300',
    label: 'Consistent',
    text: (s) => `The extracted facts satisfy every encoded element of s.${s}.`,
  },
  INCONSISTENT: {
    cls: 'bg-red-100 text-red-800 ring-red-300',
    label: 'Inconsistent',
    text: (s) => `At least one element of s.${s} is contradicted by the facts.`,
  },
  INSUFFICIENT: {
    cls: 'bg-amber-100 text-amber-800 ring-amber-300',
    label: 'Insufficient facts',
    text: (s) => `Some elements of s.${s} are not established by the facts (nothing contradicts them either).`,
  },
}

const ELEMENT = {
  satisfied: { icon: '✓', cls: 'text-emerald-700', label: 'satisfied' },
  violated: { icon: '✗', cls: 'text-red-700', label: 'contradicted' },
  missing: { icon: '?', cls: 'text-amber-700', label: 'not established' },
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

function Pill({ status, section }) {
  const s = STATUS[status]
  return <span className={`rounded px-2 py-0.5 text-xs font-semibold ring-1 ${s.cls}`}>{section ? `s.${section} · ` : ''}{s.label}</span>
}

/** Status badge, element-by-element breakdown with evidence quotes, other fitting sections. */
export function VerificationResultView({ result: r, className = "space-y-4 rounded-xl bg-white p-4 ring-1 ring-slate-200" }) {
  return (
      <section className={className} data-testid="verify-result">
        <div className="flex flex-wrap items-center gap-3">
          <span className={`rounded-lg px-3 py-1 text-sm font-bold ring-1 ${STATUS[r.status].cls}`} data-testid="verify-status">
            {STATUS[r.status].label}
          </span>
          <p className="text-sm text-slate-700">
            {STATUS[r.status].text(r.cited_section)} <span className="text-slate-500">({r.title})</span>
          </p>
        </div>
  
        <table className="w-full text-left text-sm">
          <thead className="text-xs text-slate-500">
            <tr>
              <th className="pb-1 pr-2 font-medium" />
              <th className="pb-1 pr-2 font-medium">Element</th>
              <th className="pb-1 pr-2 font-medium">Found</th>
              <th className="pb-1 font-medium">Evidence quoted from the facts</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {r.elements.map((el) => (
              <tr key={el.predicate} data-testid="verify-element" data-status={el.status} className="align-top">
                <td className={`py-1.5 pr-2 font-bold ${ELEMENT[el.status].cls}`} title={ELEMENT[el.status].label}>
                  {ELEMENT[el.status].icon}
                </td>
                <td className="py-1.5 pr-2 text-slate-800">
                  {el.description}
                  <span className={`ml-1 text-xs ${ELEMENT[el.status].cls}`}>({ELEMENT[el.status].label})</span>
                </td>
                <td className="py-1.5 pr-2 font-mono text-xs text-slate-700">{el.found}</td>
                <td className="py-1.5 text-xs italic text-slate-600">{el.evidence ? `“${el.evidence}”` : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
  
        {r.alternatives.some((a) => a.status === 'CONSISTENT') && (
          <p className="text-sm text-slate-700">
            These facts also satisfy:{' '}
            {r.alternatives
              .filter((a) => a.status === 'CONSISTENT')
              .map((a) => `s.${a.section} (${a.title})`)
              .join('; ')}
          </p>
        )}
        <div className="flex flex-wrap gap-1.5">
          <span className="text-xs text-slate-500">Other encoded sections:</span>
          {r.alternatives.map((a) => (
            <Pill key={a.section} status={a.status} section={a.section} />
          ))}
        </div>
  
        {r.warnings.length > 0 && (
          <details className="text-xs text-amber-800">
            <summary className="cursor-pointer">{r.warnings.length} extraction note(s)</summary>
            <ul className="mt-1 list-disc pl-5">
              {r.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          </details>
        )}
        <details className="text-xs text-slate-600">
          <summary className="cursor-pointer">All facts extracted by {r.model_id}</summary>
          <ul className="mt-1 space-y-0.5">
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

export default function VerifyPanel() {
  const [meta, setMeta] = useState(null)
  const [text, setText] = useState('')
  const [section, setSection] = useState('')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

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
    try {
      setResult(await verifyApi.verify(text, section))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const sectionDef = meta?.sections.find((s) => s.section === section)

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6">
      <div className="mx-auto max-w-3xl space-y-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Check facts against an IPC section</h2>
          <p className="mt-1 text-sm text-slate-600">
            Paste the facts of a case and choose the section it is charged under. An AI model extracts the facts the section
            depends on (each with a quote from your text), and a rule engine checks them against the section's elements.
            Proof of concept: {meta ? meta.sections.length : 'a few'} sections are encoded.
          </p>
          {meta && <p className="mt-1 text-xs text-slate-500">{meta.disclaimer}</p>}
        </div>

        <form onSubmit={run} className="space-y-3 rounded-xl bg-white p-4 ring-1 ring-slate-200">
          <label className="block text-sm font-medium text-slate-700" htmlFor="verify-text">
            Case facts
          </label>
          <textarea
            id="verify-text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={7}
            maxLength={20000}
            placeholder="e.g. The accused was driving a truck on the highway when…"
            className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
          />
          <div className="flex flex-wrap items-end gap-3">
            <label className="text-sm text-slate-700">
              <span className="block font-medium">Section charged</span>
              <select
                value={section}
                onChange={(e) => setSection(e.target.value)}
                className="mt-1 rounded-lg border border-slate-300 px-2 py-1.5 text-sm"
                aria-label="Section charged"
              >
                {meta?.sections.map((s) => (
                  <option key={s.section} value={s.section}>
                    s.{s.section} — {s.title}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="submit"
              disabled={busy || text.trim().length < 40 || !section}
              className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-40"
            >
              {busy ? 'Checking…' : 'Check'}
            </button>
            <button
              type="button"
              onClick={() => {
                setText(EXAMPLE.text)
                setSection(EXAMPLE.section)
              }}
              className="text-sm text-indigo-600 hover:underline"
            >
              Try an example
            </button>
            {text.trim().length > 0 && text.trim().length < 40 && <span className="text-xs text-slate-500">at least 40 characters</span>}
          </div>
          {sectionDef && (
            <details className="text-xs text-slate-600">
              <summary className="cursor-pointer">What s.{sectionDef.section} requires ({sectionDef.elements.length} elements)</summary>
              <ul className="mt-1 list-disc pl-5">
                {sectionDef.elements.map((el) => (
                  <li key={el.predicate}>{el.description}</li>
                ))}
              </ul>
            </details>
          )}
          {busy && <p className="text-xs text-slate-500">Extracting facts (twice, to keep only what is stated consistently) and checking the rules — about 10–20 seconds.</p>}
        </form>

        {error && <p className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-200">{error}</p>}

        {result && <VerificationResultView result={result} />}
      </div>
    </div>
  )
}
