import { useEffect, useState } from 'react'
import { reviewApi } from '../api.js'

const cache = new Map() // `${sentenceId}|${label}` -> explanation

const pct = (x) => `${Math.round(x * 100)}%`

/** The sentence with each word shaded by its attribution: green supports the role, red argues against it. */
function Highlighted({ text, words, method }) {
  const parts = []
  let at = 0
  words.forEach((w, i) => {
    if (w.start > at) parts.push(<span key={`g${i}`}>{text.slice(at, w.start)}</span>)
    const weight = method === 'removal' ? w.removal_weight : w.weight
    const a = Math.min(1, Math.abs(weight))
    const bg = weight >= 0 ? `rgba(16, 185, 129, ${0.08 + 0.55 * a})` : `rgba(239, 68, 68, ${0.08 + 0.55 * a})`
    parts.push(
      <span
        key={i}
        style={{ backgroundColor: bg }}
        className="rounded-sm px-0.5"
        title={method === 'removal' ? `leaving this word out changes the confidence by ${(-w.removal_drop * 100).toFixed(1)} points` : `IG ${weight.toFixed(2)}`}
      >
        {text.slice(w.start, w.end)}
      </span>,
    )
    at = w.end
  })
  if (at < text.length) parts.push(<span key="end">{text.slice(at)}</span>)
  return <p className="break-words leading-loose text-slate-800" data-testid="explain-highlight">{parts}</p>
}

export default function RoleExplanation({ caseId, item, labels, colour }) {
  const [label, setLabel] = useState(null) // null = the model's own top role
  const [method, setMethod] = useState('removal') // removal (more faithful here) | ig
  const [data, setData] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    const key = `${item.sentence_id}|${label || ''}`
    if (cache.has(key)) {
      setData(cache.get(key))
      return
    }
    setData(null)
    setError('')
    reviewApi
      .explain(caseId, item.sentence_id, label)
      .then((d) => {
        cache.set(key, d)
        setData(d)
      })
      .catch((e) => setError(e.message))
  }, [caseId, item.sentence_id, label])

  if (error) return <p className="mt-2 text-xs text-red-700">Could not explain: {error}</p>
  if (!data)
    return <p className="mt-2 animate-pulse text-xs text-slate-500" data-testid="explain-loading">Working out which words drove the suggestion (a few seconds)…</p>

  const probs = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1])
  const top = probs[0][0]
  return (
    <div className="mt-2 space-y-3 rounded-lg bg-white p-3 text-xs ring-1 ring-slate-200" data-testid="role-explanation" onClick={(e) => e.stopPropagation()}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium text-slate-700">
          Why {label && label !== top ? 'it might be' : 'the model says'}{' '}
          <span className={`rounded px-1.5 py-0.5 ring-1 ${colour[data.label]}`}>{data.label}</span>
        </span>
        <span className="text-slate-400">· words in green point to it, red against it</span>
        <span className="ml-auto inline-flex rounded bg-slate-100 p-0.5" role="radiogroup" aria-label="Explanation method">
          {[
            ['removal', 'Word removal'],
            ['ig', 'Integrated Gradients'],
          ].map(([m, name]) => (
            <button
              key={m}
              role="radio"
              aria-checked={method === m}
              onClick={() => setMethod(m)}
              className={`rounded px-1.5 py-0.5 ${method === m ? 'bg-white shadow-sm' : 'text-slate-500'}`}
            >
              {name}
            </button>
          ))}
        </span>
      </div>

      <Highlighted text={item.text} words={data.words} method={method} />
      <p className="text-slate-400">
        {method === 'removal'
          ? 'Shading = how much the confidence drops when the word is left out (hover a word for the number).'
          : 'Shading = Integrated Gradients attribution (how much each word moved the score from a blank sentence).'}
      </p>

      <p className={data.context_share > 0.5 ? 'text-amber-800' : 'text-slate-600'}>
        {pct(data.context_share)} of the evidence came from the <b>context</b> (where the sentence sits and its neighbouring
        sentences) rather than the sentence itself.
        {data.context_share > 0.5 && ' Mostly decided by the context: check whether the neighbouring sentences mislead it.'}
      </p>

      <div className="grid gap-1 sm:grid-cols-2">
        {probs.map(([name, p]) => (
          <div key={name} className="flex items-center gap-2">
            <span className="w-24 shrink-0 text-slate-600">{name}</span>
            <span className="h-2 flex-1 rounded bg-slate-100">
              <span className="block h-2 rounded bg-indigo-400" style={{ width: pct(p) }} />
            </span>
            <span className="w-9 text-right tabular-nums text-slate-500">{pct(p)}</span>
          </div>
        ))}
      </div>

      {data.neighbours.length > 0 && (
        <div>
          <p className="mb-1 font-medium text-slate-700">Most similar sentences labelled by people</p>
          <ul className="space-y-1">
            {data.neighbours.map((n) => (
              <li key={n.sentence_id} className="flex gap-2" data-testid="explain-neighbour">
                <span className={`h-fit shrink-0 rounded px-1.5 py-0.5 ring-1 ${colour[n.label] || ''}`}>{n.label}</span>
                <span className="text-slate-600">{n.text.length > 220 ? `${n.text.slice(0, 220)}…` : n.text}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-1">
        <span className="text-slate-500">Why not…</span>
        {labels
          .filter((l) => l !== data.label)
          .map((l) => (
            <button key={l} onClick={() => setLabel(l)} className="rounded px-1.5 py-0.5 text-slate-600 ring-1 ring-slate-200 hover:bg-slate-50">
              {l}
            </button>
          ))}
        {label && (
          <button onClick={() => setLabel(null)} className="text-indigo-600 hover:underline">
            back to the suggestion
          </button>
        )}
      </div>

      <p className="text-[11px] text-slate-400">
        {data.method}. Model {data.model}
        {data.queued_model && !data.queued_model.startsWith(data.model) && data.queued_suggestion !== top
          ? ` (the queued suggestion "${data.queued_suggestion}" came from ${data.queued_model})`
          : ''}
        .
      </p>
    </div>
  )
}
