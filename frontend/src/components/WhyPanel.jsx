import { useState } from 'react'

const BAND = {
  High: 'bg-emerald-100 text-emerald-800 ring-emerald-300',
  Medium: 'bg-amber-100 text-amber-800 ring-amber-300',
  Low: 'bg-red-100 text-red-800 ring-red-300',
}

const pct = (x) => `${Math.round(x * 100)}%`

function Bar({ value, threshold }) {
  return (
    <div className="relative h-1.5 w-20 shrink-0 overflow-hidden rounded-full bg-slate-200" aria-hidden="true">
      <div className="h-full bg-indigo-500" style={{ width: pct(Math.max(0, Math.min(1, value))) }} />
      {threshold != null && <div className="absolute inset-y-0 w-px bg-slate-700" style={{ left: pct(threshold) }} />}
    </div>
  )
}

/**
 * "Why this answer?" — built from the explanation SSE event. Every number here describes how the
 * answer relates to the retrieved IPC text; none of them says the answer is legally correct.
 */
export default function WhyPanel({ explanation: ex, sources, onOpenSource }) {
  const [hoverSentence, setHoverSentence] = useState(null)
  const [hoverSource, setHoverSource] = useState(null)
  const byMarker = new Map((sources || []).map((s) => [s.marker, s]))
  const atts = ex.attributions || []
  const supported = atts.filter((a) => a.supported).length
  const highlightedSource = hoverSentence != null ? atts[hoverSentence]?.source_marker : null
  const highlightedSentences = hoverSource != null ? new Set(ex.sources.find((s) => s.marker === hoverSource)?.supports || []) : new Set()

  return (
    <details className="mt-3 rounded-lg bg-slate-50 text-xs ring-1 ring-slate-200" open={ex.warnings.length > 0} data-testid="why-panel">
      <summary className="flex cursor-pointer select-none flex-wrap items-center gap-2 px-3 py-2">
        <span className="font-medium text-slate-700">Why this answer?</span>
        <span className={`rounded px-1.5 py-0.5 font-semibold ring-1 ${BAND[ex.band]}`} data-testid="why-band">
          Evidence match: {ex.band} ({ex.relevance.toFixed(2)})
        </span>
        {!ex.refused && atts.length > 0 && (
          <span className={supported === atts.length ? 'text-emerald-700' : 'text-red-700'} data-testid="why-support">
            {supported} of {atts.length} sentences supported by a source
          </span>
        )}
        {ex.warnings.length > 0 && (
          <span className="rounded bg-red-600 px-1.5 py-0.5 font-semibold text-white">
            {ex.warnings.length} warning{ex.warnings.length > 1 ? 's' : ''}
          </span>
        )}
      </summary>

      <div className="space-y-3 border-t border-slate-200 px-3 py-3">
        <p className="text-slate-500" data-testid="why-band-note">
          {ex.band_note} A strong match means closely related IPC text was found; check the answer against it, and
          consult a lawyer for advice.
        </p>

        {ex.warnings.length > 0 && (
          <ul className="space-y-1 rounded-md bg-red-50 px-3 py-2 text-red-800 ring-1 ring-red-200" data-testid="why-warnings">
            {ex.warnings.map((w) => (
              <li key={w}>⚠ {w}</li>
            ))}
          </ul>
        )}

        {atts.length > 0 && (
          <section>
            <h4 className="mb-1 font-semibold text-slate-700">The answer, sentence by sentence</h4>
            <ol className="space-y-1">
              {atts.map((a) => {
                const mismatch = a.supported && a.cited_supported === false
                const lit = hoverSentence === a.sentence_idx || highlightedSentences.has(a.sentence_idx)
                return (
                  <li
                    key={a.sentence_idx}
                    onMouseEnter={() => setHoverSentence(a.sentence_idx)}
                    onMouseLeave={() => setHoverSentence(null)}
                    data-testid="why-sentence"
                    data-supported={a.supported}
                    className={`rounded-md px-2 py-1.5 ring-1 ${
                      !a.supported ? 'bg-red-50 ring-red-200' : mismatch ? 'bg-amber-50 ring-amber-200' : 'bg-white ring-slate-200'
                    } ${lit ? 'outline outline-2 outline-indigo-400' : ''}`}
                  >
                    <p className="text-slate-800">{a.sentence}</p>
                    <div className="mt-1 flex flex-wrap items-center gap-2 text-slate-500">
                      {a.supported ? (
                        <>
                          <span>Supported by</span>
                          <button
                            onClick={() => byMarker.get(a.source_marker) && onOpenSource(byMarker.get(a.source_marker))}
                            className="rounded bg-indigo-50 px-1 font-semibold text-indigo-700 ring-1 ring-indigo-200 hover:bg-indigo-100"
                          >
                            [{a.source_marker}]
                          </button>
                          <span>{pct(a.similarity)} similar</span>
                        </>
                      ) : (
                        <span className="font-medium text-red-700">
                          No retrieved source supports this sentence (closest: [{a.source_marker}] at {pct(a.similarity)}, needs{' '}
                          {pct(ex.support_threshold)})
                        </span>
                      )}
                      {mismatch && (
                        <span className="font-medium text-amber-800">
                          but it cites {a.cited_markers.map((m) => `[${m}]`).join('')}, which does not support it
                        </span>
                      )}
                    </div>
                    {lit && a.supported && (
                      <p className="mt-1 border-l-2 border-indigo-300 pl-2 italic text-slate-600">“{a.evidence}”</p>
                    )}
                  </li>
                )
              })}
            </ol>
          </section>
        )}

        {ex.sources.length > 0 && (
          <section>
            <h4 className="mb-1 font-semibold text-slate-700">Retrieved sources, best match first</h4>
            <ul className="space-y-1">
              {ex.sources.map((s) => (
                <li
                  key={s.marker}
                  onMouseEnter={() => setHoverSource(s.marker)}
                  onMouseLeave={() => setHoverSource(null)}
                  data-testid="why-source"
                  className={`flex flex-wrap items-center gap-2 rounded-md bg-white px-2 py-1.5 ring-1 ring-slate-200 ${
                    highlightedSource === s.marker || hoverSource === s.marker ? 'outline outline-2 outline-indigo-400' : ''
                  }`}
                >
                  <button
                    onClick={() => byMarker.get(s.marker) && onOpenSource(byMarker.get(s.marker))}
                    className="font-semibold text-indigo-700 hover:underline"
                  >
                    [{s.marker}]
                  </button>
                  <span className="min-w-0 flex-1 truncate text-slate-700" title={s.citation_path}>
                    {s.citation_path.replace(/^IPC > /, '')}
                  </span>
                  <Bar value={s.similarity} />
                  <span className="w-9 text-right tabular-nums text-slate-500">{pct(s.similarity)}</span>
                  <span className={s.cited ? 'text-emerald-700' : 'text-slate-400'}>{s.cited ? 'cited' : 'not cited'}</span>
                  <span className="text-slate-500">
                    {s.supports.length ? `supports sentence ${s.supports.map((i) => i + 1).join(', ')}` : 'supports no sentence'}
                  </span>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </details>
  )
}
