const pct = (x) => `${Math.round(x * 100)}%`

function Bar({ value, threshold }) {
  return (
    <div className="relative h-1.5 w-20 shrink-0 overflow-hidden rounded-full bg-slate-200" aria-hidden="true">
      <div className="h-full bg-indigo-500" style={{ width: pct(Math.max(0, Math.min(1, value))) }} />
      {threshold != null && <div className="absolute inset-y-0 w-px bg-slate-700" style={{ left: pct(threshold) }} />}
    </div>
  )
}

/** A sentence as a reader should see it: no table bars, no [n] markers. */
const plain = (t) => t.replace(/\|/g, ' ').replace(/\s*\[\d+\]/g, '').replace(/\s+/g, ' ').trim()

function Sentence({ a, tone, children, onOpenSource, byMarker }) {
  const source = byMarker.get(a.source_marker)
  return (
    <li className={`border-l-2 py-1 pl-3 ${tone}`} data-testid="why-sentence" data-supported={a.supported}>
      <p className="text-[15px] leading-snug text-slate-900">{plain(a.sentence)}</p>
      <p className="mt-0.5 text-[13px] leading-snug text-slate-600">
        {children}{' '}
        {source && (
          <button onClick={() => onOpenSource(source)} className="press font-medium text-indigo-600 underline underline-offset-2">
            [{a.source_marker}] {source.section ? `s.${source.section}` : 'source'}
          </button>
        )}
      </p>
    </li>
  )
}

function Group({ title, note, open = false, children, count }) {
  if (!count) return null
  return (
    <details className="group" open={open}>
      <summary className="press inline-flex cursor-pointer select-none items-center gap-2 text-[15px] font-medium text-slate-900">
        <span aria-hidden="true" className="text-slate-500 transition-transform duration-200 group-open:rotate-90">›</span>
        {title} ({count})
      </summary>
      {note && <p className="mt-1 text-[14px] leading-snug text-slate-600">{note}</p>}
      <ol className="mt-2 space-y-2">{children}</ol>
    </details>
  )
}

/**
 * Every sentence of the answer against the sources, built from the explanation SSE event. What needs
 * reading comes first and is open; the rest waits. Every number here describes how the answer relates
 * to the retrieved text; none of them says the answer is legally correct.
 */
export function WhatToCheck({ explanation: ex, sources, onOpenSource }) {
  const byMarker = new Map((sources || []).map((s) => [s.marker, s]))
  const atts = ex.attributions || []
  // legal claims (section numbers, punishments, quotes) must be backed by a source. Other unsupported
  // sentences are often the law applied to the user's facts, which text similarity cannot verify, but
  // possibly off-topic (attribution_analyzer.sentence_kind)
  const isAnalysis = (a) => a.kind === 'analysis'
  const claims = atts.filter((a) => !isAnalysis(a))
  const supportedClaims = claims.filter((a) => a.supported).length
  const unsupported = atts.filter((a) => !a.supported && !isAnalysis(a))
  const mismatched = atts.filter((a) => a.supported && a.cited_supported === false)
  const applied = atts.filter((a) => !a.supported && isAnalysis(a))
  const backed = atts.filter((a) => a.supported && a.cited_supported !== false)
  const props = { onOpenSource, byMarker }

  return (
    <div className="space-y-4">
      <div className="grid gap-x-8 gap-y-3 sm:grid-cols-2">
        {!ex.refused && claims.length > 0 && (
          <div data-testid="why-support">
            <p className="text-[1.75rem] font-semibold leading-none tracking-[-0.03em]">{supportedClaims} of {claims.length}</p>
            <p className="mt-1 text-[14px] leading-snug text-slate-600">legal statements have a source</p>
          </div>
        )}
        <div data-testid="why-band">
          <p className="text-[1.75rem] font-semibold leading-none tracking-[-0.03em]">{ex.band} <span className="text-[1rem] font-medium text-slate-600">({ex.relevance.toFixed(2)})</span></p>
          <p className="mt-1 text-[14px] leading-snug text-slate-600" data-testid="why-band-note">match between the law found and your question. It does not say the answer is correct.</p>
        </div>
      </div>

      <Group title="Read these first" count={unsupported.length + mismatched.length} open>
        {unsupported.map((a) => (
          <Sentence key={a.sentence_idx} a={a} tone="border-red-500" {...props}>
            No source says this. Closest match {pct(a.similarity)}, needs {pct(ex.support_threshold)}.
          </Sentence>
        ))}
        {mismatched.map((a) => (
          <Sentence key={a.sentence_idx} a={a} tone="border-amber-500" {...props}>
            It cites {a.cited_markers.map((m) => `[${m}]`).join('')}, but a different source supports it.
          </Sentence>
        ))}
      </Group>

      <Group title="The law applied to your facts" count={applied.length} note="These sentences use your own situation, so they cannot be matched to the text of a section. Read them against the cited section.">
        {applied.map((a) => (
          <Sentence key={a.sentence_idx} a={a} tone="border-slate-300" {...props}>
            Closest source, at {pct(a.similarity)}.
          </Sentence>
        ))}
      </Group>

      <Group title="Backed by a source" count={backed.length}>
        {backed.map((a) => (
          <Sentence key={a.sentence_idx} a={a} tone="border-emerald-500" {...props}>
            {pct(a.similarity)} similar.
          </Sentence>
        ))}
      </Group>

      {ex.warnings.length > 0 && (
        <details className="group text-[14px] text-slate-600">
          <summary className="press inline-flex cursor-pointer select-none items-center gap-2 font-medium text-slate-800">
            <span aria-hidden="true" className="text-slate-500 transition-transform duration-200 group-open:rotate-90">›</span>
            All notes from the check ({ex.warnings.length})
          </summary>
          <ul className="mt-1 space-y-0.5 pl-4" data-testid="why-warnings">
            {ex.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  )
}

/** Every retrieved source, best match first, and which sentences it supports. */
export function SourceMatches({ explanation: ex, sources, onOpenSource }) {
  const byMarker = new Map((sources || []).map((s) => [s.marker, s]))
  return (
    <ul>
      {ex.sources.map((s) => (
        <li key={s.marker} data-testid="why-source" className="flex flex-wrap items-center gap-x-3 gap-y-0.5 border-b border-slate-200 py-1.5 text-[14px]">
          <button onClick={() => byMarker.get(s.marker) && onOpenSource(byMarker.get(s.marker))} className="w-6 text-left font-semibold text-indigo-600 hover:underline">
            [{s.marker}]
          </button>
          <span className="min-w-[10rem] flex-1 truncate text-slate-800" title={s.citation_path}>
            {s.section ? `s.${s.section}` : s.citation_path.replace(/^IPC > /, '')}
            {s.title ? ` ${s.title}` : ''}
          </span>
          <Bar value={s.similarity} />
          <span className="w-9 text-right tabular-nums text-slate-600">{pct(s.similarity)}</span>
          <span className={`w-16 ${s.cited ? 'font-medium text-emerald-700' : 'text-slate-500'}`}>{s.cited ? 'cited' : 'not cited'}</span>
          <span className="text-slate-600">{s.supports.length ? `sentence ${s.supports.map((i) => i + 1).join(', ')}` : 'no sentence'}</span>
        </li>
      ))}
    </ul>
  )
}

/** The same check, as one folded panel, for the case analysis. */
export default function WhyPanel({ explanation: ex, sources, retrieval, onOpenSource }) {
  const claims = (ex.attributions || []).filter((a) => a.kind !== 'analysis')
  const supported = claims.filter((a) => a.supported).length
  return (
    <details className="group mt-3 bg-white p-3.5 text-[15px]" data-testid="why-panel">
      <summary className="press flex cursor-pointer select-none flex-wrap items-center gap-x-3 gap-y-1">
        <span aria-hidden="true" className="text-slate-500 transition-transform duration-200 group-open:rotate-90">›</span>
        <span className="font-medium text-slate-900">Check this against its sources</span>
        {!ex.refused && claims.length > 0 && (
          <span className={supported === claims.length ? 'text-emerald-700' : 'font-medium text-amber-700'}>{supported} of {claims.length} legal statements have a source</span>
        )}
      </summary>
      <div className="mt-3 space-y-4">
        <WhatToCheck explanation={ex} sources={sources} onOpenSource={onOpenSource} />
        {ex.sources.length > 0 && <SourceMatches explanation={ex} sources={sources} onOpenSource={onOpenSource} />}
        {retrieval && <HowChosen r={retrieval} />}
      </div>
    </details>
  )
}

/** How the sources were chosen: floor, scope check, lay-term expansion, follow-up context. */
export function HowChosen({ r }) {
  const lowered = r.floor < r.base_floor
  const lines = []
  if (r.background_ignored?.length)
    lines.push(`Left out of the search, because the law does not turn on it: ${r.background_ignored.join(' · ')}. The answer was still written from your whole question.`)
  if (r.translation)
    lines.push(`Your question is in Hindi; the IPC index is English, so it was searched in translation: "${r.translation.text}". The answer was asked for in Hindi with the same citations.`)
  if (r.named_sections.length) lines.push(`You named ${r.named_sections.map((s) => `s.${s}`).join(', ')}, so those sections were fetched directly.`)
  if (r.lexicon_terms.length)
    lines.push(`Everyday words in your question were matched to legal wording for the search: ${r.lexicon_terms.map((t) => `"${t.split(':')[0]}"`).join(', ')}.`)
  if (r.defined_sections?.length)
    lines.push(`Your question asks what a term means, so the section titled with it (${r.defined_sections.map((s) => `s.${s}`).join(', ')}) was fetched as well.`)
  if (r.used_context) lines.push('Your question looked like a follow-up, so the previous question was searched too.')
  const rw = r.rewrite
  if (rw?.used)
    lines.push(
      `The model read your question as: ${rw.offences.map((o) => `${o.name} (s.${o.section})`).join(', ')}. ` +
        'The text of those sections was fetched. This only chose what to read: the answer must still come from, and cite, the statute text.',
    )
  else if (rw && !rw.criminal) lines.push('The model judged this not to be a criminal matter, so its reading was not used for the search.')
  else if (rw?.offences?.length)
    lines.push(`The model suggested ${rw.offences.map((o) => `s.${o.section}`).join(', ')}, but the scope check did not agree that this is about a crime, so the suggestion was not used.`)
  if (r.scope_probability != null)
    lines.push(
      lowered
        ? `No section reached the usual match level (${pct(r.base_floor)}), but the question was judged to be about a crime (${pct(r.scope_probability)} sure), so sources down to ${pct(r.floor)} were used.`
        : r.floor > 1
          ? `The question was judged not to be about an IPC offence (${pct(r.scope_probability)} likely to be), so no sources were used.`
          : `The question was judged ${r.scope_probability >= 0.5 ? '' : 'not '}to be about a crime (${pct(r.scope_probability)}).`,
    )
  if (!r.sources.length && r.floor <= 1) lines.push(`The best match was ${pct(r.best_similarity)}, below the ${pct(r.floor)} needed, so no sources were used.`)
  return (
    <section data-testid="why-retrieval">
      <ul className="list-disc space-y-1 pl-5 text-[14px] leading-snug text-slate-600">
        {lines.length ? lines.map((l) => <li key={l}>{l}</li>) : <li>Sources with a match of at least {pct(r.floor)} were used, best first.</li>}
      </ul>
      {(new Set(r.sources.map((s) => s.weight)).size > 1 || r.sources.some((s) => s.named)) && (
        <p className="mt-1 text-[13px] text-slate-500">
          Ranking weights: {r.sources.map((s) => `[${s.marker}] ×${s.weight}${s.named ? ' (named)' : ''}`).join(' · ')} (bare-act text ×1.2;
          state amendments and Chapter I general provisions down-weighted).
        </p>
      )}
    </section>
  )
}
