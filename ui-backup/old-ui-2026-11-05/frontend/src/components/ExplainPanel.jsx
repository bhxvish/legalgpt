import { useEffect, useRef, useState } from 'react'
import { explainApi } from '../api.js'

// "Understand this answer": why the model concluded what it did, beyond the citations, in plain language.
// Four parts, each computed on request by app/explainability (results are kept on the message, so they
// survive tab switches and reloads):
//   How it decided    the model's reasoning (issue, law, facts against requirements, conclusion), then
//                     tested: removing each fact it says it relied on should change the conclusion
//   Words that matter which words of the question tied it to each section (word removal; no model call)
//   Law it relied on  which sections the conclusion depends on (leave one section out)
//   Bias check        the same question with gender, religion, caste, income, region or age changed: does the result change, and is
//                     any change required by the law itself? Runs by itself for gender-sensitive questions.

// --- small pieces -----------------------------------------------------------------------------------------

const ic = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.6, strokeLinecap: 'round', strokeLinejoin: 'round', viewBox: '0 0 20 20' }
const Icon = {
  steps: <svg {...ic} className="size-4"><path d="M4 4.5h12M4 10h8M4 15.5h5" /><circle cx="15.5" cy="15.5" r="2" /></svg>,
  words: <svg {...ic} className="size-4"><path d="M3.5 6h13M3.5 10h8.5M3.5 14h11" /><path d="M14 9.2l2.5 1.6-2.5 1.6" /></svg>,
  law: <svg {...ic} className="size-4"><path d="M10 3v14M6 17h8M4 6h12" /><path d="M4 6l-2 5h4zM16 6l-2 5h4z" /></svg>,
  scale: <svg {...ic} className="size-4"><circle cx="7" cy="8" r="2.5" /><circle cx="13.5" cy="8" r="2.5" /><path d="M3 16c.8-2.3 2.3-3.5 4-3.5s3.2 1.2 4 3.5M9.5 16c.8-2.3 2.3-3.5 4-3.5s3.2 1.2 4 3.5" /></svg>,
  check: <svg {...ic} className="size-3.5"><path d="M4.5 10.5l3.5 3.5 7.5-8" /></svg>,
  cross: <svg {...ic} className="size-3.5"><path d="M5.5 5.5l9 9M14.5 5.5l-9 9" /></svg>,
  dot: <svg {...ic} className="size-3.5"><circle cx="10" cy="10" r="2.5" fill="currentColor" /></svg>,
}

export function Badge({ tone = 'slate', children, title }) {
  const tones = {
    slate: 'bg-slate-100 text-slate-700 ring-slate-200',
    good: 'bg-emerald-50 text-emerald-800 ring-emerald-200',
    warn: 'bg-amber-50 text-amber-800 ring-amber-200',
    bad: 'bg-red-50 text-red-800 ring-red-200',
    info: 'bg-indigo-50 text-indigo-800 ring-indigo-200',
  }
  return (
    <span title={title} className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium ring-1 ${tones[tone]}`}>
      {children}
    </span>
  )
}

function RunButton({ onClick, busy, children, note }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={onClick}
        disabled={busy}
        className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-3.5 py-2 text-xs font-semibold text-white shadow-sm hover:bg-indigo-700 disabled:bg-slate-300"
      >
        {busy && <span className="size-3 animate-spin rounded-full border-2 border-white/40 border-t-white" />}
        {busy ? 'Working on it…' : children}
      </button>
      {note && <span className="text-[11px] text-slate-500">{note}</span>}
    </div>
  )
}

function Intro({ children }) {
  return <p className="text-[13px] leading-relaxed text-slate-600">{children}</p>
}

function HowItWorks({ children }) {
  return (
    <details className="group rounded-md text-[11px] text-slate-500">
      <summary className="inline-flex cursor-pointer select-none items-center gap-1 font-medium text-slate-500 hover:text-slate-700">
        <span className="grid size-3.5 place-items-center rounded-full ring-1 ring-slate-300 text-[9px]">?</span> How does this work?
      </summary>
      <p className="mt-1 leading-relaxed">{children}</p>
    </details>
  )
}

function Markers({ list, onCite, sources }) {
  const by = new Map((sources || []).map((s) => [s.marker, s]))
  return (list || []).map((n) => (
    <button key={n} type="button" onClick={() => by.get(n) && onCite(by.get(n))} title="Open this source" className="ml-1 rounded bg-indigo-50 px-1 font-mono text-[10px] font-semibold text-indigo-700 ring-1 ring-indigo-200 hover:bg-indigo-100">
      [{n}]
    </button>
  ))
}

function SectionChip({ section, title, tone = 'slate', strike = false }) {
  const tones = {
    slate: 'bg-white text-slate-700 ring-slate-200',
    law: 'bg-indigo-50 text-indigo-800 ring-indigo-200',
    variation: 'bg-slate-100 text-slate-500 ring-slate-200',
    bad: 'bg-red-50 text-red-800 ring-red-300',
  }
  return (
    <span title={title || undefined} className={`inline-flex max-w-full items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-semibold ring-1 ${tones[tone]} ${strike ? 'line-through opacity-60' : ''}`}>
      s.{section}
      {title && <span className="truncate font-normal opacity-80">{title.length > 34 ? `${title.slice(0, 32)}…` : title}</span>}
    </span>
  )
}

// --- How it decided -------------------------------------------------------------------------------------------

function Step({ n, title, children, last = false }) {
  return (
    <li className="relative flex gap-3">
      <div className="flex flex-col items-center">
        <span className="grid size-6 shrink-0 place-items-center rounded-full bg-indigo-600 text-[11px] font-bold text-white">{n}</span>
        {!last && <span className="mt-1 w-px flex-1 bg-indigo-200" />}
      </div>
      <div className="min-w-0 flex-1 pb-4">
        <p className="mb-1 text-[13px] font-semibold text-slate-900">{title}</p>
        {children}
      </div>
    </li>
  )
}

const HOLDS = {
  yes: ['bg-emerald-100 text-emerald-700', Icon.check, 'Shown by the facts'],
  no: ['bg-red-100 text-red-700', Icon.cross, 'Not shown'],
  unclear: ['bg-amber-100 text-amber-700', Icon.dot, 'Unclear'],
}

function Reasoning({ data, facts, onTest, testing, onCite, sources }) {
  if (data.error) return <p className="text-red-700">{data.error}</p>
  const applies = data.rules.filter((r) => r.applies)
  const not = data.rules.filter((r) => !r.applies)
  return (
    <div className="space-y-4">
      <ol>
        <Step n="1" title="What is the legal question?">
          <p className="text-[13px] text-slate-700">{data.issue}</p>
        </Step>
        <Step n="2" title="Which laws did it consider?">
          <div className="space-y-1.5">
            {applies.map((r) => (
              <div key={r.section} className="flex gap-2 rounded-lg bg-emerald-50/70 p-2 ring-1 ring-emerald-200">
                <span className="mt-0.5 grid size-4 shrink-0 place-items-center rounded-full bg-emerald-600 text-white">{Icon.check}</span>
                <p className="text-[12.5px] text-slate-700">
                  <b className="font-semibold text-slate-900">Applies: s.{r.section}</b> {r.title && <span className="text-slate-500">({r.title})</span>}
                  <br />
                  {r.why}
                  <Markers list={r.sources} onCite={onCite} sources={sources} />
                </p>
              </div>
            ))}
            {not.length > 0 && (
              <details className="rounded-lg bg-white p-2 ring-1 ring-slate-200">
                <summary className="cursor-pointer select-none text-[12px] text-slate-600">
                  Also considered but not applied: {not.map((r) => `s.${r.section}`).join(', ')}
                </summary>
                <ul className="mt-1.5 space-y-1">
                  {not.map((r) => (
                    <li key={r.section} className="text-[12px] text-slate-600">
                      <b className="font-semibold">s.{r.section}</b> {r.title && <span className="text-slate-400">({r.title})</span>}: {r.why}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        </Step>
        <Step n="3" title="Do the facts meet what the law requires?">
          <ul className="space-y-1.5">
            {data.application.map((a, i) => {
              const [cls, icon, word] = HOLDS[a.holds]
              return (
                <li key={i} className="grid gap-1 rounded-lg bg-white p-2 ring-1 ring-slate-200 sm:grid-cols-[1fr_auto_1fr] sm:items-center sm:gap-3">
                  <span className="text-[12.5px] text-slate-800">“{a.fact}”</span>
                  <span className={`inline-flex w-fit items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold ${cls}`}>
                    {icon}
                    {word}
                  </span>
                  <span className="text-[12px] text-slate-600">
                    {a.element}
                    {a.section && <span className="text-slate-400"> · s.{a.section}</span>}
                    <Markers list={a.sources} onCite={onCite} sources={sources} />
                  </span>
                </li>
              )
            })}
          </ul>
        </Step>
        <Step n="4" title="Conclusion" last>
          <p className="rounded-lg bg-indigo-50 p-2.5 text-[13px] font-medium text-indigo-950 ring-1 ring-indigo-200">{data.conclusion}</p>
          {data.key_facts.length > 0 && (
            <p className="mt-2 flex flex-wrap items-center gap-1.5 text-[12px] text-slate-600">
              It says this depends on:
              {data.key_facts.map((k) => (
                <span key={k} className="rounded-full bg-white px-2 py-0.5 text-slate-800 ring-1 ring-slate-300">“{k}”</span>
              ))}
            </p>
          )}
        </Step>
      </ol>

      <div className="rounded-xl bg-white p-3 ring-1 ring-slate-200">
        <p className="text-[13px] font-semibold text-slate-900">Can we trust this explanation?</p>
        <p className="mt-0.5 text-[12px] text-slate-600">
          An AI can describe reasoning it did not really use. To check, each fact it says it relied on is taken out of the question and
          the answer is written again: if the fact really mattered, the answer changes.
        </p>
        <div className="mt-2.5">
          {!facts ? (
            <RunButton onClick={onTest} busy={testing} note="takes about a minute">Check the explanation</RunButton>
          ) : facts.error ? (
            <p className="text-red-700">{facts.error}</p>
          ) : (
            <TrustResult data={facts} />
          )}
        </div>
      </div>
    </div>
  )
}

function TrustResult({ data }) {
  const named = data.facts.filter((r) => r.named_by_trace)
  const ok = named.filter((r) => r.influential).length
  const share = named.length ? ok / named.length : null
  const [tone, head, line] =
    share == null
      ? ['slate', 'Could not be checked', 'The explanation named no facts that could be tested.']
      : share >= 0.75
        ? ['good', 'Trustworthy', 'The facts it said it relied on really drove the answer.']
        : share >= 0.4
          ? ['warn', 'Partly trustworthy', 'Some of the facts it named really drove the answer; others made no difference.']
          : ['bad', 'Not reliable here', 'The facts it named did not change the answer when removed, so the explanation does not show what really drove it.']
  const bar = { good: 'bg-emerald-500', warn: 'bg-amber-500', bad: 'bg-red-500', slate: 'bg-slate-300' }[tone]
  return (
    <div className="space-y-2.5">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex gap-1" aria-hidden="true">
          {named.map((r) => (
            <span key={r.fact} className={`h-2.5 w-7 rounded-full ${r.influential ? bar : 'bg-slate-200'}`} />
          ))}
        </div>
        <p className="text-[13px]">
          <b className="font-semibold text-slate-900">{head}</b>
          <span className="text-slate-600">
            {' '}· {ok} of {named.length} facts confirmed. {line}
          </span>
        </p>
      </div>
      <ul className="space-y-1">
        {data.facts.map((r) => (
          <li key={r.fact} className="flex flex-wrap items-center gap-2 text-[12px]">
            <span className={`grid size-4 place-items-center rounded-full ${r.influential ? 'bg-emerald-600 text-white' : 'bg-slate-200 text-slate-500'}`}>
              {r.influential ? Icon.check : Icon.cross}
            </span>
            <span className="text-slate-700">Without “{r.fact}”</span>
            <span className="text-slate-500">
              {!r.changed
                ? 'the answer stays the same'
                : r.refusal_changed
                  ? 'the answer is declined'
                  : [r.lost.length && `it no longer relies on s.${r.lost.join(', s.')}`, r.gained.length && `it adds s.${r.gained.join(', s.')}`].filter(Boolean).join('; ')}
            </span>
            {!r.named_by_trace && <Badge>not named by the explanation</Badge>}
          </li>
        ))}
      </ul>
      <HowItWorks>
        The question is also asked again unchanged (a consistency check). Any change that also appears then is the AI's normal
        variation and is not counted as a fact's influence.
        {data.control?.changed && ` Here, asking again already changed ${[...data.control.lost, ...data.control.gained].map((s) => `s.${s}`).join(', ') || 'whether it answers'}.`}
      </HowItWorks>
    </div>
  )
}

// --- Words that matter -------------------------------------------------------------------------------------------

function KeyWords({ data, question }) {
  const [sec, setSec] = useState(data.sections[0]?.marker)
  const m = String(sec)
  const s0 = data.sections.find((s) => s.marker === sec)
  const max = Math.max(0.02, ...data.words.map((w) => Math.max(0, w.match_drop[m] || 0)))
  const spans = []
  let at = 0
  for (const w of [...data.words].sort((a, b) => a.start - b.start)) {
    if (w.start > at) spans.push({ text: question.slice(at, w.start) })
    spans.push({ text: question.slice(w.start, w.end), w })
    at = w.end
  }
  spans.push({ text: question.slice(at) })
  const top = [...data.words].sort((a, b) => (b.match_drop[m] || 0) - (a.match_drop[m] || 0)).filter((w) => (w.match_drop[m] || 0) > 0.003).slice(0, 5)
  const lex = data.words.filter((w) => w.lexicon.length)
  const crimeWord = [...data.words].sort((a, b) => (b.scope_drop || 0) - (a.scope_drop || 0))[0]
  return (
    <div className="space-y-3">
      <Intro>Which words in the question led the system to this law? Pick a section to see the words that linked the question to it.</Intro>
      <label className="flex flex-wrap items-center gap-2 text-[12px] text-slate-600">
        Section:
        <select value={sec} onChange={(e) => setSec(Number(e.target.value))} className="max-w-full rounded-lg border border-slate-200 bg-white px-2 py-1 text-[12px] text-slate-800">
          {data.sections.map((s) => (
            <option key={s.marker} value={s.marker}>
              s.{s.section} {s.title}
            </option>
          ))}
        </select>
      </label>
      <p className="rounded-xl bg-white p-3 text-[15px] leading-8 text-slate-800 ring-1 ring-slate-200" data-testid="word-heatmap">
        {spans.map((s, i) => {
          if (!s.w) return <span key={i}>{s.text}</span>
          const v = Math.max(0, s.w.match_drop[m] || 0)
          const a = Math.min(1, v / max)
          return (
            <span key={i} title={`Without this word the link to s.${s0?.section} weakens by ${Math.round(v * 100)} points`} style={{ background: `rgba(79,70,229,${0.06 + 0.55 * a})`, color: a > 0.6 ? 'white' : undefined }} className="rounded px-0.5">
              {s.text}
            </span>
          )
        })}
      </p>
      <div className="flex items-center gap-2 text-[11px] text-slate-500">
        <span>weak link</span>
        <span className="h-2 w-28 rounded-full bg-gradient-to-r from-indigo-100 to-indigo-600" />
        <span>strong link</span>
      </div>
      {top.length > 0 && (
        <div className="space-y-1">
          <p className="text-[12px] font-semibold text-slate-800">The words that mattered most for s.{s0?.section}</p>
          {top.map((w) => (
            <div key={w.start} className="flex items-center gap-2 text-[12px]">
              <span className="w-28 truncate text-right font-medium text-slate-800">{w.word}</span>
              <span className="h-2 rounded-full bg-indigo-500" style={{ width: `${Math.max(6, (w.match_drop[m] / max) * 60)}%` }} />
            </div>
          ))}
        </div>
      )}
      <div className="grid gap-2 sm:grid-cols-2">
        {data.base.scope_probability != null && (
          <div className="rounded-lg bg-white p-2.5 ring-1 ring-slate-200">
            <p className="text-[11px] text-slate-500">Is this about a crime?</p>
            <p className="text-[13px] text-slate-800">
              <b className="font-semibold">{Math.round(data.base.scope_probability * 100)}% likely</b>
              {crimeWord && crimeWord.scope_drop > 0.01 && <span className="text-slate-600">, mostly because of “{crimeWord.word}”</span>}
            </p>
          </div>
        )}
        {lex.length > 0 && (
          <div className="rounded-lg bg-white p-2.5 ring-1 ring-slate-200">
            <p className="text-[11px] text-slate-500">Everyday words it translated into legal terms</p>
            {lex.slice(0, 3).map((w) => (
              <p key={w.start} className="text-[12px] text-slate-700">
                “{w.word}” → <i>{w.lexicon[0].split(':')[0]}</i>
              </p>
            ))}
          </div>
        )}
      </div>
      <HowItWorks>
        Each word is removed in turn and the question is matched against the law again; the more the match drops, the more that
        word mattered. No AI model is involved, so this shows how the search itself read the question.
        {data.rewrite_gain?.[m] != null && ` The AI's restatement of the question added ${Math.round(data.rewrite_gain[m] * 100)} points to this section's match.`}
      </HowItWorks>
    </div>
  )
}

// --- Law it relied on ----------------------------------------------------------------------------------------------

const list = (xs) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs.at(-1)}`)

function SourceResults({ data }) {
  const decisive = data.sources.filter((r) => r.decisive)
  return (
    <div className="space-y-3">
      <p className="rounded-xl bg-white p-3 text-[13px] text-slate-800 ring-1 ring-slate-200">
        {decisive.length ? (
          <>
            The conclusion rests on <b className="font-semibold">{list(decisive.map((r) => `s.${r.section}`))}</b>: without{' '}
            {decisive.length > 1 ? 'them' : 'it'}, the answer changes.
          </>
        ) : (
          'No single section was decisive: the answer stayed the same whichever one was taken away.'
        )}
      </p>
      <ul className="space-y-1.5">
        {data.sources.map((r) => (
          <li key={r.section} className={`flex flex-wrap items-center gap-2 rounded-lg p-2 ring-1 ${r.decisive ? 'bg-emerald-50/70 ring-emerald-200' : 'bg-white ring-slate-200'}`}>
            <span className={`h-8 w-1.5 shrink-0 rounded-full ${r.decisive ? 'bg-emerald-500' : 'bg-slate-200'}`} />
            <div className="min-w-0 flex-1">
              <p className="text-[12.5px] text-slate-800">
                <b className="font-semibold">s.{r.section}</b> <span className="text-slate-500">{r.title}</span>
              </p>
              <p className="text-[11.5px] text-slate-500">
                {r.decisive
                  ? `The answer depends on this. Without it, ${r.refusal_changed ? 'the question is declined' : `the answer ${[r.lost.length && `no longer relies on ${list(r.lost.map((x) => `s.${x}`))}`, r.gained.length && `turns to ${list(r.gained.map((x) => `s.${x}`))}`].filter(Boolean).join(' and ')}`}.`
                  : 'Not needed: the answer stays the same without it.'}
              </p>
            </div>
          </li>
        ))}
      </ul>
      <HowItWorks>
        Each section is taken away in turn and the answer is written again from the rest. A section is decisive if the answer changes
        without it. The question is also asked again unchanged, so the AI's normal variation is not mistaken for a section's effect.
      </HowItWorks>
    </div>
  )
}

// --- Fairness ---------------------------------------------------------------------------------------------------------

export const ATTRIBUTES = [['gender', 'Gender'], ['religion', 'Religion'], ['caste', 'Caste'], ['income', 'Income'], ['region', 'Region'], ['age', 'Age']]
const NOUN = { gender: 'gender', religion: 'religion', caste: 'caste', income: 'income level', region: 'region', age: 'age' }
const SWAPPED = {
  gender: 'gender (husband and wife, he and she, names)',
  religion: 'religion (names and places)',
  caste: 'caste (caste words and surnames that suggest caste, which is only a rough signal)',
  income: 'income level (words such as wealthy and poor, never amounts of money)',
  region: 'region (where the person is from)',
  age: 'age (stated ages and words such as boy or elderly)',
}
const biasKey = (a) => (a === 'gender' ? 'bias' : a === 'religion' ? 'biasReligion' : `bias_${a}`)

const VERDICT = {
  no_difference: ['good', 'No bias detected', 'Swapping the {noun} gave the same answer.'],
  law_required: ['info', 'No bias: the law itself makes the difference', 'The answer changed only where the law itself depends on {noun}.'],
  variation: ['slate', 'No bias detected', 'The small differences also appear when the same question is simply asked again.'],
  possible_bias: ['bad', 'Possible bias', 'Changing only the {noun} changed the answer in a way the law does not explain.'],
  not_applicable: ['slate', 'Nothing to check', 'The question says nothing about {noun} to swap.'],
}
const VERDICT_STYLE = {
  good: ['bg-emerald-50 ring-emerald-200', 'bg-emerald-600'],
  info: ['bg-indigo-50 ring-indigo-200', 'bg-indigo-600'],
  slate: ['bg-slate-50 ring-slate-200', 'bg-slate-500'],
  bad: ['bg-red-50 ring-red-200', 'bg-red-600'],
}
const KIND = {
  law: ['law', 'required by law'],
  consequence: ['law', 'follows from the law'],
  variation: ['variation', 'normal variation'],
  unexplained: ['bad', 'unexplained'],
}

function marked(text, words) {
  if (!text || !words.length) return text
  const esc = words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
  const re = new RegExp(`\\b(${esc.join('|')})\\b`, 'g')
  return text.split(re).map((part, i) => (i % 2 ? <mark key={i} className="rounded bg-amber-200/70 px-0.5 text-slate-900">{part}</mark> : part))
}

function VersionCard({ label, text, run, kinds, words, titles }) {
  return (
    <div className="flex min-w-0 flex-col gap-2 rounded-xl bg-white p-3 ring-1 ring-slate-200">
      <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">{label}</p>
      {text && <p className="max-h-28 overflow-auto whitespace-pre-wrap text-[12.5px] leading-relaxed text-slate-700">{marked(text, words)}</p>}
      <div className="mt-auto">
        <p className="mb-1 text-[11px] text-slate-500">{run.refused ? 'Declined' : 'Relies on'}</p>
        <div className="flex flex-wrap gap-1">
          {run.cited.length === 0 && !run.refused && <span className="text-[12px] text-slate-400">no section</span>}
          {run.cited.map((s) => (
            <SectionChip key={s} section={s} title={titles[s]} tone={kinds[s] || 'slate'} />
          ))}
        </div>
        {Object.keys(run.elements || {}).length > 0 && (
          <p className="mt-1.5 text-[11.5px] text-slate-500">
            {Object.entries(run.elements).map(([k, v]) => `${k}: ${v.toLowerCase()}`).join(' · ')}
          </p>
        )}
      </div>
    </div>
  )
}

export function BiasResult({ data, attribute, titles = {} }) {
  if (data.error) return <p className="text-red-700">{data.error}</p>
  const noun = NOUN[attribute] || attribute
  const [tone, head, rawLine] = VERDICT[data.verdict] || ['slate', data.verdict, '']
  const line = rawLine.replace('{noun}', noun)
  const [box, dot] = VERDICT_STYLE[tone]
  const added = data.mode === 'added'
  const kinds = {}
  const t = { ...titles }
  for (const d of data.differences || []) {
    if (d.side?.includes('only')) kinds[d.section] = KIND[d.kind][0]
    if (d.title && !d.title.startsWith('s.')) t[d.section] = t[d.section] || d.title
  }
  const thing = Object.keys(data.original?.elements || {}).length ? 'case file' : 'question'
  const consistent = data.control && JSON.stringify([...data.control.cited].sort()) === JSON.stringify([...data.original.cited].sort())
  return (
    <div className="space-y-3" data-testid="bias-result">
      <div className={`flex gap-3 rounded-xl p-3 ring-1 ${box}`}>
        <span className={`mt-0.5 grid size-7 shrink-0 place-items-center rounded-full text-white ${dot}`}>{tone === 'bad' ? '!' : Icon.check}</span>
        <div>
          <p className="text-[14px] font-semibold text-slate-900">{head}</p>
          <p className="text-[12.5px] text-slate-700">{line}</p>
        </div>
      </div>

      {added && (
        <p className="text-[12px] text-slate-600">
          Your {thing} does not mention {noun}, so one sentence about the accused was added to two otherwise identical copies. Only that sentence differs.
        </p>
      )}
      {data.original && data.swapped && (
        <div className="grid gap-2 sm:grid-cols-2">
          <VersionCard label={added ? `Accused: ${data.swaps[0].from}${attribute === 'age' ? ' years old' : ''}` : `Your ${thing}`} text={data.original.text} run={data.original} kinds={kinds} words={data.swaps.map((s) => s.from)} titles={t} />
          <VersionCard label={added ? `Accused: ${data.swaps[0].to}${attribute === 'age' ? ' years old' : ''}` : `Same ${thing}, ${noun} swapped`} text={data.swapped_text} run={data.swapped} kinds={kinds} words={data.swaps.map((s) => s.to)} titles={t} />
        </div>
      )}

      {data.differences?.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-[12px] font-semibold text-slate-800">What changed, and why</p>
          {data.differences.map((d, i) => (
            <div key={i} className="flex flex-wrap items-center gap-2 text-[12px]">
              <SectionChip section={d.section} tone={KIND[d.kind][0]} />
              <span className="text-slate-500">{d.side.includes('only') ? (d.side.startsWith('original') ? (added ? 'only in the first version' : `only in your ${thing}`) : (added ? 'only in the second version' : 'only in the swapped one')) : d.side}</span>
              <span className="text-slate-700">
                · {d.kind === 'law' ? `the law defines it by ${noun}` : d.kind === 'consequence' ? `it replaces a section the law defines by ${noun}` : d.kind === 'variation' ? 'it also changes when the same question is asked again' : `nothing in the law explains this`}
              </span>
            </div>
          ))}
          <div className="flex flex-wrap gap-2 pt-1 text-[11px] text-slate-500">
            <span className="inline-flex items-center gap-1"><span className="size-2.5 rounded-sm bg-indigo-200 ring-1 ring-indigo-300" /> required by law</span>
            <span className="inline-flex items-center gap-1"><span className="size-2.5 rounded-sm bg-slate-200 ring-1 ring-slate-300" /> normal variation</span>
            <span className="inline-flex items-center gap-1"><span className="size-2.5 rounded-sm bg-red-200 ring-1 ring-red-300" /> unexplained (possible bias)</span>
          </div>
        </div>
      )}

      {data.law_notes?.length > 0 && (
        <div className="rounded-lg bg-amber-50 p-2.5 text-[12px] text-amber-900 ring-1 ring-amber-200">
          <p className="font-semibold">Worth checking</p>
          {data.law_notes.map((n) => (
            <p key={n.section || n.why}>
              {n.plain
                ? `${n.why}.`
                : attribute === 'gender'
                  ? `s.${n.section} applies only to ${/woman|wife|girl/i.test(n.why) ? 'a woman' : 'one gender'} in law, yet both versions rely on it. In one of them it may be applied wrongly.`
                  : `s.${n.section} depends on ${noun} in law, yet both versions rely on it. In one of them it may be applied wrongly.`}
            </p>
          ))}
        </div>
      )}
      {data.stereotypes?.length > 0 && (
        <div className="rounded-lg bg-red-50 p-2.5 text-[12px] text-red-800 ring-1 ring-red-200">
          <p className="font-semibold">Stereotyped wording found</p>
          {data.stereotypes.map((s, i) => (
            <p key={i}>“{s.sentence}” ({s.why})</p>
          ))}
        </div>
      )}
      {data.control && (
        <p className="flex items-center gap-1.5 text-[12px] text-slate-600">
          <span className={`grid size-4 place-items-center rounded-full ${consistent ? 'bg-emerald-600 text-white' : 'bg-slate-300 text-white'}`}>{consistent ? Icon.check : Icon.dot}</span>
          Consistency check: run again unchanged, the result {consistent ? 'was the same' : 'varied a little'}.
        </p>
      )}
      <HowItWorks>
        Only the people's {SWAPPED[attribute] || noun} is changed and everything else stays the same.
        {attribute === 'gender' && ' Some offences are defined by gender in the law itself (dowry death, cruelty by a husband, rape), so a change in those is what the law requires.'}
        {attribute === 'age' && ' Some offences depend on age in the law itself (a child under seven, kidnapping a minor), so a change in those is what the law requires.'}
        {' '}If the two differ, each is run again, and only a difference that holds both times counts, so the AI's normal variation is not mistaken for bias.
      </HowItWorks>
    </div>
  )
}

// --- the panel -------------------------------------------------------------------------------------------------------

function Tile({ id, active, onClick, icon, title, children, tone }) {
  const ring = { good: 'ring-emerald-300', warn: 'ring-amber-300', bad: 'ring-red-300' }[tone] || 'ring-slate-200'
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      data-tile={id}
      onClick={onClick}
      className={`flex min-w-0 flex-col gap-1 rounded-xl bg-white p-2.5 text-left ring-1 transition hover:-translate-y-px hover:shadow-sm ${active ? 'ring-2 ring-indigo-500' : ring}`}
    >
      <span className="flex items-center gap-1.5 text-[11.5px] font-semibold text-slate-800">
        <span className="text-indigo-600">{icon}</span>
        {title}
      </span>
      <span className="line-clamp-2 text-[11.5px] leading-snug text-slate-500">{children}</span>
    </button>
  )
}

export default function ExplainPanel({ msg, onUpdate, onCite }) {
  const ex = msg.explain || {}
  const [tab, setTab] = useState(ex.tab || 'reasoning')
  const [busy, setBusy] = useState({})
  const [attribute, setAttribute] = useState('gender')
  const sensitivityAsked = useRef(false)
  const base = { question: msg.question, answer: msg.content, sources: msg.sources, model: msg.modelKey || '', law: msg.law || 'auto' }
  const save = (patch) => onUpdate((m) => ({ ...m, explain: { ...(m.explain || {}), ...patch } }))

  const run = async (key, fn) => {
    setBusy((b) => ({ ...b, [key]: true }))
    try {
      save({ [key]: await fn() })
    } catch (e) {
      save({ [key]: { error: e.message.replace(/^HTTP \d+: /, '') } })
    } finally {
      setBusy((b) => ({ ...b, [key]: false }))
    }
  }
  const runBias = (attr = attribute) =>
    run(biasKey(attr), () => explainApi.bias({ ...base, refused: !!msg.refused, attribute: attr, control: true }))
  const runWords = () =>
    !ex.words &&
    !busy.words &&
    run('words', () => explainApi.words({ ...base, search_text: msg.retrieval?.translation?.text || msg.retrieval?.searched || null, rewrite_text: msg.retrieval?.rewrite?.used ? msg.retrieval.rewrite.text : null }))

  // gender-sensitive questions (dowry, cruelty, sexual offences...): the bias check runs by itself
  useEffect(() => {
    if (sensitivityAsked.current || ex.sensitive !== undefined || !msg.question) return
    sensitivityAsked.current = true
    const cited = (msg.sources || []).filter((s) => s.kind !== 'judgment').map((s) => s.section)
    explainApi
      .sensitive({ text: msg.question, sections: cited, law: msg.retrieval?.law === 'bns' ? 'bns' : 'ipc' })
      .then((r) => {
        const attr = r.attribute || 'gender'
        save({ sensitive: r.reason || null, sensitiveAttr: attr })
        if (r.reason && !ex[biasKey(attr)]) {
          setTab('bias')
          setAttribute(attr)
          runBias(attr)
          runWords()
        }
      })
      .catch(() => save({ sensitive: null }))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const statute = (msg.sources || []).filter((s) => s.kind !== 'judgment')
  const titles = Object.fromEntries(statute.map((s) => [s.section, s.title]))
  const biasData = ex[biasKey(attribute)]
  const shownBias = biasData || ex[biasKey(ex.sensitiveAttr || 'gender')]
  const topWords = ex.words?.words
    ? [...ex.words.words].sort((a, b) => Math.max(...Object.values(b.match_drop)) - Math.max(...Object.values(a.match_drop))).slice(0, 3).map((w) => `“${w.word}”`)
    : null
  const decisive = ex.sources?.sources?.filter((r) => r.decisive).map((r) => `s.${r.section}`)
  const verdict = shownBias?.verdict && VERDICT[shownBias.verdict]
  return (
    <details
      className="mt-3 rounded-xl bg-slate-50 text-xs ring-1 ring-slate-200"
      open={!!ex.sensitive || undefined}
      onToggle={(e) => e.currentTarget.open && runWords()}
      data-testid="explain-panel"
    >
      <summary className="flex cursor-pointer select-none flex-wrap items-center gap-2 px-3 py-2.5">
        <span className="text-[13px] font-semibold text-slate-800">Understand this answer</span>
        <span className="text-slate-500">how it was decided · what mattered · bias check</span>
        {verdict && <Badge tone={verdict[0] === 'bad' ? 'bad' : verdict[0] === 'good' ? 'good' : 'info'}>Bias check: {verdict[1]}</Badge>}
      </summary>
      <div className="space-y-3 border-t border-slate-200 px-3 py-3">
        {ex.sensitive && (
          <p className="rounded-lg bg-amber-50 px-3 py-2 text-[12px] text-amber-900 ring-1 ring-amber-200">
            {ex.sensitiveAttr === 'caste' ? 'This question touches on caste' : 'This question touches on gender (dowry, marriage or a gender-specific offence)'}, so the bias check ran by itself.
          </p>
        )}
        <div className="grid grid-cols-2 gap-2 lg:grid-cols-4" role="tablist">
          <Tile id="reasoning" active={tab === 'reasoning'} onClick={() => setTab('reasoning')} icon={Icon.steps} title="How it decided"
            tone={ex.facts?.facts && (ex.facts.faithfulness.key_facts_confirmed >= Math.max(1, ex.facts.faithfulness.key_facts * 0.75) ? 'good' : 'warn')}>
            {ex.reasoning?.conclusion || 'Step by step, from the question to the conclusion'}
          </Tile>
          <Tile id="words" active={tab === 'words'} onClick={() => { setTab('words'); runWords() }} icon={Icon.words} title="Words that mattered">
            {topWords?.length ? topWords.join(', ') : busy.words ? 'Working…' : 'Which words led to this law'}
          </Tile>
          <Tile id="sources" active={tab === 'sources'} onClick={() => setTab('sources')} icon={Icon.law} title="Law it relied on">
            {decisive ? (decisive.length ? `Rests on ${decisive.join(', ')}` : 'No single section decisive') : 'Which sections the answer depends on'}
          </Tile>
          <Tile id="bias" active={tab === 'bias'} onClick={() => setTab('bias')} icon={Icon.scale} title="Bias check"
            tone={verdict ? (shownBias.verdict === 'possible_bias' ? 'bad' : 'good') : undefined}>
            {verdict ? verdict[1] : busy[biasKey(attribute)] ? 'Checking for bias…' : 'Change gender, caste, income…: does the answer change?'}
          </Tile>
        </div>

        <div className="rounded-xl bg-white/60 p-3 ring-1 ring-slate-200">
          {tab === 'reasoning' &&
            (!ex.reasoning ? (
              <div className="space-y-2.5">
                <Intro>See how the AI went from your question to its answer: the legal question, the laws it considered, whether your facts meet what each law requires, and its conclusion.</Intro>
                <RunButton onClick={() => run('reasoning', () => explainApi.reasoning(base))} busy={busy.reasoning} note="a few seconds">
                  Show how it decided
                </RunButton>
              </div>
            ) : (
              <Reasoning
                data={ex.reasoning}
                facts={ex.facts}
                sources={msg.sources}
                onCite={onCite}
                testing={busy.facts}
                onTest={() => run('facts', () => explainApi.facts({ ...base, key_facts: ex.reasoning.key_facts || [] }))}
              />
            ))}

          {tab === 'words' &&
            (!ex.words ? (
              <RunButton onClick={runWords} busy={busy.words} note="instant, no AI involved">Show the words that mattered</RunButton>
            ) : ex.words.error ? (
              <p className="text-red-700">{ex.words.error}</p>
            ) : (
              <KeyWords data={ex.words} question={msg.retrieval?.translation?.text || msg.retrieval?.searched || msg.question} />
            ))}

          {tab === 'sources' &&
            (!ex.sources ? (
              <div className="space-y-2.5">
                <Intro>Which laws did the answer actually depend on? Each section is taken away in turn to see whether the answer changes.</Intro>
                <RunButton onClick={() => run('sources', () => explainApi.sources(base))} busy={busy.sources} note={`about ${new Set(statute.map((s) => s.section)).size + 1} short AI calls`}>
                  Find the deciding sections
                </RunButton>
              </div>
            ) : ex.sources.error ? (
              <p className="text-red-700">{ex.sources.error}</p>
            ) : (
              <SourceResults data={ex.sources} />
            ))}

          {tab === 'bias' && (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2 text-[12px]">
                <span className="text-slate-600">Bias check: swap the people's</span>
                {ATTRIBUTES.map(([a, label]) => (
                  <button key={a} type="button" onClick={() => setAttribute(a)} className={`rounded-full px-2.5 py-1 ring-1 ${attribute === a ? 'bg-indigo-600 text-white ring-indigo-600' : 'bg-white text-slate-600 ring-slate-200'}`}>
                    {label}
                    {ex[biasKey(a)]?.verdict === 'possible_bias' && ' !'}
                  </button>
                ))}
              </div>
              {!biasData ? (
                <div className="space-y-2.5">
                  <Intro>Bias detection: would the answer be different if the people in your question were of a different {NOUN[attribute]}? The same question is asked again with only that changed, and every difference is explained.</Intro>
                  <RunButton onClick={() => runBias(attribute)} busy={busy[biasKey(attribute)]} note="about a minute">
                    Run the bias check
                  </RunButton>
                </div>
              ) : (
                <BiasResult data={biasData} attribute={attribute} titles={titles} />
              )}
            </div>
          )}
        </div>
      </div>
    </details>
  )
}
