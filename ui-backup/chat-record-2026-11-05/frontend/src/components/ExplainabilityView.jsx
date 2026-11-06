import { useEffect, useRef, useState } from 'react'
import { explainApi } from '../api.js'
import { useSeen } from './motion.jsx'
import { Scales } from './LegalArt.jsx'

// The Explainability page, as a cross-examination: five plain questions put to the system, one at a
// time. Each answer is one sentence, one measured figure and a way to try it; the full numbers wait
// behind "See the evidence" (docs/reports/explainability_results.json, built by
// backend/scripts/make_explain_report.py). The tools themselves live under each answer in Chat.

const pct = (x) => (x == null ? '—' : `${Math.round(x * 100)}%`)
const sys = (k) => (k === 'closed' ? 'Model alone' : 'LegalGPT')

function Figure({ value, label }) {
  return (
    <div>
      <p className="text-[clamp(3.25rem,2.4rem+3vw,5.5rem)] font-semibold leading-[0.95] tracking-[-0.05em] [font-variant-numeric:tabular-nums] text-slate-900">{value}</p>
      <p className="mt-2 max-w-md text-[17px] leading-snug text-slate-700">{label}</p>
    </div>
  )
}

function Small({ value, label }) {
  return (
    <div className="bg-slate-100 p-3.5">
      <p className="text-2xl font-semibold tracking-[-0.03em] [font-variant-numeric:tabular-nums] text-slate-900">{value}</p>
      <p className="mt-1 text-[13px] leading-snug text-slate-600">{label}</p>
    </div>
  )
}

/** Two systems side by side, out of the same scenarios. */
function Bars({ title, rows }) {
  const ref = useRef(null)
  const seen = useSeen(ref, '0px')
  return (
    <div ref={ref}>
      <p className="text-[15px] font-medium text-slate-900">{title}</p>
      <div className="mt-1.5 space-y-1.5">
        {rows.map(([name, n, of, tone]) => (
          <div key={name} className="grid grid-cols-[6.5rem_minmax(0,1fr)_4.5rem] items-center gap-2 text-[15px]">
            <span className="text-slate-700">{name}</span>
            <span className="h-4 bg-slate-100">
              <span className={`block h-full origin-left transition-transform duration-1000 ease-[cubic-bezier(0.22,1,0.36,1)] ${tone}`} style={{ transform: `scaleX(${seen ? n / of : 0})` }} />
            </span>
            <span className="text-right font-medium tabular-nums text-slate-900">{n} of {of}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function Evidence({ children }) {
  return (
    <details className="group mt-6 text-[15px] text-slate-700">
      <summary className="press inline-flex cursor-pointer select-none items-center gap-2 font-medium text-slate-900 underline underline-offset-4">
        See the evidence
        <span aria-hidden="true" className="transition-transform duration-200 group-open:rotate-90">→</span>
      </summary>
      <div className="mt-4 space-y-4 leading-relaxed">{children}</div>
    </details>
  )
}

function TryIt({ where }) {
  return (
    <p className="mt-6 flex flex-wrap gap-2">
      {where.map(([hash, label]) => (
        <a key={hash} href={`#${hash}`} className="press inline-flex h-11 items-center bg-ink px-4 text-[15px] font-medium text-paper hover:bg-field-red hover:text-ink">{label}</a>
      ))}
    </p>
  )
}

function Why({ r }) {
  const f = r?.reasoning?.facts
  return (
    <>
      {f && <Figure value={`${f.key_facts_confirmed} of ${f.key_facts}`} label="key facts it named really changed the conclusion when they were taken away" />}
      <TryIt where={[['chat', 'Try it on an answer']]} />
      {r?.reasoning && (
        <Evidence>
          <p>
            On {r.reasoning.questions} held-out scenarios the model wrote its reasoning in the IRAC form lawyers use, and each claim was
            tested by removing the fact or section it names. A control run, the same input asked again, separates real influence from
            run-to-run variation.
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            <Small value={pct(f.agreement_mean)} label="agreement: a fact is named exactly when it matters" />
            <Small value={pct(r.reasoning.trace.key_facts_verbatim)} label="key facts quoted word for word from the question" />
          </div>
          {f.unnamed > 0 && <p>Unnamed facts that still changed the conclusion: {f.unnamed_influential} of {f.unnamed}.</p>}
        </Evidence>
      )}
    </>
  )
}

function Words({ r }) {
  const w = r?.word_importance
  const k = w?.by_k?.['1']
  return (
    <>
      {k && <Figure value={pct(k.match_top_beats_random)} label="of questions: taking away the top word hurt the match with the law more than a random word" />}
      <TryIt where={[['chat', 'Try it on an answer']]} />
      {k && (
        <Evidence>
          <p>Deletion test on {w.questions} held-out scenarios. No model is called: only the search is run again.</p>
          <div className="grid gap-2 sm:grid-cols-2">
            <Small value={k.match_drop_top.toFixed(3)} label="match lost when the top word is removed" />
            <Small value={k.match_drop_random.toFixed(3)} label="match lost when a random word is removed" />
          </div>
          {r?.classifier && (
            <p>
              InLegalBERT's word explanations in the Label tab: on 100 held-out sentences, removing the top 10% of words lowers the predicted
              role's probability {r.classifier.comprehensiveness_at_10.occlusion} (word removal) and {r.classifier.comprehensiveness_at_10.integrated_gradients} (Integrated
              Gradients), against {r.classifier.comprehensiveness_at_10.random} for random words.
            </p>
          )}
        </Evidence>
      )}
    </>
  )
}

function Sources({ r }) {
  return (
    <>
      {r?.reasoning && <Figure value={pct(r.reasoning.sections.precision)} label="of the sections the reasoning says apply are decisive when left out" />}
      <TryIt where={[['chat', 'Try it on an answer']]} />
    </>
  )
}

function Bias({ r }) {
  const models = r?.bias_models ? Object.entries(r.bias_models) : []
  const attrs = r?.bias_attributes ? Object.entries(r.bias_attributes) : []
  const totals = Object.entries(r?.bias_attribute_totals || {})
  const flagged = (d) => Object.values(d.by_group).reduce((n, v) => n + v.possible_bias, 0)
  const pairs = (d) => Object.values(d.by_group).reduce((n, v) => n + v.no_difference + v.law_required + v.variation + v.possible_bias, 0)
  return (
    <>
      <div className="space-y-5">
        {totals.map(([m, t]) => (
          <Bars
            key={m}
            title={`${m}: caste, income, region and age`}
            rows={[['Model alone', t.closed, t.pairs, 'bg-slate-400'], ['LegalGPT', t.legalgpt, t.pairs, 'bg-field-red']]}
          />
        ))}
      </div>
      <p className="mt-4 text-[15px] leading-snug text-slate-700">
        Flagged as possible bias. LegalGPT is not consistently better here; the differences are not significant. These are signals to
        investigate, not proof either way.
      </p>
      <TryIt where={[['chat', 'Check an answer'], ['case', 'Check a case file']]} />
      {models.length > 0 && (
        <Evidence>
          <div id="bias" className="space-y-3">
            {models.map(([m, bias]) => (
              <Bars key={m} title={`${m}: gender swapped`} rows={Object.entries(bias).map(([s, d]) => [sys(s), flagged(d), pairs(d), s === 'closed' ? 'bg-slate-400' : 'bg-field-red'])} />
            ))}
          </div>
          <p>
            24 gender-sensitive scenarios and 16 where gender should not matter, each run with the genders swapped and again unchanged. A
            difference in a section the law defines by gender is what the law requires, not bias, and is not counted.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[34rem] text-left text-[13px]">
              <thead className="text-slate-500">
                <tr>
                  {['Model', '', 'Scenarios', 'No difference', 'Required by law', 'Variation', 'Possible bias'].map((h, i) => (
                    <th key={i} className="py-1.5 pr-3 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {models.flatMap(([m, bias]) => Object.entries(bias).flatMap(([s, d]) =>
                  Object.entries(d.by_group).map(([g, v]) => (
                    <tr key={m + s + g} className="border-t border-slate-200">
                      <td className="py-1.5 pr-3 text-slate-500">{m}</td>
                      <td className="py-1.5 pr-3 font-medium text-slate-800">{sys(s)}</td>
                      <td className="py-1.5 pr-3">{g === 'sensitive' ? 'sensitive' : 'neutral'}</td>
                      <td className="py-1.5 pr-3 tabular-nums">{v.no_difference}</td>
                      <td className="py-1.5 pr-3 tabular-nums">{v.law_required}</td>
                      <td className="py-1.5 pr-3 tabular-nums">{v.variation}</td>
                      <td className={`py-1.5 tabular-nums ${v.possible_bias ? 'font-semibold text-red-700' : ''}`}>{v.possible_bias}</td>
                    </tr>
                  )),
                ))}
              </tbody>
            </table>
          </div>
          {attrs.length > 0 && (
            <div id="bias-attributes" className="overflow-x-auto">
              <p className="mb-2">
                Caste, income, region and age: 10 scenarios each, identical except for one sentence about the accused. Both versions run twice,
                and a section counts only if it holds in both runs of one version and neither run of the other.
              </p>
              <table className="w-full min-w-[30rem] text-left text-[13px]">
                <thead className="text-slate-500">
                  <tr>
                    {['Model', 'Changed', 'Model alone', 'LegalGPT'].map((h) => (
                      <th key={h} className="py-1.5 pr-3 font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {attrs.flatMap(([m, list]) => Object.entries(list).map(([a, d]) => (
                    <tr key={m + a} className="border-t border-slate-200">
                      <td className="py-1.5 pr-3 text-slate-500">{m}</td>
                      <td className="py-1.5 pr-3 font-medium text-slate-800">{a === 'income' ? 'income level' : a}</td>
                      {['closed', 'legalgpt'].map((s) => (
                        <td key={s} className={`py-1.5 pr-3 tabular-nums ${d[s].possible_bias ? 'font-semibold text-red-700' : ''}`}>{d[s].possible_bias} of {d[s].pairs}</td>
                      ))}
                    </tr>
                  )))}
                </tbody>
              </table>
            </div>
          )}
          <ul className="space-y-1">
            {totals.map(([m, t]) => (
              <li key={m}>{m}: p = {t.mcnemar_p}, {t.stereotypes} stereotyped sentences, {t.refusal_changes} changed refusals.</li>
            ))}
          </ul>
        </Evidence>
      )}
    </>
  )
}

function Inside({ r }) {
  const l = r?.lora
  const inf = l?.behaviour?.influence
  return (
    <>
      {inf && <Figure value={`${inf.top_source_is_cited} of ${inf.answers_with_citations}`} label="answers: the source that most influenced the model, measured inside it, is one it cites" />}
      {l && (
        <Evidence>
          <div id="lora" className="space-y-4">
            <p>Following Section 7 of the LoRA paper (Hu et al., 2022), the adapter's weight update is analysed directly, and the tuned model is compared with the base model on its own answers.</p>
            {l.weights && (
              <>
                <img src="/explain/fig6_lora_layers.png" alt="Heat map of the relative weight change per layer and module" className="w-full" />
                <div className="grid gap-2 sm:grid-cols-3">
                  <Small value={`${l.weights.mean_eff_rank}/${l.weights.r}`} label="effective rank of the update" />
                  <Small value={`${(l.weights.by_module.gate_proj.rel * 100).toFixed(2)}%`} label="largest relative change (MLP gate projection)" />
                  <Small value={l.weights.mean_amplification} label="amplification of directions the base model already had" />
                </div>
              </>
            )}
            {l.behaviour && (
              <>
                <img src="/explain/fig7_lora_tokens.png" alt="What the adapter made more likely" className="w-full" />
                <Small value={`${inf.mean_drop_cited} vs ${inf.mean_drop_uncited}`} label="drop in the answer's log-probability when a cited vs an uncited source is removed" />
                <p>Tokens the adapter raised most: {l.behaviour.top_tokens_raised.slice(0, 8).map((t) => `“${t.token}”`).join(', ')}.</p>
              </>
            )}
          </div>
        </Evidence>
      )}
    </>
  )
}

const QUESTIONS = [
  { id: 'why', q: 'Why did you reach that conclusion?', a: 'It sets out its reasoning the way a lawyer does: the issue, each section, each fact against each element. Then every fact it named is taken away, to see whether the conclusion really depended on it.', Body: Why },
  { id: 'words', q: 'Which of my words mattered?', a: 'Each word of your question is removed in turn. The words that change the match with the law are the ones that counted.', Body: Words },
  { id: 'sources', q: 'What if a source were missing?', a: 'Each section is left out and the answer written again. A section is decisive if the conclusion changes without it.', Body: Sources },
  { id: 'bias', q: 'Would you answer differently for someone else?', a: 'The same facts are run again with only gender, religion, caste, income, region or age changed, and the sections it relied on are compared.', Body: Bias },
  { id: 'inside', q: 'What changed inside the model?', a: 'For our own fine-tuned model we can open the weights and measure which sources an answer truly depends on, instead of asking it.', Body: Inside },
]

export default function ExplainabilityView() {
  const [r, setR] = useState(null)
  const [err, setErr] = useState(null)
  const [at, setAt] = useState(0)
  const panel = useRef(null)
  useEffect(() => {
    explainApi.results().then(setR).catch((e) => setErr(e.message))
  }, [])
  const { q, a, Body, id } = QUESTIONS[at]

  return (
    <div className="light-scope min-h-0 flex-1 overflow-y-auto bg-field-bone text-ink">
      <div className="mx-auto max-w-6xl px-4 py-8 sm:px-8 sm:py-10">
        <header className="relative text-center">
          <Scales tilt={[-5, 4, -3, 6, -2][at]} className="mx-auto h-24 text-ink sm:h-28" />
          <h2 className="head mt-3">Cross-examine the answer</h2>
          <p className="standfirst mx-auto mt-3">A citation shows where an answer came from. Put these five questions to it, and see why.</p>
        </header>
        {err && <p className="mt-6 bg-amber-100 px-4 py-3 text-[15px] text-amber-900">Results could not be loaded: {err}</p>}

        <div className="mt-8 grid items-start gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,8fr)]">
          <ol role="tablist" aria-label="Questions" aria-orientation="vertical" className="grid gap-1.5">
            {QUESTIONS.map((item, i) => (
              <li key={item.id} role="presentation">
                <button
                  type="button"
                  role="tab"
                  aria-selected={i === at}
                  onClick={() => {
                    setAt(i)
                    // on a phone the answer sits below the questions
                    if (!window.matchMedia('(min-width: 1024px)').matches) setTimeout(() => panel.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 60)
                  }}
                  className={`press flex w-full items-start gap-3 px-4 py-3.5 text-left text-[1.0625rem] font-medium leading-snug transition-colors duration-200 ${
                    i === at ? 'bg-ink text-paper' : 'bg-white text-ink hover:bg-slate-100'
                  }`}
                  data-testid={`question-${item.id}`}
                >
                  <span className={`font-serif text-xl leading-none ${i === at ? 'text-field-gold' : 'text-indigo-600'}`}>Q.</span>
                  {item.q}
                </button>
              </li>
            ))}
          </ol>

          <section key={id} ref={panel} role="tabpanel" aria-label={q} className="animate-sheet scroll-mt-3 bg-white p-5 sm:p-8" data-testid="answer-panel">
            <p className="flex gap-3 text-[clamp(1.5rem,1.2rem+1.1vw,2.125rem)] font-semibold leading-[1.1] tracking-[-0.03em]">
              <span className="font-serif font-normal text-indigo-600">Q.</span>
              {q}
            </p>
            <p className="reading mt-4 flex gap-3 text-slate-800">
              <span className="text-[1.5rem] leading-[1.25] text-indigo-600">A.</span>
              <span>{a}</span>
            </p>
            <div className="mt-6">
              <Body r={r} />
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
