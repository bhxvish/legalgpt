import { useEffect, useState } from 'react'
import { explainApi } from '../api.js'
import { stagger } from './motion.jsx'

// The Explainability page: how LegalGPT opens the black box, and how faithful each explanation is.
// Per answer, the "Understand this answer" panel (Chat) and the bias check (Case analysis) do the work;
// this page shows what each explanation is and the measured results behind them
// (docs/reports/explainability_results.json, built by backend/scripts/make_explain_report.py).

const pct = (x) => (x == null ? '—' : `${Math.round(x * 100)}%`)

function Card({ title, children, i = 0, id }) {
  return (
    <section id={id} className="animate-rise scroll-mt-4 pt-2" style={stagger(i, 60)}>
      <h3 className="mb-3 text-[1.625rem] leading-tight text-slate-900">{title}</h3>
      <div className="space-y-3 text-sm leading-relaxed text-slate-700">{children}</div>
    </section>
  )
}

function Stat({ value, label, tone = 'text-slate-900' }) {
  return (
    <div className="border-t border-rule pt-2.5">
      <p className={`font-serif text-[2.25rem] leading-none tracking-[-0.02em] [font-variant-numeric:lining-nums_tabular-nums] ${tone}`}>{value}</p>
      <p className="mt-1.5 text-xs leading-snug text-slate-500">{label}</p>
    </div>
  )
}

const TOOLS = [
  ['Reasoning', 'Chat', 'The model sets out its reasoning in the IRAC form lawyers use: the issue, each section and why it applies or not, each fact matched to each element, and the conclusion with the facts it depends on.'],
  ['Faithfulness test', 'Chat', 'A model can describe reasoning it did not use. Each fact is removed in turn and the answer written again: a fact the model really relied on changes the conclusion. The agreement is shown as a score.'],
  ['Key words', 'Chat', 'Each word of the question is removed and the question matched against the law again (no model call): a heat-map shows which words tied the question to each section and to "this is a crime".'],
  ['Source influence', 'Chat', 'Each section is left out and the answer written again: a section is decisive if the conclusion changes without it.'],
  ['Bias check', 'Chat and Case analysis', 'The same question or case file with only the gender, religion, caste, income level, region or age changed. Differences are explained: required by the law itself (dowry death, rape, kidnapping a minor), normal variation, or possible bias. Runs automatically for gender- and caste-sensitive cases.'],
  ['Inside the local model', 'This page', 'For the fine-tuned model we can open the weights: what LoRA changed, layer by layer, and which sources the answer depends on, measured inside the model.'],
]

export default function ExplainabilityView() {
  const [r, setR] = useState(null)
  const [err, setErr] = useState(null)
  useEffect(() => {
    explainApi.results().then(setR).catch((e) => setErr(e.message))
  }, [])

  const models = r?.bias_models ? Object.entries(r.bias_models) : []
  const attrModels = r?.bias_attributes ? Object.entries(r.bias_attributes) : []
  const sys = (k) => (k === 'closed' ? 'Model alone' : 'LegalGPT')
  return (
    <div className="relative isolate min-h-0 flex-1 overflow-y-auto px-4 py-8 sm:py-10">
      <div className="mx-auto max-w-4xl space-y-6">
        <header className="border-b border-rule pb-6">
          <p className="kicker mb-2 animate-rise">
            Explainable AI
          </p>
          <h2 className="page-title">Opening the black box</h2>
          <p className="standfirst mt-4">
            A language model is a black box: citations show where an answer came from, not why the model concluded it. LegalGPT
            explains the decision itself. For hosted models, whose insides we cannot see, it changes the input and watches the
            output (perturbation). For our own fine-tuned model, it looks inside the weights. Every explanation is itself tested,
            because an explanation can be wrong too.
          </p>
        </header>

        <Card title="Explain any answer" i={1}>
          <p>Open <b>Understand this answer</b> under any legal answer in Chat, or the <b>Bias check</b> section of a case analysis.</p>
          <ul className="grid gap-x-8 sm:grid-cols-2">
            {TOOLS.map(([name, where, what]) => (
              <li key={name} className="border-t border-slate-200 py-3">
                <p className="flex flex-wrap items-baseline justify-between gap-2">
                  <b className="font-serif text-[1.15rem] font-medium text-slate-900">{name}</b>
                  <span className="text-[11px] text-slate-500">{where}</span>
                </p>
                <p className="mt-1 text-[13px] text-slate-600">{what}</p>
              </li>
            ))}
          </ul>
        </Card>

        {err && <p className="rounded-xl bg-amber-50 px-4 py-3 text-sm text-amber-900 ring-1 ring-amber-200">Results could not be loaded: {err}</p>}

        {models.length > 0 && (
          <Card title="Is the model biased? Gender swapped in 40 scenarios" i={2} id="bias">
            <p>
              24 gender-sensitive scenarios (dowry, cruelty, sexual offences, stalking) and 16 where gender should not matter (theft,
              cheating, hurt). Each is run with the genders swapped and again unchanged (control), by the model alone and inside
              LegalGPT ({models.map(([m]) => m).join('; ')}).
            </p>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[34rem] text-left text-[13px]">
                <thead className="text-slate-500">
                  <tr>
                    <th className="py-1.5 pr-3 font-medium">Model</th>
                    <th className="py-1.5 pr-3 font-medium" />
                    <th className="py-1.5 pr-3 font-medium">Scenarios</th>
                    <th className="py-1.5 pr-3 font-medium">No difference</th>
                    <th className="py-1.5 pr-3 font-medium">Required by law</th>
                    <th className="py-1.5 pr-3 font-medium">Run-to-run variation</th>
                    <th className="py-1.5 font-medium">Possible bias</th>
                  </tr>
                </thead>
                <tbody>
                  {models.flatMap(([m, bias]) => Object.entries(bias).flatMap(([s, d]) =>
                    Object.entries(d.by_group).map(([g, v]) => (
                      <tr key={m + s + g} className="border-t border-slate-100">
                        <td className="py-1.5 pr-3 text-slate-500">{m}</td>
                        <td className="py-1.5 pr-3 font-medium text-slate-800">{sys(s)}</td>
                        <td className="py-1.5 pr-3">{g === 'sensitive' ? 'gender-sensitive' : 'gender-neutral'}</td>
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
            <div className="grid gap-3 sm:grid-cols-2">
              {Object.entries(models[0][1]).map(([s, d]) => (
                <Stat
                  key={s}
                  value={d.gender_specific_in_both}
                  label={`${sys(s)} (${models[0][0]}): scenarios where a section the law defines by gender (e.g. dowry death of a woman) was still relied on after the genders were reversed, so likely misapplied`}
                  tone={d.gender_specific_in_both ? 'text-amber-700' : 'text-emerald-700'}
                />
              ))}
            </div>
            <p className="text-[13px] text-slate-500">
              A difference in a section the law defines by gender is what the law requires, not bias; when such a section stops
              applying, the general offences that take its place (hurt, culpable homicide) follow from it too. Sections an answer
              names only to rule them out are not counted.
            </p>
            {attrModels.length > 0 && (
              <div id="bias-attributes" className="space-y-2 border-t border-slate-100 pt-4">
                <h4 className="font-semibold text-slate-900">Caste, income level, region and age</h4>
                <p>
                  10 scenarios each where none of these should matter (theft, cheating, hurt). The two versions are identical except
                  for one sentence about the accused, for example from an upper-caste or a Dalit family. Both run twice, and a section
                  counts as possible bias only if it holds in both runs of one version and neither run of the other.
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[30rem] text-left text-[13px]">
                    <thead className="text-slate-500">
                      <tr>
                        <th className="py-1.5 pr-3 font-medium">Model</th>
                        <th className="py-1.5 pr-3 font-medium">Changed</th>
                        <th className="py-1.5 pr-3 font-medium">Possible bias: model alone</th>
                        <th className="py-1.5 font-medium">Possible bias: LegalGPT</th>
                      </tr>
                    </thead>
                    <tbody>
                      {attrModels.flatMap(([m, attrs]) => Object.entries(attrs).map(([a, d]) => (
                        <tr key={m + a} className="border-t border-slate-100">
                          <td className="py-1.5 pr-3 text-slate-500">{m}</td>
                          <td className="py-1.5 pr-3 font-medium text-slate-800">{a === 'income' ? 'income level' : a}</td>
                          {['closed', 'legalgpt'].map((s) => (
                            <td key={s} className={`py-1.5 pr-3 tabular-nums ${d[s].possible_bias ? 'font-semibold text-red-700' : ''}`}>
                              {d[s].possible_bias} of {d[s].pairs}
                            </td>
                          ))}
                        </tr>
                      )))}
                    </tbody>
                  </table>
                </div>
                <ul className="space-y-1 text-[13px] text-slate-500">
                  {Object.entries(r.bias_attribute_totals || {}).map(([m, t]) => (
                    <li key={m}>
                      {m}, all four together: {t.closed} of {t.pairs} for the model alone, {t.legalgpt} of {t.pairs} inside LegalGPT
                      (not a significant difference, p = {t.mcnemar_p}). {t.stereotypes} stereotyped sentences, {t.refusal_changes} changed refusals.
                    </li>
                  ))}
                </ul>
                <p className="text-[13px] text-slate-500">
                  LegalGPT is not consistently better here. An added sentence about the accused can change which sections the search
                  finds; that shows where it enters, not why the result changed. Some differences are the definition and the punishment
                  of one offence. With 10 scenarios per attribute, these are signals to investigate, not proof either way.
                </p>
              </div>
            )}
          </Card>
        )}

        {r?.reasoning && (
          <Card title="Is the stated reasoning faithful?" i={3}>
            <p>
              On {r.reasoning.questions} held-out scenarios the model wrote its reasoning trace, and each claim was tested by
              removing the fact or section it names.
            </p>
            <div className="grid gap-3 sm:grid-cols-3">
              <Stat value={`${r.reasoning.facts.key_facts_confirmed}/${r.reasoning.facts.key_facts}`} label="key facts the reasoning named that really changed the conclusion when removed" />
              <Stat value={pct(r.reasoning.facts.agreement_mean)} label="agreement: a fact is named exactly when it matters (faithfulness)" />
              <Stat value={pct(r.reasoning.sections.precision)} label="sections the reasoning says apply that are decisive when left out" />
            </div>
            <p className="text-[13px] text-slate-500">
              {r.reasoning.facts.unnamed > 0 && `Unnamed facts that still changed the conclusion: ${r.reasoning.facts.unnamed_influential} of ${r.reasoning.facts.unnamed}. `}Key facts quoted word for word from the
              question: {pct(r.reasoning.trace.key_facts_verbatim)}. A control run (the unchanged input asked again) separates real influence from run-to-run variation.
            </p>
          </Card>
        )}

        {r?.word_importance && (
          <Card title="Do the key-word explanations reflect what the system uses?" i={4}>
            <p>
              Deletion test on {r.word_importance.questions} held-out scenarios: removing the words the explanation ranks highest should
              hurt the match with the law far more than removing random words.
            </p>
            <div className="grid gap-3 sm:grid-cols-3">
              <Stat value={r.word_importance.by_k['1'].match_drop_top.toFixed(3)} label="match lost when the top word is removed" tone="text-indigo-700" />
              <Stat value={r.word_importance.by_k['1'].match_drop_random.toFixed(3)} label="match lost when a random word is removed" />
              <Stat value={pct(r.word_importance.by_k['1'].match_top_beats_random)} label="questions where the top word beats random words" tone="text-emerald-700" />
            </div>
          </Card>
        )}

        {r?.lora && (
          <Card title="Inside the fine-tuned model: what LoRA changed" i={5} id="lora">
            <p>
              Following Section 7 of the LoRA paper (Hu et al., 2022), the adapter's weight update dW = BA is analysed directly, and
              the tuned model is compared with the base model on its own answers.
            </p>
            {r.lora.weights && (
              <>
                <img src="/explain/fig6_lora_layers.png" alt="Heat map of the relative weight change per layer and module" className="w-full rounded-lg ring-1 ring-slate-200" />
                <div className="grid gap-3 sm:grid-cols-3">
                  <Stat value={`${r.lora.weights.mean_eff_rank}/${r.lora.weights.r}`} label="effective rank of the update: the adapter uses most of its capacity" />
                  <Stat value={`${(r.lora.weights.by_module.gate_proj.rel * 100).toFixed(2)}%`} label="largest relative change (MLP gate projection); attention layers change least" />
                  <Stat value={r.lora.weights.mean_amplification} label="amplification factor: how strongly dW boosts directions the base model already had" />
                </div>
              </>
            )}
            {r.lora.behaviour && (
              <>
                <img src="/explain/fig7_lora_tokens.png" alt="What the adapter made more likely" className="w-full rounded-lg ring-1 ring-slate-200" />
                <div className="grid gap-3 sm:grid-cols-2">
                  <Stat
                    value={`${r.lora.behaviour.influence.top_source_is_cited}/${r.lora.behaviour.influence.answers_with_citations}`}
                    label="answers whose most influential source (measured inside the model) is one it cites: the citations reflect what the model used"
                    tone="text-emerald-700"
                  />
                  <Stat value={`${r.lora.behaviour.influence.mean_drop_cited} vs ${r.lora.behaviour.influence.mean_drop_uncited}`} label="drop in the answer's log-probability when a cited vs an uncited source is removed" />
                </div>
                <p className="text-[13px] text-slate-500">
                  Tokens the adapter raised most: {r.lora.behaviour.top_tokens_raised.slice(0, 8).map((t) => `“${t.token}”`).join(', ')}.
                </p>
              </>
            )}
          </Card>
        )}

        {r?.classifier && (
          <Card title="InLegalBERT's explanations (Label tab)" i={6}>
            <p>
              Every sentence-role suggestion has a "Why this suggestion?" word explanation (word removal, or Integrated Gradients). On
              100 held-out sentences, removing the top 10% of words it ranks highest lowers the predicted role's probability{' '}
              <b>{r.classifier.comprehensiveness_at_10.occlusion}</b> (word removal) and {r.classifier.comprehensiveness_at_10.integrated_gradients} (IG), against{' '}
              {r.classifier.comprehensiveness_at_10.random} for random words.
            </p>
          </Card>
        )}
      </div>
    </div>
  )
}
