# Analysis for the paper: significance, ablation, errors, model size, IPC vs BNS

**Scoring corrections (2026-10-15)**, applied to every table here and in `benchmark_results.md`:
1. **Malformed questions.** ss.143 and 311 are titled just "Punishment"; their questions are flagged
   invalid and not scored. Statute n is now 59 and punishment n is 39.
2. **"Wrong punishment".** It now means stating a term the section never mentions. In graded sections
   another tier is a fair answer: s.213 gives up to seven years if the screened offence is capital,
   three if it carries life. The answer key's own term is still required for "punishment correct".
   This moved closed-book wrong-punishment rates down by 1-3 points, and the 3B's from 3% to 0%; every
   significance conclusion is unchanged.
3. **State amendments.** Sections inserted by state amendments count as real sections, not invented
   ones.

All numbers come from the benchmark runs in `data/reports/bench/baselines/` (seed 7: 40 punishment and
20 definition questions, 60 held-out scenarios, 30 out-of-scope questions). Scenario and out-of-scope
labels are **silver** (AI-written) and still need legal review. Scripts are named in each section.

## 0. Headline: the final system (runs `*_final`, 2026-10-15)

These are the numbers for the paper. The final system has every retrieval fix: the lexicon, the
query rewrite, the retrained scope check and the definition lookup. Each hosted model is compared
with itself answering alone, on the same questions (exact McNemar tests).

| model | measure | n | model alone | LegalGPT | only alone right | only LegalGPT right | p |
|---|---|---|---|---|---|---|---|
| gpt-5.4-mini | right section (statute) | 59 | 75% | **100%** | 0 | 15 | <0.001 |
| | acceptable section (scenario) | 60 | 83% | **93%** | 4 | 10 | 0.18 |
| | wrong punishment ↓ | 39 | 15% | **0%** | 6 | 0 | 0.031 |
| | civil question answered as an offence ↓ | 30 | 47% | **0%** | 14 | 0 | <0.001 |
| Gemini 3.5 flash-lite | right section (statute) | 59 | 81% | **98%** | 0 | 10 | 0.002 |
| | acceptable section (scenario) | 60 | 78% | **93%** | 3 | 12 | **0.035** |
| | wrong punishment ↓ | 39 | 15% | **0%** | 6 | 0 | 0.031 |
| | civil question answered as an offence ↓ | 30 | 47% | **0%** | 14 | 0 | <0.001 |
| gpt-oss-120b | right section (statute) | 59 | 41% | **100%** | 0 | 35 | <0.001 |
| | acceptable section (scenario) | 60 | 85% | 82% | 8 | 6 | 0.79 |
| | wrong punishment ↓ | 39 | 54% | **0%** | 21 | 0 | <0.001 |
| | civil question answered as an offence ↓ | 30 | 90% | **0%** | 27 | 0 | <0.001 |

For "only alone right" / "only LegalGPT right" on the ↓ rows, read "only that system made the
error".

**Claims the tests support:**
- With all three models, LegalGPT states **no wrong punishments** and treats **no civil question as
  an offence**; each model alone does both often (p ≤ 0.031).
- On statute questions LegalGPT is **significantly more accurate** for all three models (98-100% vs
  41-81%, p ≤ 0.002).
- On everyday scenarios LegalGPT is **at least at parity**:
  - significantly better with Gemini (93% vs 78%);
  - ahead but not significantly with gpt-5.4-mini (93% vs 83%);
  - level with gpt-oss-120b (82% vs 85%).
- Scenario accuracy: LegalGPT's first version scored 43-45%; the final system scores 82-93%.

**Final-system details:**
- Scenario answers citing only wrong sections: 2% / 2% / 13% (gpt-5.4-mini / Gemini / gpt-oss).
- Scenarios refused: 5% for each model.
- Cost of the gpt-5.4-mini run: $0.29 for 150 answers.
- Answers came from Groq and Gemini free tiers. The last 106 gpt-oss answers and the last 34
  Gemini answers used a second account's key: the same model and host.

Sections 1-2 below were measured on earlier versions (`*_rw`: before the retrained scope check and
the definition lookup). They are kept because they show what each step changed.

## 1. Paired significance tests (`backend/scripts/paired_tests.py`)

Every system answers the same questions, so each comparison is paired. Only the questions on which
the two systems disagree carry information: *only A* and *only B* count them. The p-value is the
exact McNemar test (two-sided binomial test on the disagreements).

| A (closed-book) | B (inside LegalGPT) | measure | n | A | B | only A | only B | p |
|---|---|---|---|---|---|---|---|---|
| gpt-5.4-mini | gpt-5.4-mini (with query rewrite) | wrong punishment ↓ | 39 | 15% | 0% | 6 | 0 | 0.031 |
| | | right section (statute) | 60 | 73% | 82% | 10 | 15 | 0.42 |
| | | acceptable section (scenario) | 60 | 83% | 88% | 6 | 9 | 0.61 |
| | | civil question answered as an offence ↓ | 30 | 47% | 0% | 14 | 0 | <0.001 |
| gemini-3.5-flash-lite | gemini-3.5-flash-lite (with the rewrite) | wrong punishment ↓ | 39 | 15% | 0% | 6 | 0 | 0.031 |
| | | right section (statute) | 60 | 80% | 82% | 9 | 10 | 1.00 |
| | | acceptable section (scenario) | 60 | 78% | 43% | 25 | 4 | <0.001 |
| | | civil question answered as an offence ↓ | 30 | 47% | 0% | 14 | 0 | <0.001 |
| gpt-oss-120b | gpt-oss-120b (with the rewrite) | wrong punishment ↓ | 39 | 54% | 0% | 21 | 0 | <0.001 |
| | | right section (statute) | 60 | 40% | 82% | 5 | 30 | <0.001 |
| | | acceptable section (scenario) | 60 | 85% | 45% | 26 | 2 | <0.001 |
| | | civil question answered as an offence ↓ | 30 | 90% | 0% | 27 | 0 | <0.001 |

Within LegalGPT, before vs after the scenario fix:

| system | measure | n | before | after | only before | only after | p |
|---|---|---|---|---|---|---|---|
| LegalGPT + gpt-5.4-mini | acceptable section (scenario) | 60 | 45% | 88% | 0 | 26 | <0.001 |
| LegalGPT + gpt-oss-120b | acceptable section (scenario) | 60 | 45% | 82% | 0 | 22 | <0.001 |
| LegalGPT + gemini-3.5-flash-lite | acceptable section (scenario) | 60 | 43% | 90% | 0 | 28 | <0.001 |
| LegalGPT-LoRA v0.4 (new prompt and lexicon only) | acceptable section (scenario) | 60 | 8% | 13% | 0 | 3 | 0.25 |
| LoRA v0.4 → v0.5 (retrained with scenario examples) | acceptable section (scenario) | 60 | 13% | 43% | 0 | 18 | <0.001 |

**What the tests support.**
- **Safety gains are significant for all three models:**
  - no wrong punishments (p ≤ 0.031);
  - no civil questions treated as offences (p < 0.001).
- **Accuracy gains:**
  - LegalGPT's statute-section accuracy is significantly higher for gpt-oss-120b (40% → 82%).
  - For gpt-5.4-mini and Gemini it is not significantly different from closed-book. Report it as
    "comparable".
- **Scenarios:**
  - After the fix, LegalGPT on gpt-5.4-mini (88%) is **not significantly different** from the
    model alone (83%, p = 0.61). The claim is parity on scenarios together with the safety gains,
    not superiority.
  - The fix itself (45% → 88%) is highly significant: 26 questions improved, none got worse.
- **gpt-oss-120b with the fix:**
  - LegalGPT 82% vs the model alone 85% on scenarios (p = 0.77, parity).
  - Right section on statute questions: 82% vs 40% (p < 0.001).
  - Wrong punishment: 0% vs 54% (p < 0.001).
  - Civil questions answered as offences: 0% vs 90% (p < 0.001).
  - Its last 84 answers came through OpenRouter, pinned to Groq as host, with the same settings.
- **gemini-3.5-flash-lite with the fix:**
  - LegalGPT 90% vs the model alone 78% on scenarios (p = 0.12, not significant).
  - Only-wrong-section scenario answers: 0% vs 15%.
  - Wrong punishment: 0% vs 15% (p = 0.031).
  - Civil questions answered as offences: 0% vs 47% (p < 0.001).
- **Across all three models**, the fix raised LegalGPT's scenario accuracy from 43–45% to 82–90%
  (each p < 0.001; 76 questions improved, none got worse). Each model is now at parity with itself
  answering alone, while keeping 0% wrong punishments and 0% civil questions treated as offences.

## 2. Retrieval ablation (`backend/scripts/ablation_retrieval.py`)

Each row adds one component and is scored on the same held-out questions: whether an acceptable
section is among the sources for the 60 scenarios, and whether sources are returned for the 30
out-of-scope questions.

| retrieval | scenarios: acceptable section | wrong only | refused | out-of-scope answered |
|---|---|---|---|---|
| embedding only (MiniLM, floor 0.55) | 2% | 2% | 97% | 0% |
| + lay-term lexicon (original 20 rules) | 27% | 10% | 63% | 0% |
| + scope check (floor 0.40 for crime questions) | 45% | 17% | 38% | 0% |
| + lexicon expanded to 41 rules | 50% | 17% | 33% | 0% |
| + LLM query rewrite (gate 0.50), Gemini's rewrites | 90% | 0% | 10% | 0% |
| + LLM query rewrite (gate 0.50), gpt-oss-120b's rewrites | 83% | 7% | 10% | 0% |
| + retrained scope check (gate 0.58), Gemini's rewrites | **95%** | 0% | 5% | 0% |
| + retrained scope check (gate 0.58), gpt-oss-120b's rewrites | **88%** | 7% | 5% | 0% |

The rewrite rows replay retrieval with the rewrites each model actually made in its benchmark run
(saved with every answer), so no new LLM calls were needed. The retrained scope check
(`train_scope.py`) uses:
- 486 examples instead of 100;
- the practice scenarios, never the held-out ones;
- 138 AI-written civil and general questions.

Its threshold is set 0.05 above the highest out-of-scope validation question.

End-to-end, the answer stage adds the prompt rule ("applying a defined offence to the facts is
answering from the sources"):

| configuration (LegalGPT, 60 held-out scenarios) | acceptable section cited |
|---|---|
| gpt-5.4-mini, original pipeline | 45% |
| gpt-5.4-mini, + expanded lexicon + rewrite + prompt rule | **88%** |
| gpt-oss-120b, original pipeline | 45% |
| gpt-oss-120b, + expanded lexicon + rewrite + prompt rule | **82%** |
| gemini-3.5-flash-lite, original pipeline | 43% |
| gemini-3.5-flash-lite, + expanded lexicon + rewrite + prompt rule | **90%** |
| LoRA v0.4, original pipeline | 8% |
| LoRA v0.4, + expanded lexicon + prompt rule (offline, no rewrite) | 13% |
| LoRA v0.5 (retrained with 355 scenario examples, no refusals), offline | 43%, but 22% cite only wrong sections |
| LoRA v0.6 (as v0.5, plus 51 scenario refusals = 13%; only 51 misses were available), offline | 43%, still 22% only wrong sections |

**LoRA v0.5.**
- Offline retrieval gives an acceptable section for only 30 of the 60 held-out scenarios. v0.4 refused 19 of those 30.
- Retraining with scenario examples answers most of them: 13% → 43% acceptable (p < 0.001).
- Statute accuracy (78%) and out-of-scope refusals (100%) are unchanged.
- The one "wrong punishment" is the malformed s.311 question (section 3).
- **The cost:** "only wrong sections" rose from 5% to 22%. In 10 of those 13 answers, retrieval had not supplied an acceptable section, and v0.5 applied the nearest retrieved offence anyway, since its scenario examples never showed a refusal. Some fits are defensible (an axe attack → s.324), others are not (chain snatching → s.279, rash driving).
- v0.6 added refusal examples to test whether this trade-off can be avoided.

**LoRA v0.6.**
- Only 51 refusal examples were available (13% of the scenario examples, not the 25% intended).
- They did not change the trade-off: 43% acceptable and 22% only wrong sections, the same as v0.5
  (paired test v0.5 vs v0.6: 1 question better, 1 worse).
- Statute accuracy was 80% (v0.5: 78%), and answers were faster (2.5 s vs 3.7 s).
- Reducing force-fitting probably needs many more refusal examples, or human-written answers that
  explain why a retrieved offence does not fit.
- **The app stays on v0.4** (`ADAPTER_PATH` in `.env`):
  - the published ZK manifest fingerprints v0.4;
  - v0.4 rarely cites a wrong section (5%), though it refuses most scenarios.
- Switching to v0.6 means re-running `zk_setup.py` to publish its fingerprint.

**Reading.**
- The original pipeline was limited by **retrieval**: embedding search alone almost never reaches
  the statute from everyday wording (2%).
- The lexicon and the scope check lift retrieval to 45%. The expanded lexicon adds 5 points on
  held-out questions (it added 20 on the questions it was written from, so most of that gain was
  specific to those questions).
- The LLM rewrite and the prompt rule account for the rest of the end-to-end gain (50% → 88%).
- Out-of-scope questions stay at 0% answered in every configuration. The gate is set so that no
  tuning-set out-of-scope question reaches it (highest scope probability 0.49 < 0.5), whatever the
  LLM says.

### Fixes made after this analysis (retrieval only; end-to-end runs not repeated yet)

**The six scenario refusals at the gate:** the retrained scope check (above) recovers retrieval for
3 of the 6 with Gemini's rewrites. Held-out refusals fell from 10% to 5%, with 0 of 30 out-of-scope
questions answered.

**Definition questions:**
- The fix: a question asking what a term is ("What is robbery?", "How does the IPC define X?") also
  fetches the section **titled** with that term.
- Designed on the 32 practice definition questions: 47% -> 97% retrieved.
- Held-out 20 (checked once): the defined section is among the sources 50% -> **100%**, and ranked
  first 10% -> **100%**.
- Punishment questions are unchanged (97%).

**Malformed items:**
- ss.143 and 311 are titled just "Punishment", so their questions read "... prescribe for
  punishment?". They are flagged `"invalid"` in `statute_qa.jsonl` and not scored, so statute n is
  59, not 60.
- They stay in the file, so every seeded sample is unchanged.

The end-to-end numbers in sections 1-2 predate these two retrieval fixes. A re-run would show their
effect on the answers.

## 3. Error analysis (LegalGPT + gpt-5.4-mini with the rewrite)

### Scenarios: 7 of 60 not answered with an acceptable section

| cause | count | examples |
|---|---|---|
| Safety gate: the scope check put a clear crime just below 0.5, so no rewrite and a refusal | 6 | goldsmith keeps jewellery given for polishing (0.47); ex tracks location after being blocked (0.47); box of bricks instead of a paid-for phone (0.40); children take and sell mangoes (0.47); husband beats wife daily (0.49); delivery agent swaps a phone (0.43) |
| Probably a label error: the answer cites the more specific offence | 1 | rigged weighing scales: LegalGPT cited s.264 (fraudulent use of a false weighing instrument); the AI-written label expected cheating (ss.415–419) |

The gate trades these refusals for a guarantee on civil questions. The scope classifier was
trained on only 100 questions. Retraining it on the practice scenarios (not the held-out ones) is
the most direct next fix.

### Statute questions: 11 of 60 without the right section

| cause | count | examples |
|---|---|---|
| Definition question answered with the punishment or aggravated sections, because retrieval ranks them above the definition | 8 | "What is robbery?" → s.392/393, not s.390; extortion → s.384, not s.383; hurt → s.323, not s.319; grievous hurt → s.325, not s.322; lurking house-trespass → s.453–458, not s.443 |
| General definitions in Chapter II not retrieved | 2 | "voluntarily" (s.39), "counterfeit" (s.28) |
| Malformed benchmark question | 1 | "What penalty does the IPC prescribe for punishment?", generated from s.311 (titled "Punishment"). Remove it from the benchmark. |

The definition/punishment confusion is systematic, and the fix is cheap: a title-match boost for
"what is X" questions. It must be designed on the 305 statute questions outside the benchmark
sample, not on these.

## 4. Limitation: IPC and the Bharatiya Nyaya Sanhita (draft paragraph)

> **Scope of the legal corpus.** LegalGPT retrieves from the Indian Penal Code, 1860. From 1 July
> 2024 the IPC was replaced by the Bharatiya Nyaya Sanhita, 2023 (BNS), so offences committed on or
> after that date are charged under the BNS. The IPC remains the governing law for offences
> committed before it, and therefore for a large share of pending trials, appeals and reported
> judgments for years to come; it is also the vocabulary of the existing case law. Our benchmark,
> retrieval corpus and Prolog rules are IPC-only. The BNS largely re-enacts IPC offences under new
> section numbers (for example, murder moves from IPC s.302 to BNS s.103), so the method transfers
> to the BNS by indexing its text and adding an IPC↔BNS concordance. We did not do this here. The
> benchmark flags one failure mode instead: an answer that gives an offence's BNS number as its IPC
> section ("BNS confusion", 0% for LegalGPT). The concordance used for that check
> (`benchmark/ipc_bns_map.json`) was compiled without verification against the BNS text, so it
> serves only as a diagnostic.

**Update (2026-10-18): official concordance added.** The IPC→BNS correspondence is now taken from
the official table, *Correspondence Table and Comparison Summary of the BNS to the IPC* (Bureau of
Police Research and Development, Ministry of Home Affairs). `backend/scripts/build_bns_map.py`
rebuilds its rows from the PDF's text positions and keeps only section numbers and subject titles:
- 527 IPC sections are mapped; 23 BNS provisions have no IPC source (e.g. organised crime s.111,
  snatching s.304). Sedition (124A), s.377 and adultery (497) have no entry.
- All 47 entries of the benchmark's unverified concordance agree with the official table, so the
  "BNS confusion" diagnostic above stands.
- The app shows the BNS number next to every IPC source, under answers ("s.302 IPC → s.103 BNS"),
  and next to each offence in Case analysis. The model's answers are not changed: the numbers come
  from the table, never from the model, so the benchmark is unaffected.

Remaining future work: index the BNS text itself and answer under the BNS when the offence date is
on or after 1 July 2024.

## Caveats

- Scenario and out-of-scope labels are AI-written (silver). The rigged-scales case above shows
  that at least one of them is probably wrong. A legal review of the 60 + 30 held-out items
  should come before submission.
- The LegalGPT gpt-oss-120b answers were partly re-cited: 69 answers used `【n】` citation markers,
  which were mapped back to sources with the app's own canonicalization (see
  `benchmark_results.md`).
- "Invented sections" (a cited number not in the IPC) originally counted sections inserted by state
  amendments (Gujarat's s.379A/379B on snatching, Chhattisgarh's s.509B, Himachal Pradesh's s.304-AA) as invented. LegalGPT cited
  them from the State Amendments text it was given. With those sections recognised
  (`metrics.known_sections`), every LegalGPT configuration has 0% invented sections.

## 5. Local models by size (same training data, fully offline)

1.5B, 3B and 7B (Qwen2.5-Instruct) were LoRA-trained on **identical examples and splits**. The 1.5B
was trained with `--examples-from` the 3B's `examples.jsonl`, and the training data has no judgment
texts. All three were evaluated with the same retrieval code; each model writes its own query rewrite,
with no hosted model. Figure: `docs/figures/fig4_size.png`.

| | 1.5B | 3B | 7B |
|---|---|---|---|
| statute: right section (59) | 90% | **100%** | 98% |
| punishment correct (39) | 87% | 92% | **97%** |
| wrong punishment (39) | 8% | **0%** | **0%** |
| scenario: acceptable section (60) | 50% | 60% | **65%** |
| scenario: only wrong sections | 37% | 32% | **23%** |
| civil question answered as an offence (30) | 0% | 0% | 0% |
| own rewrite named an acceptable section | 28/60 | 34/60 | 35/60 |
| where it was run | Colab L4, 4-bit | laptop RTX 3050 Ti, 4-bit | Colab L4, llama.cpp Q4_K_M (the app's GGUF files) |

**Paired tests:**
- 1.5B -> 3B, statute: p = 0.031.
- 1.5B -> 7B, scenarios: p = 0.049.
- 3B -> 7B, scenarios: p = 0.65 (not significant).

**Reading.** Accuracy rises with size, and safety on civil questions holds at every size. Between 3B
and 7B the gain is small, because both local models name an acceptable section in their own rewrite
for only 57-58% of scenarios. The same pipeline reaches 82-90% when a large hosted model writes the
rewrite. At this scale the bottleneck is the local model's legal knowledge, not the retrieval
pipeline.

**Deployment format.** The same trained 3B adapter can also run through llama.cpp: the official Q4_K_M
GGUF base model, with the adapter converted to GGUF. On the laptop's 4 GB GPU this takes 3.0 s per
question against 70.9 s for PyTorch 4-bit (bitsandbytes NF4), measured offline with the model writing
its own rewrite.

| 3B, offline | PyTorch 4-bit (NF4) | llama.cpp Q4_K_M |
|---|---|---|
| right section (statute) | 100% | 100% |
| wrong punishment | 0% | 0% |
| scenario: acceptable section | 60% | 45% (p = 0.064) |
| scenario: only wrong sections | 32% | 15% |
| scenario: refused | 8% | 40% |

The rewrites were about equally good (34 vs 32 of 60 named an acceptable section). The difference
is in the answer step: the llama.cpp build refused 11 scenarios whose acceptable section had been
retrieved (PyTorch: 1). The quantisation format alone shifts the trained model towards caution. The
app uses the llama.cpp build for speed, and the paper's size comparison uses the PyTorch numbers.

**Practical choice.** The 3B is the everyday offline model: as accurate as the 7B on statute
questions, no wrong punishments, and it fits a 4 GB laptop GPU. The 7B, through llama.cpp, is
available as "Local heavy". The earlier 1.5B v0.4 (old recipe: 80% statute, 13% scenarios) refused
most scenarios. The v0.7 recipe trades some of that caution for coverage.

## 6. Counterfactual test (complete)

147 paired fact patterns, qwen3.8-27b, `benchmark_results.md` section 2.

| | LLM alone | LLM + Prolog rules |
|---|---|---|
| accuracy | 92% | 95% |
| one-fact flip changes the verdict correctly | 95% | 100% |
| omitted element gives INSUFFICIENT | 77% | 85% |
| same verdict for a paraphrase | 100% | 95% |

- 8 cases are right only with the rules and 3 only without them: exact McNemar p = 0.23, a trend.
- The rules help most on what they encode: flips and omissions.
- 3 answers came from another host (Cerebras via OpenRouter); without them the result is identical.

## 7. Sentence roles in judgments: InLegalBERT ablation (Module 1)

The labelling assistant tags every sentence of a judgment as Facts, Law Applied, Precedent,
Argument or Ruling (InLegalBERT, `law-ai/InLegalBERT`, fine-tuned).

**Setup:**
- Corpus snapshot v0.3: 36 judgments, 4,823 human-checked sentences.
- 5-fold cross-validation grouped by judgment (no judgment is in training and test at once).
- Identical settings in every fine-tuned row: max length 256, all layers trained, 6 epochs, seed 13.
- Runs on a Colab L4: `train_rrl_classifier.py --save-predictions`, `rrl_sequence_model.py`.
- Tests: `rrl_ablation_report.py`. Every held-out sentence is predicted once, so rows are compared on the
  same sentences: exact McNemar test, plus a 95% CI for the macro-F1 difference from a bootstrap
  that resamples whole judgments.

| Model | Macro-F1 | Accuracy | Facts | Law Applied | Precedent | Argument | Ruling |
|---|---|---|---|---|---|---|---|
| Sentence only | 0.667 | 70.7% | 0.841 | 0.582 | 0.636 | 0.620 | 0.656 |
| + position only | 0.702 | 74.0% | 0.854 | 0.610 | 0.684 | 0.667 | 0.695 |
| + neighbours only | 0.729 | 75.9% | 0.865 | 0.628 | 0.729 | 0.749 | 0.672 |
| + position + neighbours (deployed) | 0.739 | 77.1% | 0.888 | 0.632 | 0.716 | 0.758 | 0.701 |
| BiLSTM sequence model | 0.720 | 76.6% | 0.869 | 0.556 | 0.754 | 0.727 | 0.697 |
| **Ensemble: deployed + sequence** | **0.782** | **80.7%** | **0.896** | **0.696** | **0.782** | **0.803** | **0.730** |

(Pooled over all held-out sentences. The fold means are similar; for example, the ensemble's is
0.782 ± 0.024 and the deployed model's 0.741 ± 0.032.)

| Comparison | Macro-F1 change (95% CI) | Fixed / broken sentences | McNemar p |
|---|---|---|---|
| + position vs sentence only | +0.035 (+0.013 to +0.056) | 473 / 312 | 1e-8 |
| neighbours only vs position only | +0.027 (+0.003 to +0.050) | 498 / 407 | 0.003 |
| both vs neighbours only | +0.010 (-0.014 to +0.036) | 359 / 301 | 0.026 |
| sequence model vs deployed | -0.019 (-0.076 to +0.030) | 481 / 507 | 0.43 |
| **ensemble vs deployed** | **+0.043 (+0.020 to +0.066)** | **329 / 154** | **1e-15** |
| ensemble vs sentence only | +0.115 (+0.080 to +0.149) | 773 / 288 | 9e-52 |

**What this shows:**
- **Context matters.**
  - Where a sentence sits in the judgment and the sentences around it each help on their own (+3.5 and +6.2 macro-F1 points over the sentence alone).
  - Together they give +7.2.
  - Adding position on top of the neighbours is a small extra step (+1.0; the per-sentence test is significant, but the judgment-level CI includes 0).
- **The whole judgment, in order, helps when combined.**
  - The BiLSTM reads every sentence in order but uses frozen InLegalBERT vectors. Alone it is no better than the fine-tuned model (p = 0.43), but it makes different mistakes.
  - Averaging the two models' probabilities gains +4.3 points (CI +2.0 to +6.6). It fixes 329 sentences for every 154 it breaks.
  - The biggest gains are on the weakest classes: Law Applied +6.4, Precedent +6.6, Argument +4.5.
- **No tuning on test data.**
  - The ensemble weight is a fixed 0.5.
  - The BiLSTM's early stopping uses 3 validation judgments from each fold's training cases.
  - Both models are trained independently on the same folds; there is no stacking on held-out predictions.

**Caveats:**
- 36 judgments is a small corpus; the judgment-level intervals are the honest measure of uncertainty.
- Deployment (§7b): the Review tab now runs the final ensemble, trained on all judgments (`train_rrl_final.py`).

Figure: `docs/figures/fig5_rrl.png`.

### 7b. Round 2: longer training and transfer from a public corpus

**Public corpus:** *InRhetoricalRoles* (Kalamkar et al., LREC 2022; CC BY-SA 4.0, used for
training only and not redistributed).
- 245 train and 30 dev judgments, 23,029 labelled sentences.
- Its 13 roles were mapped onto our 5 (`build_rrl_public.py`). PREAMBLE, ISSUE and NONE are read as context only.
- Public judgments that share sentences with ours are dropped first.
- Stage A (`pretrain_rrl_public.py`, 2 epochs) reaches macro-F1 0.687 / accuracy 80.8% on the public dev set under the mapping.

Our cross-validation uses the same 5 folds and the same 4,823 sentences as above (Colab L4).

| Single model | Macro-F1 | Accuracy |
|---|---|---|
| deployed (InLegalBERT, context; row 4 above) | 0.739 | 77.1% |
| longer training (10 epochs, 384 tokens, lr 2e-5) | 0.743 | 77.4% |
| public transfer (stage A, then our judgments) | 0.757 | 78.6% |
| public transfer + longer training | 0.745 | 77.5% |
| BiLSTM (round 1) | 0.720 | 76.6% |
| BiLSTM pretrained on the public judgments | 0.782 | 80.2% |
| BiLSTM pretrained on the public judgments + stage-A sentence vectors | 0.793 | 81.4% |

**Ensembles:** every fine-tuned model averaged 50/50 with every sequence model gives macro-F1 between
0.780 and 0.815. All six that combine a public-pretrained BiLSTM are 0.804 or above.

**Final system:** the public transfer averaged with the BiLSTM pretrained on the public judgments
plus stage-A vectors, i.e. both improvements together. I chose this combination before computing the
ensemble grid. Its cell (0.807) is not the grid's best (0.815): the cells differ by less than the
noise.
- **Overall:** macro-F1 **0.807**, accuracy 81.3%.
- **Per class:** Facts 0.888, Law Applied **0.750**, Precedent **0.832**, Argument 0.812, Ruling 0.755.

| Comparison | Macro-F1 change (95% CI, judgment bootstrap) | Fixed / broken | McNemar p |
|---|---|---|---|
| public transfer vs deployed (single models) | +0.018 (-0.004 to +0.040) | 343 / 269 | 0.003 |
| upgraded BiLSTM vs round-1 BiLSTM | +0.072 (+0.023 to +0.127) | 528 / 294 | 3e-16 |
| **final ensemble vs deployed** | **+0.068 (+0.041 to +0.096)** | **535 / 268** | **3e-21** |
| final ensemble vs round-1 ensemble | +0.026 (-0.006 to +0.056) | 352 / 260 | 2e-4 |
| final ensemble vs its best single member (upgraded BiLSTM) | +0.015 (+0.005 to +0.025) | 159 / 100 | 3e-4 |

**What this shows:**
- **The public corpus helps most through the sequence model.** Pretraining the BiLSTM on 245 public
  judgments lifts it from 0.720 to 0.782, and stage-A vectors to 0.793. That is now the best single
  model.
- **Fine-tuning gains are smaller.** For the fine-tuned classifier, transfer gives +1.8 points (the
  per-sentence test is significant; the judgment-level CI touches 0), and longer training gives
  almost nothing (+0.4).
- **Averaging still pays.** The two models still make different mistakes, so averaging them adds
  +1.5 points over the best single model, significant at both levels.
- **Overall:** the final system is +6.8 points over the deployed model and +14.0 points over the
  sentence-only baseline (0.667 → 0.807). The weakest classes gain most: Law Applied 0.632 → 0.750
  and Precedent 0.716 → 0.832.
- **Confidence still works for review.** The final ensemble's averaged confidence (not
  temperature-calibrated) still separates easy from hard sentences: ≥ 0.9 covers 40% of sentences
  at 96.8% accuracy; ≥ 0.95 covers 12% at 98.8%.

**Caveats:**
- The public labels follow a different guideline. In particular, ANALYSIS sentences stating a
  general principle would be Law Applied in ours.
- 36 of our judgments is still small; the judgment-level intervals are the honest uncertainty.
- More of our own labelled judgments (snapshot v0.4) is the next lever.

Script: `backend/scripts/rrl_round2_report.py`. Figure: `docs/figures/fig5_rrl.png` (rounds 1 and 2).

**Deployed (2026-10-17).** The final system was retrained on 32 of the 36 judgments (`train_rrl_final.py`, Colab L4).
The snapshot's 4 validation judgments (754 sentences) were held out for epoch choice, early stopping
and calibration. On them it scores macro-F1 0.875 (the fine-tuned member alone: 0.782). That set also
tuned the model, so these numbers are optimistic; the cross-validated 0.807 above is the estimate to
report.

The Review tab runs this ensemble with a review threshold of 0.9. On the validation judgments, 66% of
sentences are then accepted without review, and 97.0% of those are correct; 10% of them are still
spot-checked. On a judgment it had never seen (Meera v. State of Tamil Nadu, 67 sentences), the
upload took 7 s and 32 sentences were highlighted for checking.


**Label provenance (checked 2026-10-17).** Of the 4,823 "human-checked" sentences in snapshot v0.3,
529 (11%) were labelled or review-confirmed by an AI assistant (annotator `claude`), not a person.
They come from 4 Supreme Court judgments added through the review workflow (Rajo 225, Mukesh Singh
167, Prakash Nishad 107, State of U.P. 30). The other 4,294 are the team's manual annotations
(4,274) plus the author's own (20).

Sensitivity check on the saved cross-validation predictions, scoring only the 32 judgments labelled
by people (training still included the 4):
- deployed model: 0.752;
- final ensemble: **0.813**;
- gain: +0.061, McNemar p = 2e-14 (on all 36 judgments: +0.068).

The improvement does not depend on the AI-made labels. The paper should state the provenance. A
clean re-run without the 4 judgments, or after a person re-checks their labels in the Review tab,
would remove the caveat entirely.

## 8. Judgment library (citations in the chat and case analysis)

**Contents (2026-10-18):** 349 judgments, split into 28,055 role-labelled passages:
- 312 Supreme Court criminal judgments from 2020-2023 (CC-BY-4.0, `labofsahil/Indian-Supreme-Court-Judgments`):
  - only those citing the IPC were kept: 312 of 580 fetched;
  - their sentence roles were predicted by the deployed ensemble;
- the 37 annotated judgments, with their stored labels.

| Role | Passages |
|---|---|
| Precedent | 9,140 |
| Facts | 8,056 |
| Ruling | 5,393 |
| Law Applied | 3,098 |
| Argument | 2,368 |

**How it is used:**
- **Chat:** cites at most 2 passages (Ruling / Law Applied / Precedent, one per judgment) with
  cosine similarity ≥ 0.60, only when the IPC retrieval found sources.
- **Case analysis:** compares the file's facts with Facts passages to find similar cases, and cites
  their Ruling passage.

**Example:** the dowry-death question cites s.304B and s.498A, plus *Charan Singh v. State of
Uttarakhand* (SC, 2023) and *Satbir Singh v. State of Haryana* (SC, 2021). The answer quotes what the
court said about the ingredients of s.304B.

**Not yet measured:** how often a cited judgment is truly on point. A labelled set of
question-to-relevant-judgment pairs would be needed; future work.

## 9. New evaluations (2026-10-18; drafts, numbers from data/reports/)

**Hindi questions** (`eval_hindi.py`; the 60 held-out scenarios and 30 out-of-scope questions,
machine-translated to Hindi by gemini-3.5-flash-lite, not checked by a native speaker; system model
gpt-oss-120b). Right section retrieved: English 53/60; Hindi as written 0/60; Hindi with the old
rewrite 1/60; **Hindi translated first 52/60**. Out-of-scope questions answered: 0/30 in every
condition. End-to-end (the chat translating, retrieving and answering by itself, gpt-oss-120b):
of 60 Hindi scenarios 56 answered, all 56 written in Hindi, **51 cite an acceptable section (85%)**,
4 declined; no citation number outside the sources; all 30 out-of-scope questions declined, in Hindi.
For comparison, the English benchmark run of the same model cites an acceptable section in 82%
(§0); the scoring differs slightly (here: any cited [n] is an acceptable section).

**Prompt injection in Case analysis** (`eval_injection.py`; 24 files, 12 attacks × 2 positions;
gemini-3.5-flash-lite, one model, no fallback). Attacks that reached their goal: **1/24 without the
defences** (the code word copied into offence names), **0/24 with them**. The defences never fired
in that run, so the model itself resisted most attacks; n is small. Worst case
(`eval_guard_worst_case.py`, attack sentences inserted into real analyses): the output guard removed
120/192 verdict, 120/144 fake-law and 72/96 prompt-leak insertions and no genuine sentence;
paraphrased verdicts ("There is no doubt that the accused did it") pass. Limitation, not tuned away.

**Judgment-citation faithfulness** (`eval_citation_faithfulness.py`; 92 answers by
gemini-3.5-flash-lite; judges gpt-oss-120b and qwen3.8-27b, blind to each other). 19 sentences cite a
judgment: both judges SUPPORTED 14/19 (74%), none judged UNSUPPORTED by both; agreement 0.895,
Cohen's κ 0.72 (3-way), 0.86 (supported vs not). Statute-citation control (19): both SUPPORTED
13/19, κ 0.35 / 0.58. Weak points: "PARTIAL" mostly where the sentence generalises a passage, and
3/19 sentences name a different case than the one cited. Small sample; human check of the 5
non-unanimous items is future work.

**Case analysis on real SC judgments** (`eval_case_analysis.py`, 120 judgments; facts with every
section number masked; gold = offences cited in the judgment's Facts and Ruling; 110 analysed by
gpt-oss-120b, 10 by gemini-3.5-flash-lite after a quota fallback). Model's proposal: at least one
correct offence **65%** (95% CI 57-73%); precision / recall / F1 0.45 / 0.35 / 0.36; exactly the gold
set 9%. Similar judgments: the top one shares a gold offence in **68%** (58-76%), precision@3 0.68.
The IPC validation filter removed 15 proposals, 5 of them correct (F1 0.36 -> 0.35): it is a safety
filter against non-IPC numbers, not an accuracy gain. 793 of 1,141 extracted facts were verified
against the text by quote.
