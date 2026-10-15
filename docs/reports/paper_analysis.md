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

Suggested future-work line: index the BNS alongside the IPC, verify the concordance against the
official correspondence table, and answer with both section numbers when the offence date is not
given.

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
