# Analysis for the paper: significance, ablation, errors, IPC vs BNS

All numbers come from the benchmark runs in `data/reports/bench/baselines/` (seed 7: 40 punishment and
20 definition questions, 60 held-out scenarios, 30 out-of-scope questions). Scenario and out-of-scope
labels are **silver** (AI-written) and still need legal review. Scripts are named in each section.

## 1. Paired significance tests (`backend/scripts/paired_tests.py`)

Every system answers the same questions, so each comparison is paired. Only the questions on which
the two systems disagree carry information: *only A* and *only B* count them. The p-value is the
exact McNemar test (two-sided binomial test on the disagreements).

| A (closed-book) | B (inside LegalGPT) | measure | n | A | B | only A | only B | p |
|---|---|---|---|---|---|---|---|---|
| gpt-5.4-mini | gpt-5.4-mini (with query rewrite) | wrong punishment ↓ | 40 | 18% | 0% | 7 | 0 | 0.016 |
| | | right section (statute) | 60 | 73% | 82% | 10 | 15 | 0.42 |
| | | acceptable section (scenario) | 60 | 83% | 88% | 6 | 9 | 0.61 |
| | | civil question answered as an offence ↓ | 30 | 47% | 0% | 14 | 0 | <0.001 |
| gemini-3.5-flash-lite | gemini-3.5-flash-lite (before the rewrite) | wrong punishment ↓ | 40 | 18% | 0% | 7 | 0 | 0.016 |
| | | right section (statute) | 60 | 80% | 82% | 9 | 10 | 1.00 |
| | | acceptable section (scenario) | 60 | 78% | 43% | 25 | 4 | <0.001 |
| | | civil question answered as an offence ↓ | 30 | 47% | 0% | 14 | 0 | <0.001 |
| gpt-oss-120b | gpt-oss-120b (before the rewrite) | wrong punishment ↓ | 40 | 55% | 0% | 22 | 0 | <0.001 |
| | | right section (statute) | 60 | 40% | 82% | 5 | 30 | <0.001 |
| | | acceptable section (scenario) | 60 | 85% | 45% | 26 | 2 | <0.001 |
| | | civil question answered as an offence ↓ | 30 | 90% | 0% | 27 | 0 | <0.001 |

Within LegalGPT, before vs after the scenario fix:

| system | measure | n | before | after | only before | only after | p |
|---|---|---|---|---|---|---|---|
| LegalGPT + gpt-5.4-mini | acceptable section (scenario) | 60 | 45% | 88% | 0 | 26 | <0.001 |
| LegalGPT-LoRA v0.4 (new prompt and lexicon only) | acceptable section (scenario) | 60 | 8% | 13% | 0 | 3 | 0.25 |
| LoRA v0.4 → v0.5 (retrained with scenario examples) | acceptable section (scenario) | 60 | 13% | 43% | 0 | 18 | <0.001 |

**What the tests support.**
- **Safety gains are significant for all three models:**
  - no wrong punishments (p ≤ 0.016);
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
- **Gemini and gpt-oss-120b** have not yet been re-run with the fix. Their scenario rows above are
  the old pipeline, so the scenario comparison for them is still pending.

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
| + LLM query rewrite (gate 0.5) | *pending: needs free-model rewrites for these 90 questions* | | | |

End-to-end, the answer stage adds the prompt rule ("applying a defined offence to the facts is
answering from the sources"):

| configuration (LegalGPT, 60 held-out scenarios) | acceptable section cited |
|---|---|
| gpt-5.4-mini, original pipeline | 45% |
| gpt-5.4-mini, + expanded lexicon + rewrite + prompt rule | **88%** |
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
