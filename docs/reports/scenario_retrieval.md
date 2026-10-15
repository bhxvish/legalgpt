# Retrieval on everyday fact-pattern questions

**Problem.** The retrieval floor (0.55 cosine similarity) was calibrated on short legal questions
("What is the punishment for theft?"). People more often describe what happened ("Someone slapped me
at the bus stop"). Such narratives score 0.3–0.5 against statute text, so almost all of them were
refused. A lower floor alone doesn't help: civil narratives ("My landlord won't return my deposit")
score just as high, and the embedding model ranks the wrong sections first.

**Fix (three parts, all local, no LLM call).**

1. **Lay-to-statutory query expansion** (`app/core/lay_terms.py`). 20 rules map everyday
   descriptions of an offence category to the IPC's wording. For example, "slapped / punched /
   kicked" maps to "voluntarily causing hurt; assault; criminal force", and "took my money and
   never delivered" maps to "cheating: deceiving a person and dishonestly inducing delivery of
   property". The wording is appended to the search query only. Triggers are everyday words, never
   IPC terms: an early version that also fired on "murder" sent "punishment for murder" to s.300
   instead of s.302.
2. **Scope check** (`app/core/scope.py`). A logistic regression on the question's MiniLM embedding
   answers "is this about an IPC offence?". It was trained on 100 separate questions
   (`app/core/scope_examples.json`, 95% 5-fold CV accuracy). Below the 0.55 floor, an in-scope
   question may use sources down to 0.40. When the lexicon added wording, the question is answered
   **only** if the scope check agrees. The check reads the plain question, not the expanded one.
3. **Boilerplate penalty.** State amendments and Chapter I general provisions are down-weighted and
   don't count as a substantive match. It had no measurable effect on these sets (see the table).

## Evaluation

`docs/eval/scenario_questions.json` has 40 everyday questions, each with the acceptable sections:
20 **dev** questions (used while designing the fix) and 20 **test** questions (held out and scored
once, after the configuration was frozen). There are also 24 out-of-scope questions that must be
refused: 10 civil narratives plus the 14 short ones from `scripts/calibrate_retrieval.py`. A hit
means an acceptable section is among the 6 sources given to the model.

| Configuration | Dev hit | Test hit | Test top-1 | Test refused | Out-of-scope answered |
|---|---|---|---|---|---|
| Before (floor only) | 10% | 5% | 5% | 90% | 0 / 24 |
| + scope check only | 35% | 25% | 15% | 45% | 0 / 24 |
| + lexicon only (no scope check) | 75% | 60% | 45% | 25% | **1 / 24** |
| Full system without boilerplate penalty | 85% | 80% | 40% | 15% | 0 / 24 |
| **Full system** | **85%** | **80%** | **40%** | **15%** | **0 / 24** |

Short legal questions are unaffected: 22 of 22 still retrieve their section, including "What is the
punishment for murder?" → s.302.

**Test misses (4 of 20):**
- "My phone was taken from my bag by a co-passenger while I was asleep": "was taken" is passive,
  and no rule fired.
- "An online seller took my payment … never delivered": the right sections were retrieved at
  0.54, but the scope check said no.
- "A bank manager transferred customers' deposits into his own account": a lexicon rule matched,
  but the scope check said no, so it was refused.
- "A goon … would kidnap his son unless he paid": s.390 (robbery) was retrieved instead of
  extortion.

**Reproduce:** `python backend/scripts/eval_scenario_retrieval.py [--no-scope] [--no-lexicon] [--no-boilerplate] --show`

**Caveats.**
- The questions, expected sections and lexicon rules were written by Claude (an AI); a lawyer
  should review the expected sections.
- 20 test questions is small, so the 80% has a wide confidence interval (about 58–92%).
- The lexicon is hand-written and covers about 20 offence categories. Rarer offences described in
  everyday words will still be refused, safely.

## Update 2026-10-15: LLM query rewrite, retrained scope check

Evaluated on the benchmark's 60 held-out scenarios and 30 held-out out-of-scope questions
(`benchmark/`). Design used only the other 248 scenarios (`eval_scenario_bench.py`).

| retrieval | held-out scenarios: acceptable section among sources | out-of-scope answered |
|---|---|---|
| embedding only | 2% | 0/30 |
| + lexicon (20 rules) + scope check (the 2026-10-14 system) | 45% | 0/30 |
| + lexicon expanded to 41 rules | 50% | 0/30 |
| + LLM query rewrite (`query_rewriter.py`), gate 0.50 | 83-90% | 0/30 |
| + scope check retrained (486 examples), gate 0.58 | **88-95%** | **0/30** |

The rewrite rows range over two models' rewrites (gpt-oss-120b, Gemini 3.5 flash-lite).

**How the rewrite works.** The chat model names the likely IPC offences and sections as JSON. Their
statute text is fetched; the answer must still be written from it and cite it.

**When the rewrite is used.** Only when the model says the matter is criminal *and* the scope check
gives at least the gate. The gate is the scope check's threshold, set 0.05 above the highest
out-of-scope validation question, so no validation civil question reaches it whatever the LLM says.

**Answer quality.** End to end (the full answer, scored by the benchmark), LegalGPT's held-out
scenario accuracy rose from 43-45% to:

| model | after the fix |
|---|---|
| gpt-5.4-mini | 88% |
| gpt-oss-120b | 82% |
| Gemini 3.5 flash-lite | 90% |

Each rise is significant (p < 0.001), and each is at parity with the same model answering alone.
Wrong punishments and civil questions treated as offences stay at 0%. These end-to-end runs used the
scope check before retraining. Details: `paper_analysis.md`.

The earlier evaluation set (`docs/eval/scenario_questions.json`) improved from 85% to 90% on dev
and from 80% to 85% on test, with 0 of 24 out-of-scope questions answered.
