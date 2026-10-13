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
