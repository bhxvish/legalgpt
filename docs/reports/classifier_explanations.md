# How faithful are the classifier's explanations?

100 sentences from the held-out test cases of corpus v0.3 (case12, case15, case17, case19), classifier `v0.3-20261013-234503`, explaining the role it predicts. ERASER deletion metrics (DeYoung et al., 2020); the context segment is unchanged.

| method | comprehensiveness@10% | @20% | @50% | sufficiency@10% | @20% | @50% |
|---|---|---|---|---|---|---|
| ig | 0.091 | 0.104 | 0.128 | 0.054 | 0.071 | 0.052 |
| occlusion | 0.163 | 0.203 | 0.207 | 0.059 | 0.025 | -0.030 |
| random | 0.022 | 0.039 | 0.074 | 0.120 | 0.118 | 0.071 |

Comprehensiveness: higher is better (removing the words the method ranks highest should hurt the prediction most). Sufficiency: lower is better (keeping only those words should preserve it).

- IG and occlusion pick the same most important word in **20%** of sentences; mean Spearman rank correlation of their word rankings **0.13**.
- Mean share of attribution on the context segment (position + neighbouring sentences): **65%**.
- Mean IG completeness gap at 128 steps: 0.0341 logits.

Reproduce: `.venv-gpu\Scripts\python backend/scripts/eval_explanations.py --n 100`

## What this shows

- **Both methods are better than random.** Removing their top words lowers the predicted role's
  probability 2–7× more than removing random words, and keeping only their top words preserves it
  better. The explanations do reflect what the model uses.
- **Word removal (occlusion) is more faithful than Integrated Gradients on this test**, for
  comprehensiveness at every k and sufficiency at 20% and 50%. Part of that is built in, since the
  test itself removes words. It is also cheaper (one forward pass per word, about 1 s on CPU vs
  about 6 s for 128 IG steps) and easier to read ("leaving this word out drops the confidence by
  12 points"). **The Review tab therefore shows word removal by default**, with IG as a switch.
- **The two methods often disagree:** the same top word in 20% of sentences, rank correlation
  0.13. This is a known property of post-hoc explanations, not specific to this model. An
  explanation is evidence about the model, not ground truth.
- **The classifier relies heavily on context:** on average 65% of the IG attribution falls on the
  context segment (the sentence's position and its neighbours). This is consistent with the
  observed Ruling/Precedent errors, where a sentence next to a quoted precedent is itself labelled
  Precedent. The Review tab flags explanations where context dominates (over 50%).

Caveats: 100 sentences from 4 test judgments, one classifier, automatic deletion metrics only
(no human study of whether the explanations help reviewers).
