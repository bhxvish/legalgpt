# LoRA v0.4: plain-language answers

**Change.** v0.3 was trained to answer with the statute's own words ("whoever causes the death ...
shall be punished with imprisonment of either description for a term which may extend to two
years..."). For v0.4, offence sections of the form "Whoever *conduct*, shall be punished with
*punishment*" (320 of 567 sections) get training answers that **explain** the section. The
punishment is rewritten in plain words, such as "imprisonment (simple or rigorous) of up to two
years, or a fine, or both", followed by who it applies to, each with its citation
(`app/finetuning/ipc_qa.py`, `plain_punishment` and `offence_parts`). The rewrite is mechanical, so
the targets stay correct by construction. Other sections are still quoted.

**Training.** The same as v0.3: corpus v0.2, `--ipc-qa --no-case-refusals`, the same IPC sections
held out. One difference: v0.4 ran 3 epochs (the trainer default) and v0.3 ran 2, so the comparison
mixes the answer-style change with one more epoch. A 2-epoch re-run would isolate it.

## Held-out IPC sections (41 questions on 29 sections never seen in training)

| | v0.3 | v0.4 |
|---|---|---|
| Answers (not "I don't know") | 100% | 100% |
| Cites the asked section | 98% | 98% |
| All citations valid | 100% | 100% |
| Answers in plain wording | 0 / 41 | **13 / 41** |
| Still in statutory wording | 39 / 41 | 28 / 41 |
| Punishment terms match the statute (punishment questions) | 11 / 12 | **12 / 12** |
| Mean answer length / time | 366 chars, 3.6 s | 373 chars, 3.7 s |

The plain wording appears on punishment questions; "what does Section X provide?" answers are
mostly still quoted. Details: `data/reports/ipc_qa_eval_v04.json`.

## Blind rating, v0.3 vs v0.4 (20 questions, `docs/eval/blind_questions.txt`)

Both models answered through the app's current retrieval (with the scenario fixes), shuffled as
A/B and rated 1–5 per criterion before unblinding. Rater: Claude, an AI judge, not a lawyer. The
two styles are easy to tell apart, so the blinding was only partial.

| | v0.3 | v0.4 |
|---|---|---|
| relevance | 4.30 | 4.60 |
| groundedness | 4.75 | 4.70 |
| completeness | 3.85 | 4.25 |
| **mean** | **4.30** | **4.52** |

v0.4 was better on 4 questions, worse on 2, and tied on 14. With 20 questions this difference is
not statistically significant.
- **Gains:** it answers the surgeon scenario (s.304A), which v0.3 refused, and states the punishment
  for criminal intimidation, which v0.3 only defined.
- **Losses:** it attaches s.506's punishment to s.503 (a citation slip), and it drops the
  14-year term for robbery on a highway at night.

The rubric doesn't score readability, where the difference is largest. Full sheet, key and notes
are in `blind_eval_v04/`.

**Status:** v0.4 is the newest adapter, so the chat's *Tuned (LoRA)* option uses it.
