# Fact check against a labelled test set

15 fact patterns from `docs/eval/verification_cases.json` ({'CONSISTENT': 7, 'INCONSISTENT': 5, 'INSUFFICIENT': 3}); fact extraction by openai/gpt-oss-120b, two runs, checked in Prolog.

**Agreement: 13/15 = 87%**

| expected \ got | CONSISTENT | INCONSISTENT | INSUFFICIENT |
|---|---|---|---|
| CONSISTENT | 6 | 0 | 1 |
| INCONSISTENT | 0 | 4 | 1 |
| INSUFFICIENT | 0 | 0 | 3 |

| case | section | expected | got | element results |
|---|---|---|---|---|
| v01 | 304A | CONSISTENT | CONSISTENT | caused_death: satisfied, act_type: satisfied, intent_to_kill: satisfied, knowledge_likely_death: satisfied |
| v02 | 304A | INCONSISTENT | INCONSISTENT | caused_death: satisfied, act_type: violated, intent_to_kill: violated, knowledge_likely_death: violated |
| v03 | 304A | INSUFFICIENT | INSUFFICIENT | caused_death: satisfied, act_type: missing, intent_to_kill: missing, knowledge_likely_death: missing |
| v04 | 279 | CONSISTENT | CONSISTENT | drove_vehicle: satisfied, on_public_way: satisfied, act_type: satisfied, endangered_safety: satisfied |
| v05 | 279 | INCONSISTENT | INCONSISTENT | drove_vehicle: satisfied, on_public_way: violated, act_type: satisfied, endangered_safety: satisfied |
| v06 | 279 | INSUFFICIENT | INSUFFICIENT | drove_vehicle: satisfied, on_public_way: missing, act_type: missing, endangered_safety: missing |
| v07 | 337 | CONSISTENT | CONSISTENT | caused_hurt: satisfied, act_type: satisfied, endangered_safety: satisfied |
| v08 | 338 | CONSISTENT | CONSISTENT | caused_grievous_hurt: satisfied, act_type: satisfied, endangered_safety: satisfied |
| v09 | 323 | CONSISTENT | CONSISTENT | caused_hurt: satisfied, act_type: satisfied |
| v10 | 323 | INCONSISTENT | INCONSISTENT | caused_hurt: satisfied, act_type: violated |
| v11 | 379 | CONSISTENT | CONSISTENT | movable_property: satisfied, taken_from_possession: satisfied, without_consent: satisfied, dishonest_intention: satisfied, property_moved: satisfied |
| v12 | 379 | INCONSISTENT | INCONSISTENT | movable_property: satisfied, taken_from_possession: violated, without_consent: violated, dishonest_intention: violated, property_moved: satisfied |
| v13 | 379 | INSUFFICIENT | INSUFFICIENT | movable_property: satisfied, taken_from_possession: satisfied, without_consent: satisfied, dishonest_intention: missing, property_moved: satisfied |
| v14 | 304B | CONSISTENT | INSUFFICIENT ✗ | woman_unnatural_death: satisfied, within_seven_years_of_marriage: missing, cruelty_by_husband_or_relative: satisfied, cruelty_soon_before_death: satisfied, dowry_demand: satisfied |
| v15 | 304B | INCONSISTENT | INSUFFICIENT ✗ | woman_unnatural_death: satisfied, within_seven_years_of_marriage: missing, cruelty_by_husband_or_relative: satisfied, cruelty_soon_before_death: satisfied, dowry_demand: satisfied |

## Disagreements

- **v14** (expected CONSISTENT: all five dowry-death elements stated); got INSUFFICIENT. Elements: woman_unnatural_death=satisfied, within_seven_years_of_marriage=missing, cruelty_by_husband_or_relative=satisfied, cruelty_soon_before_death=satisfied, dowry_demand=satisfied
- **v15** (expected INCONSISTENT: the death was twelve years after the marriage, so within_seven_years_of_marriage is false); got INSUFFICIENT. Elements: woman_unnatural_death=satisfied, within_seven_years_of_marriage=missing, cruelty_by_husband_or_relative=satisfied, cruelty_soon_before_death=satisfied, dowry_demand=satisfied

## Why the two dowry-death cases missed

In both v14 and v15 the extractor answered `within_seven_years_of_marriage` correctly (true in
v14: married May 2019, died February 2022; false in v15: "twelve years after the marriage"). But
the quote it gave as evidence was not verbatim in the text (it combined the two dates), so the
evidence guard (`FactSchema.validate`) downgraded the answer to *unknown*. One run quoting and one
not also counts as a disagreement, which gives *unknown* as well. The system failed on the safe
side: INSUFFICIENT, never a wrong CONSISTENT or INCONSISTENT. The likely fix is to tell the
extractor that facts derived from two places, such as dates, must quote both pieces joined with
"...", which the guard already accepts. It was not applied here, to avoid tuning the prompt on the
test set; a fresh set of cases should measure any such fix.

**Who labelled:** the fact patterns and their expected verdicts were written by Claude (AI, not a
lawyer) from the encoded element definitions. This is a check that the pipeline does what the
rules say, not a validation of the rules by a legal expert.

## After the fix (2026-10-14)

The extractor prompt now says that a fact derived from two places in the text must be quoted as
exact pieces in text order, joined by "...", which the quote guard already accepts. The few-shot
dowry example shows it. To measure this fairly, 8 new cases (v16–v23) were written **before**
running the fixed extractor: five dowry-death cases built around dates and durations, and three
regression cases on other sections.

| cases | result |
|---|---|
| v14, v15 (the two earlier misses) | 2/2 correct |
| v16–v23 (fresh) | 8/8 correct |
| v01–v05, v07–v13 (regression check) | 12/12 correct |
| v06 (regression check) | **wrong: INCONSISTENT instead of INSUFFICIENT** (see below) |
| **All 23 cases** | **22/23** |

**v06 regression.** For "Neha was driving her car… when it hit a compound wall", the extractor
answered `act_type = accidental` in both runs. That reads more into the text than it says: a
collision alone does not show there was no rashness or negligence, so `unknown` is correct. The
rules then (correctly) treat "accidental" as contradicting "rash or negligent". It passed before the
quoting fix, and the fix doesn't touch act type, so this may be run-to-run variance in the hosted
model. **Checked on 2026-10-14:** v06 fails with the new prompt (twice) *and* with the previous
prompt from before the fix (twice), so the fix did not cause it. The hosted model now reads "hit a
compound wall" as a pure accident; on 2026-10-13 it did not. It is the only *wrong verdict* (not
just "insufficient") in the 23 cases. A possible fix is to say in the act_type question that a
collision or injury alone is not an accident unless the text says no one was at fault. That should
be measured on new cases, not tuned on v06.

Fresh cases: v16 (married March 2020, died July 2023) CONSISTENT; v17 (married 2005, died 2016)
INCONSISTENT; v18 ("five years after her wedding") CONSISTENT; v19 (no marriage date) INSUFFICIENT;
v20 (death from dengue) INCONSISTENT; v21 (s.279) CONSISTENT; v22 (s.379) CONSISTENT; v23 (s.323,
attacker unknown) INSUFFICIENT. All are in `docs/eval/verification_cases.json`.
