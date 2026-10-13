# Annotation agreement: team labels vs an independent second annotation

**Who annotated:** the team's labels (annotator `team-sheet`) against a second, blind annotation by
Claude (Opus 5.5, an AI annotator), made on 2026-10-13. The second annotator worked from the sentence
text only, following [annotation_guideline.md](../annotation_guideline.md), and did not see the
team's labels until all five cases were done. It is a second independent annotation, **not** a
second human or a legal expert. Its labels are kept in `data/agreement/claude_blind_labels.json` and
are not merged into the corpus.

## Result

| case | sentences | raw agreement | Cohen's κ |
|---|---|---|---|
| case04 | 26 | 96% | 0.948 |
| case06 | 55 | 91% | 0.872 |
| case07 | 43 | 93% | 0.906 |
| case19 | 33 | 100% | 1.000 |
| case20 | 30 | 90% | 0.844 |
| **all 5** | **187** | **93.6%** | **0.911** |

κ above 0.8 is usually read as "almost perfect" agreement.

Confusion (rows = team, columns = second annotator):

| team \ second | Facts | Law Applied | Precedent | Argument | Ruling |
|---|---|---|---|---|---|
| Facts | 62 | 0 | 0 | 0 | 8 |
| Law Applied | 0 | 1 | 1 | 0 | 0 |
| Precedent | 0 | 0 | 23 | 0 | 0 |
| Argument | 0 | 0 | 0 | 38 | 1 |
| Ruling | 2 | 0 | 0 | 0 | 51 |

## What the disagreements are

10 of the 12 disagreements are **Facts vs Ruling**: sentences where the court restates evidence
inside its own analysis ("The postmortem report indicates that…", "PW.3 claims to be an eye
witness…"). The team labelled most of these Facts; the second annotator labelled them Ruling
because they sit inside the court's evaluation. The guideline covers both readings ("witness
testimony as recorded" is Facts; "appreciation of evidence" is Ruling). A future guideline
revision could add a tie-break: *a restatement of evidence is Facts unless the sentence also says
what the court makes of it.*

There is no Ruling vs Precedent confusion in this sample. That boundary is where the classifier
struggles most, and the guideline now has an explicit tie-break for it (2026-10-13).

## Limits

- **Five short cases** (26–55 sentences, mostly road-accident and negligence judgments). Long
  Supreme Court judgments with nested quotations are harder, so agreement there is probably lower.
- **An AI second annotator** measures how clear the guideline is to a careful independent reader,
  not agreement between two trained legal annotators. A human double annotation is still the
  stronger evidence.

Reproduce: `data/agreement/claude_blind_labels.json` holds one letter per sentence
(F/L/P/A/R, in sentence order). κ compares those letters with the team's labels
(`AnnotationStore.load_case`) using `app.annotation.agreement.AgreementCalculator.cohen_kappa`.
