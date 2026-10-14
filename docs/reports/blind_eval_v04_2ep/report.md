# Blind answer rating: LoRA v0.3-ipcqa vs LoRA v0.4-plain-2ep

Rater: Claude (AI judge, Opus 5.5) - not a human or legal expert; answer styles are recognisable, so blinding was only partial. 20 questions; each model's answer was shown as A or B in random order with the retrieved sources, and rated 1-5 per criterion before the key was opened.

| criterion | LoRA v0.3-ipcqa | LoRA v0.4-plain-2ep |
|---|---|---|
| relevance | 4.30 | 4.60 |
| groundedness | 4.75 | 4.70 |
| completeness | 3.85 | 4.25 |
| **mean of all three** | **4.30** | **4.52** |

Per question (sum of the three scores, 3-15): LoRA v0.3-ipcqa better on 2, LoRA v0.4-plain-2ep better on 4, tie on 14.

| # | question | LoRA v0.3-ipcqa | LoRA v0.4-plain-2ep | better |
|---|---|---|---|---|
| 1 | What is the punishment for causing death by negligence? | 15 | 15 | tie |
| 2 | What is the difference between culpable homicide and murder? | 5 | 10 | LoRA v0.4-plain-2ep |
| 3 | What is the punishment for theft? | 15 | 15 | tie |
| 4 | A man took his neighbour's bicycle without permission and sold it. Whi | 8 | 8 | tie |
| 5 | What is dowry death under the IPC? | 14 | 15 | LoRA v0.4-plain-2ep |
| 6 | What is the punishment for voluntarily causing hurt? | 15 | 15 | tie |
| 7 | What is cheating and dishonestly inducing delivery of property under S | 14 | 14 | tie |
| 8 | What is the punishment for criminal intimidation? | 8 | 10 | LoRA v0.4-plain-2ep |
| 9 | What is the punishment for rash driving on a public way? | 15 | 15 | tie |
| 10 | What does Section 498A of the IPC provide? | 14 | 14 | tie |
| 11 | What is the punishment for criminal breach of trust? | 15 | 15 | tie |
| 12 | What is the punishment for defamation? | 15 | 15 | tie |
| 13 | When does the right of private defence of the body extend to causing d | 15 | 14 | LoRA v0.3-ipcqa |
| 14 | What is the punishment for kidnapping? | 15 | 15 | tie |
| 15 | What is the punishment for wrongful confinement? | 15 | 15 | tie |
| 16 | What is the offence of assault or criminal force to a woman with inten | 15 | 15 | tie |
| 17 | A surgeon's careless mistake during an operation caused the patient's  | 8 | 15 | LoRA v0.4-plain-2ep |
| 18 | What is the punishment for robbery? | 15 | 14 | LoRA v0.3-ipcqa |
| 19 | What is the punishment for house-trespass by night? | 7 | 7 | tie |
| 20 | What is the procedure to file for divorce in India? | 15 | 15 | tie |

## Rater notes

- 2: One answer uses s.299 correctly (murder described loosely); the other gives a wrong 'negligence vs malice aforethought' account citing s.301.
- 4: Both refused although the sources (s.378 illustration, s.403) partly cover the question.
- 8: One gives the right punishment but attributes it to s.503; the other quotes s.503 and never states the punishment.
- 17: One answers with s.304A [1]; the other refuses although s.304A was source [1].
- 18: One leaves out the 14-year term for highway robbery at night.
- 19: Both use s.448 instead of s.456 (by night).
