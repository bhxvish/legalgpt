# Blind answer rating: LoRA v0.3-ipcqa vs LoRA v0.4-plain

Rater: Claude (AI judge, Opus 5.5) - not a human or legal expert; answer styles are recognisable, so blinding was only partial. 20 questions; each model's answer was shown as A or B in random order with the retrieved sources, and rated 1-5 per criterion before the key was opened.

| criterion | LoRA v0.3-ipcqa | LoRA v0.4-plain |
|---|---|---|
| relevance | 4.30 | 4.60 |
| groundedness | 4.75 | 4.70 |
| completeness | 3.85 | 4.25 |
| **mean of all three** | **4.30** | **4.52** |

Per question (sum of the three scores, 3-15): LoRA v0.3-ipcqa better on 2, LoRA v0.4-plain better on 4, tie on 14.

| # | question | LoRA v0.3-ipcqa | LoRA v0.4-plain | better |
|---|---|---|---|---|
| 1 | What is the punishment for causing death by negligence? | 15 | 15 | tie |
| 2 | What is the difference between culpable homicide and murder? | 5 | 10 | LoRA v0.4-plain |
| 3 | What is the punishment for theft? | 15 | 15 | tie |
| 4 | A man took his neighbour's bicycle without permission and sold it. Whi | 8 | 8 | tie |
| 5 | What is dowry death under the IPC? | 14 | 15 | LoRA v0.4-plain |
| 6 | What is the punishment for voluntarily causing hurt? | 15 | 15 | tie |
| 7 | What is cheating and dishonestly inducing delivery of property under S | 14 | 14 | tie |
| 8 | What is the punishment for criminal intimidation? | 8 | 10 | LoRA v0.4-plain |
| 9 | What is the punishment for rash driving on a public way? | 15 | 15 | tie |
| 10 | What does Section 498A of the IPC provide? | 14 | 14 | tie |
| 11 | What is the punishment for criminal breach of trust? | 15 | 15 | tie |
| 12 | What is the punishment for defamation? | 15 | 15 | tie |
| 13 | When does the right of private defence of the body extend to causing d | 15 | 14 | LoRA v0.3-ipcqa |
| 14 | What is the punishment for kidnapping? | 15 | 15 | tie |
| 15 | What is the punishment for wrongful confinement? | 15 | 15 | tie |
| 16 | What is the offence of assault or criminal force to a woman with inten | 15 | 15 | tie |
| 17 | A surgeon's careless mistake during an operation caused the patient's  | 8 | 15 | LoRA v0.4-plain |
| 18 | What is the punishment for robbery? | 15 | 14 | LoRA v0.3-ipcqa |
| 19 | What is the punishment for house-trespass by night? | 7 | 7 | tie |
| 20 | What is the procedure to file for divorce in India? | 15 | 15 | tie |

## Rater notes

- 1: Where both are correct and complete, plain wording versus quotation is not scored by this rubric (readability is noted in the report, not rated).
- 2: A: wrong account (negligence vs 'malice aforethought', cites s.301). B: s.299 correct; its description of murder is looser than s.300.
- 4: Both refused although the sources (s.378 illustration, s.403) partly cover the question; s.379 itself was not retrieved.
- 8: A gives the correct punishment (s.506's) but attributes it to s.503 [3]. B quotes the s.503 definition and never states the punishment.
- 17: A answers with s.304A [1]; B refuses although s.304A was source [1].
- 18: A leaves out the 14-year term for robbery on a highway at night.
- 19: Both use s.448 (house-trespass) rather than s.456 (by night).
