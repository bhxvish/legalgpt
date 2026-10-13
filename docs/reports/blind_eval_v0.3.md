# Blind answer rating: Groq openai/gpt-oss-120b vs LoRA v0.3-ipcqa

Rater: Claude (AI judge, Opus 5.5) - not a human or legal expert; answer styles differ, so blinding was only partial. 20 questions; each model's answer was shown as A or B in random order with the retrieved sources, and rated 1-5 per criterion before the key was opened.

| criterion | Groq openai/gpt-oss-120b | LoRA v0.3-ipcqa |
|---|---|---|
| relevance | 5.00 | 4.60 |
| groundedness | 4.90 | 4.75 |
| completeness | 4.35 | 4.25 |
| **mean of all three** | **4.75** | **4.53** |

Per question (sum of the three scores, 3-15): Groq better on 4, LoRA better on 8, tie on 8.

| # | question | Groq | LoRA | better |
|---|---|---|---|---|
| 1 | What is the punishment for causing death by negligence? | 14 | 15 | adapter |
| 2 | What is the difference between culpable homicide and murder? | 11 | 5 | groq |
| 3 | What is the punishment for theft? | 14 | 15 | adapter |
| 4 | A man took his neighbour's bicycle without permission and sold it. Whi | 15 | 15 | tie |
| 5 | What is dowry death under the IPC? | 14 | 14 | tie |
| 6 | What is the punishment for voluntarily causing hurt? | 14 | 15 | adapter |
| 7 | What is cheating and dishonestly inducing delivery of property under S | 14 | 14 | tie |
| 8 | What is the punishment for criminal intimidation? | 15 | 8 | groq |
| 9 | What is the punishment for rash driving on a public way? | 14 | 15 | adapter |
| 10 | What does Section 498A of the IPC provide? | 15 | 14 | groq |
| 11 | What is the punishment for criminal breach of trust? | 14 | 15 | adapter |
| 12 | What is the punishment for defamation? | 14 | 15 | adapter |
| 13 | When does the right of private defence of the body extend to causing d | 15 | 15 | tie |
| 14 | What is the punishment for kidnapping? | 15 | 15 | tie |
| 15 | What is the punishment for wrongful confinement? | 14 | 15 | adapter |
| 16 | What is the offence of assault or criminal force to a woman with inten | 15 | 15 | tie |
| 17 | A surgeon's careless mistake during an operation caused the patient's  | 15 | 15 | tie |
| 18 | What is the punishment for robbery? | 14 | 15 | adapter |
| 19 | What is the punishment for house-trespass by night? | 14 | 7 | groq |
| 20 | What is the procedure to file for divorce in India? | 15 | 15 | tie |

## Rater notes

- 1: Completeness 4 where the answer cites [n] but does not name the section in the text, or quotes a provision cut off with '...' (here and elsewhere).
- 2: A: wrong account of the difference (negligence vs 'malice aforethought' is not IPC law) and cites s.301 for it. B: uses s.299 correctly but reduces murder to 'culpable homicide without an exception', omitting the s.300 clauses.
- 4: Both: retrieval returned no source above the floor for this everyday scenario (theft, s.378/379), so the app refused before any model ran. Retrieval gap, not a model difference.
- 8: A defines criminal intimidation (s.503) but never gives the punishment that was asked (s.506 was source [1]).
- 17: Both: retrieval returned only ss.1-2 for a medical-negligence scenario (s.304A missing), so refusing was correct given the sources. Retrieval gap.
- 19: A answers with s.448 (plain house-trespass), not the night offence; B correctly uses s.456.
