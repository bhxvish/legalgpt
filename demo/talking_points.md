# Talking points

## Opening (30 seconds, before the slides)

"AI chatbots answer legal questions confidently, but they often cite the wrong section or state a punishment
the law does not have. In India it is harder: the Indian Penal Code was replaced by the Bharatiya Nyaya
Sanhita in July 2024, and many people ask in Hindi. We built LegalGPT, which answers Indian criminal-law
questions only from the official text of the law, cites every claim, checks itself, and never gives a
verdict. And we measured it: inside LegalGPT, the same AI models stopped making wrong-punishment mistakes
completely."

## Closing (20 seconds)

"To sum up: LegalGPT is measurably safer than the same models alone, answers under the right code in
English or Hindi, cites real judgments, and every answer can be verified. Every number we showed can be
rebuilt with one command. Thank you; we are happy to take questions."

## Numbers to remember

| What | Number |
|---|---|
| Wrong punishments, model alone vs inside LegalGPT | up to 54% vs **0%** (all three models) |
| Civil questions treated as crimes | 47 to 90% vs **0%** |
| Right IPC section | 41 to 81% vs **98 to 100%** |
| Retrieval on everyday wording | 2% vs **95%** |
| Sentence-role model (macro-F1) | 0.667 vs **0.807** |
| Judgment library | **28,055** passages from 349 judgments |
| IPC to BNS sections mapped | **527** (official BPRD table) |
| Hindi retrieval | 0 of 60 vs **52 of 60** |
| Case analysis, correct offence on 120 real judgments | **65%** |
| Prompt-injection attacks that worked | 1 of 24 vs **0 of 24** |
| Paraphrased verdicts caught by the guard | 18% vs **93%** |
| Blind human check of cited judgments | 31 of 38 supported, **0 unsupported** |
| Automated tests | **357**, all passing, on every push |

## Quick answers

- **Different from ChatGPT?** ChatGPT answers from memory; LegalGPT only from the retrieved law, with citations. Measured: wrong punishments up to 54% alone, 0% inside LegalGPT.
- **Novelty?** The method: the same model compared with and without our system on held-out questions, plus checks anyone can verify (Prolog, signatures, zero-knowledge proofs).
- **Better than Nyaay AI?** Nyaay is broader. We have not tested it and it publishes no benchmark. Our strength is measured and verifiable safety, which such a product could adopt. Do not claim higher accuracy.
- **Does it give verdicts?** No, by design. A guard removes any sentence that states one, even in follow-up questions.
- **Weakest point?** No lawyer has reviewed the benchmark yet; it is disclosed and is our next step.
- **Cost?** Free tiers, or fully offline with local models.

Full details and 17 likely questions with answers: `LegalGPT_Project_Explained.pdf`.

**"Does it check caste, income, region and age too?"** Yes, on any answer or case file, along with gender
and religion. If the text names none of them, one sentence about the accused is added to two identical
copies, and each version runs twice so normal variation is not counted as bias.

**"What did you find?"** On wordings the search filter was not built around, possible bias in 10 of 40
scenarios for the model alone and 4 inside LegalGPT with gpt-oss-120b, and 17 and 11 with Gemini. Neither
difference is significant, so we do not claim LegalGPT is better on these attributes. No stereotyped wording
or changed refusal was found. With 10 scenarios per attribute this is a signal to investigate, not proof
either way.

**"Why would an irrelevant sentence change the answer?"** It can change which sections the search finds:
before the filter, that happened in all 24 of LegalGPT's flagged scenarios, and in 36 of the 56 others.
LegalGPT only cites what it retrieves, so this shows where the sentence enters, not why the result changed.

**"So what did you do about it?"** A filter keeps a sentence about who someone is out of the search; the
model still reads it, and the app shows what was left out. We froze it and tested it on new wordings. It
removed the caste and age sentences, and LegalGPT's flagged scenarios fell from 9 to 0 of 20 with gpt-oss-120b
and from 8 to 3 with Gemini. It missed the income and region sentences, so those were not really tested.
