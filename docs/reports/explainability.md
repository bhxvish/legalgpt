# Explainability results

How LegalGPT explains its decisions, and how faithful each explanation is. Reproduce: backend/scripts/eval_bias.py, eval_reasoning.py, eval_word_importance.py, analyze_lora.py, then make_explain_report.py.

## Bias check: gender swapped (40 scenario pairs), gpt-oss-120b (Groq)

Both versions run twice; a section counts only if it holds in both runs of one version and neither run of the other.

| system | group | no difference | required by law | run-to-run variation | possible bias |
|---|---|---|---|---|---|
| model alone | sensitive | 3 | 17 | 3 | 1 |
| model alone | neutral | 3 | 0 | 9 | 4 |
| LegalGPT | sensitive | 4 | 16 | 2 | 2 |
| LegalGPT | neutral | 8 | 0 | 6 | 2 |

- Model alone: possible bias in 5 of 40 pairs; a gender-specific section relied on in both versions (likely misapplied in one) in 10 pairs; 0 stereotyped sentences.
- LegalGPT: possible bias in 4 of 40 pairs; a gender-specific section relied on in both versions (likely misapplied in one) in 7 pairs; 0 stereotyped sentences.
- Model alone vs LegalGPT: exact McNemar p = 1.0.

## Bias check: gender swapped (40 scenario pairs), Gemini 3.5 flash-lite

Both versions run twice; a section counts only if it holds in both runs of one version and neither run of the other.

| system | group | no difference | required by law | run-to-run variation | possible bias |
|---|---|---|---|---|---|
| model alone | sensitive | 7 | 7 | 5 | 5 |
| model alone | neutral | 0 | 1 | 11 | 4 |
| LegalGPT | sensitive | 4 | 13 | 2 | 5 |
| LegalGPT | neutral | 8 | 0 | 5 | 3 |

- Model alone: possible bias in 9 of 40 pairs; a gender-specific section relied on in both versions (likely misapplied in one) in 17 pairs; 0 stereotyped sentences.
- LegalGPT: possible bias in 8 of 40 pairs; a gender-specific section relied on in both versions (likely misapplied in one) in 15 pairs; 0 stereotyped sentences.
- Model alone vs LegalGPT: exact McNemar p = 1.0.

## Bias check: caste, income level, region and age (10 scenarios each, where none of these should matter)

One sentence about the accused differs between two otherwise identical versions (a stated age or age word is swapped where the scenario has one). Both versions run twice: possible bias means a section held in both runs of one version and neither run of the other. The last column is the looser rule with one control.

| model | attribute | system | no difference | run-to-run variation | possible bias | with one control |
|---|---|---|---|---|---|---|
| gpt-oss-120b (Groq) | caste | model alone | 2 | 5 | 3 | 6 |
| gpt-oss-120b (Groq) | caste | LegalGPT | 6 | 1 | 3 | 6 |
| gpt-oss-120b (Groq) | income | model alone | 1 | 5 | 4 | 8 |
| gpt-oss-120b (Groq) | income | LegalGPT | 4 | 3 | 3 | 5 |
| gpt-oss-120b (Groq) | region | model alone | 3 | 6 | 1 | 6 |
| gpt-oss-120b (Groq) | region | LegalGPT | 2 | 6 | 2 | 7 |
| gpt-oss-120b (Groq) | age | model alone | 1 | 7 | 2 | 6 |
| gpt-oss-120b (Groq) | age | LegalGPT | 3 | 1 | 6 | 7 |
| Gemini 3.5 flash-lite | caste | model alone | 0 | 5 | 5 | 9 |
| Gemini 3.5 flash-lite | caste | LegalGPT | 9 | 0 | 1 | 1 |
| Gemini 3.5 flash-lite | income | model alone | 0 | 4 | 6 | 9 |
| Gemini 3.5 flash-lite | income | LegalGPT | 3 | 3 | 4 | 7 |
| Gemini 3.5 flash-lite | region | model alone | 2 | 5 | 3 | 6 |
| Gemini 3.5 flash-lite | region | LegalGPT | 5 | 4 | 1 | 4 |
| Gemini 3.5 flash-lite | age | model alone | 1 | 1 | 7 | 8 |
| Gemini 3.5 flash-lite | age | LegalGPT | 5 | 1 | 4 | 5 |

- gpt-oss-120b (Groq), four attributes together: possible bias in 10 of 40 pairs for the model alone and 14 inside LegalGPT (exact McNemar p = 0.424); 0 stereotyped sentences, 0 changed refusals. LegalGPT's search found different sections in 14 of 14 flagged pairs and 16 of 26 others.
- Gemini 3.5 flash-lite, four attributes together: possible bias in 21 of 40 pairs for the model alone and 10 inside LegalGPT (exact McNemar p = 0.013); 0 stereotyped sentences, 0 changed refusals. LegalGPT's search found different sections in 10 of 10 flagged pairs and 20 of 30 others.

LegalGPT cites only what it retrieves, so this shows where the added sentence enters, not why the result changed. Some differences are the definition and the punishment of one offence (s.441 and s.447). 10 scenarios per attribute is too few to conclude that a system is or is not biased.

## Bias check: how the figures moved (caste, income, region and age together)

One offence counted once: a definition and its punishment section are one difference, not two. The filter removes the added sentence about the person from the search, so it cannot change which sections are found; it was built around one wording per attribute, then frozen and tested on a new wording in the same scenarios. The model alone has no search, so the filter does not touch it.

| model | stage | model alone | LegalGPT | pairs | exact McNemar p |
|---|---|---|---|---|---|
| gpt-oss-120b (Groq) | As first reported | 13 | 18 | 40 | 0.36 |
| gpt-oss-120b (Groq) | One offence counted once | 10 | 14 | 40 | 0.42 |
| gpt-oss-120b (Groq) | With the filter | 10 | 6 | 40 | 0.39 |
| gpt-oss-120b (Groq) | New wordings, filter on | 10 | 4 | 40 | 0.15 |
| Gemini 3.5 flash-lite | As first reported | 21 | 13 | 40 | 0.077 |
| Gemini 3.5 flash-lite | One offence counted once | 21 | 10 | 40 | 0.013 |
| Gemini 3.5 flash-lite | With the filter | 21 | 3 | 40 | 4e-05 |
| Gemini 3.5 flash-lite | New wordings, filter on | 17 | 11 | 40 | 0.21 |

- gpt-oss-120b (Groq), new wordings: LegalGPT flagged 17 of 40 with the filter off and 4 with it on (exact McNemar p = 0.00098).
  - It removed the new caste and age wordings: 9 off, 0 on, of 20 (p = 0.0039). It missed the new income and region wordings, so the search got the same text both times: 8 and 4 of 20, which is run-to-run noise.
- Gemini 3.5 flash-lite, new wordings: LegalGPT flagged 15 of 40 with the filter off and 11 with it on (exact McNemar p = 0.42).
  - It removed the new caste and age wordings: 8 off, 3 on, of 20 (p = 0.18). It missed the new income and region wordings, so the search got the same text both times: 7 and 8 of 20, which is run-to-run noise.

## Reasoning trace faithfulness (18 held-out scenarios)

- Key facts named by the trace that really change the conclusion when removed: 26 of 38.
- Mean agreement (named == influential): 0.695.
- Sections the trace says apply that are decisive when left out: precision 0.861, recall 0.646.
- Key facts quoted word for word: 97%; invalid sections/sources removed per trace: 0.889.
- Controls (unchanged input, asked again) that changed: facts 50%, sources 33%.

## Key-word explanations: deletion test (57 held-out scenarios, no model calls)

| words removed | match drop: top words | match drop: random words | crime-probability drop: top | random |
|---|---|---|---|---|
| 1 | 0.1731 | 0.0191 | 0.0736 | 0.0097 |
| 2 | 0.2225 | 0.0437 | 0.1332 | 0.0193 |
| 3 | 0.2444 | 0.0763 | 0.1726 | 0.0408 |

## LoRA: what fine-tuning changed (v0.7-3b-l4, base Qwen/Qwen2.5-3B-Instruct)

- Largest relative updates: gate_proj 0.013421, up_proj 0.012911 (attention q/k/v/o about 0.007302); layers changed most: [2, 3, 1, 4, 5].
- Effective rank 13.04 of r=16; amplification factor 1.32.

- Tuned vs base on its own answers (16 held-out scenarios): answers 1.7383 nats per token more likely (citation markers 1.2667): a change of style across the whole answer.
- Leave-one-source-out inside the model: the most influential source is cited in 16 of 16 answers; mean drop 0.5427 (cited) vs 0.0007 (uncited).
