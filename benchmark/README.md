# IPC benchmark: grounded answers, everyday questions and rule-checked fact patterns

A benchmark for legal question answering and fact-checking under the Indian Penal Code, 1860.
It was built for LegalGPT but works with any system. It has four parts and 1,056 items:

| File | Items | What it tests | Labels |
|---|---|---|---|
| `statute_qa.jsonl` | 365 (313 punishment, 52 definition) | Does the answer cite the right section, and state the punishment the statute gives? | **Gold by construction**: the section and its punishment terms are read from the statute text |
| `scenarios.jsonl` | 308 (20 dev, 288 test) | Plain-language fact patterns ("someone slapped me at the bus stop"): is an acceptable section cited? | **Silver**: written by an AI, to be confirmed by legal reviewers |
| `out_of_scope.jsonl` | 54 | Civil, family, consumer or general questions: are they wrongly answered as IPC offences? | Silver |
| `fact_patterns.jsonl` | 329 | Given facts and a section: CONSISTENT, INCONSISTENT or INSUFFICIENT? Includes **one-fact counterfactuals** and **paraphrases** | **Rule-computed** verdicts for 306 generated cases (their wording is to be confirmed by reviewers); silver for the 23 hand-written cases |

`stats.json` has the counts. `ipc_bns_map.json` maps common IPC offences to BNS numbers. It was
written from memory and is **unverified**; it is used only to flag one kind of IPC/BNS confusion.

## How it was built (`backend/scripts/build_benchmark.py`)

- **Statute questions** (`app/benchmark/statute.py`). Every section of the form "Whoever *conduct*,
  shall be punished with *punishment*" becomes a punishment question. Its gold terms ("two years",
  "life", "death") are taken from the punishment clause. Sections that define an offence or concept
  become definition questions. `lora_held_out` marks the 27 sections that the LegalGPT LoRA adapter
  never saw in training.
- **Scenarios** (`app/benchmark/scenario_data.py`) were written by Claude (an AI), not a lawyer:
  268 new questions plus 40 from LegalGPT's earlier evaluation. Each is tagged with an offence group
  whose sections are all acceptable (definition, punishment and aggravated forms). The 20 `dev`
  questions were used to design LegalGPT's scenario retrieval, so report results on `test`.
- **Fact patterns** (`app/benchmark/factgen.py`) cover 7 sections (279, 304A, 304B, 323, 337, 338,
  379) with 4–6 scenario frames each.
  - A base case states every element of the offence.
  - A *paraphrase* states the same facts in other words.
  - A *flip* changes exactly one element to a contradicting fact.
  - An *omit* leaves one element out. Elements implied by the rest of the story are never omitted.
  - Gold verdicts come from LegalGPT's rule layout (`docs/zk/verdict/layout.json`, generated from
    the Prolog rules). They are therefore correct **if** each sentence expresses the value it is
    meant to, which is what reviewers check.

## Review status and labelling

`labelling/*.csv` has one row per item, with columns for a reviewer, agree yes/no, a corrected
label and a note. **No item has yet been reviewed by a legal expert.** Until then, report results on
the silver parts as provisional, and say so. A suggested protocol: two independent reviewers per
item (law students or advocates), disagreements resolved by discussion, and Cohen's κ reported.

## Use

```bash
python backend/scripts/run_baselines.py run --systems legalgpt:openai/gpt-oss-120b closed:openai/gpt-oss-120b
# other providers (keys in .env; docs/GUIDE.md "Other models"); --smoke checks keys on 4 questions first
python backend/scripts/run_baselines.py run --smoke --systems legalgpt+gemini:gemini-3.5-flash-lite closed+gemini:gemini-3.5-flash-lite
python backend/scripts/run_baselines.py score
python backend/scripts/run_counterfactual.py run --model qwen/qwen3.8-27b --frames 0 && python backend/scripts/run_counterfactual.py score
```

Scoring is automatic (`app/benchmark/metrics.py`):
- **Statute questions:** cited sections, with IPC kept separate from BNS, CrPC and other Acts;
  punishment terms; invented sections; BNS confusion.
- **Scenarios:** hits.
- **Out-of-scope questions:** IPC sections cited for a civil question.

## Limitations

- IPC only. Since 1 July 2024 new offences fall under the BNS, but the IPC still governs offences
  committed before then.
- Scenario and fact-pattern wording was written by an AI. It is simpler and more uniform than real
  case descriptions, and the 306 generated fact patterns follow templates.
- Automatic scoring checks *which sections and terms* an answer gives, not whether its reasoning is
  sound.
- English only.

## Licence

The IPC text is public domain (India Code). The questions, fact patterns and code were created for
this project. Choose a licence before release; CC BY 4.0 is suggested for the data.
