# LegalGPT — full guide

The detailed manual: every module, script and setting. For an overview start with the [README](../README.md).

Retrieval-augmented legal question answering for **Indian criminal law (IPC)**, with
an offline annotation + fine-tuning pipeline, an explainability layer, and a Prolog-based
verification service. See [PLANNING.md](../PLANNING.md) for modules, phases and decisions.

> Status: **all 8 phases complete** (Phase 0–7). Not legal advice: answers are grounded in the
> IPC bare act only, and the IPC was replaced by the Bharatiya Nyaya Sanhita on 1 July 2024.

## Quick start (Docker)

From a clean checkout, with [Docker](https://docs.docker.com/get-docker/) running:

```bash
cp .env.example .env            # then set GROQ_API_KEY in .env
docker compose up --build
```

- Frontend: <http://localhost:5173>:
  - **Chat:** Legal / General mode, a model switch (see "Model options in the chat" below), and
    "also verify facts against" a section;
  - **Verify**, **Annotate** and **Review** tabs.
- Backend: <http://localhost:8000/docs>.

On the first start the backend downloads the embedding model and embeds the committed IPC chunks
(`data/corpus/ipc_chunks.jsonl`) into the `chroma_db` volume — legal-mode questions work about a
minute after `/health` turns green. SWI-Prolog is inside the backend image, so verification works
without a separate service. ChromaDB, the annotation store, collected judgments and the Hugging Face
cache live in named Docker volumes (`docker compose down -v` deletes them).

**Local (LoRA) in Docker:**
- It is only enabled once an adapter exists under `data/models/lora/` (see Module 3).
- The PyTorch path needs the adapter's base model. If you already have it (e.g. from training), set
  `HF_CACHE_DIR=C:/Users/<you>/.cache/huggingface` in `.env` so Docker reuses your host cache;
  downloading it inside Docker ran at ~0.2 MB/s here (hours).
- On a CPU, answers take tens of seconds. The llama.cpp path below is the fast option on a GPU.

### With the team's data

Annotations, judgments and trained models are not in git. If you were given the team data zip,
unzip it into the repo root (it adds `data/annotation_store`, `data/raw_judgments` and
`data/models/...`), then start with the data overlay so the Annotate and Review tabs use it:

```bash
docker compose -f docker-compose.yml -f docker-compose.data.yml up --build
```

Your labels and reviews are written to `data/annotation_store/` on your machine; they do not reach
anyone else's copy unless you send that folder back.

## Run the demo

With the stack up (Docker or the local backend below):

```bash
python backend/scripts/demo.py                    # 5 questions: retrieve -> generate -> explain -> verify
python backend/scripts/demo.py --model adapter    # same questions through the LoRA-tuned model
python backend/scripts/demo.py --no-verify --only 2,5
```

Standard library only, so any Python 3.11 works. For each question it prints the answer, the
sources (`*` = cited), the evidence-match band, sentence support, warnings and — for the three fact
patterns — the rule-engine verdict with the quote behind each element, then a summary table. The
last question (income tax) is outside the IPC and shows the refusal path. The scenarios are
illustrative facts, not real cases.

## Implemented vs future work

| Module | Implemented | Future work |
|---|---|---|
| 0 Retrieval & chat | IPC bare act, hierarchical chunks, ChromaDB + MiniLM, calibrated refusal floor, scope check, 41-rule lay-term lexicon, LLM query rewrite for everyday scenarios (gated by the scope check), definition-question lookup, SSE chat with `[n]` citations, Legal/General modes, several back-ends (Groq, Gemini, OpenAI, OpenRouter, local LoRA, local 7B via llama.cpp), `OFFLINE_MODE`, **citations of real judgments** (role-labelled passages from a judgment library) | BNS/BNSS and IPC↔BNS mapping; re-ranker; a larger judgment library (High Courts) |
| 1 Dataset | Collector with screening, 5-role scheme + guideline, append-only store, kappa tooling, frozen versions with hashes | Double annotation to actually measure agreement |
| 2 Assisted labelling | InLegalBERT ensemble (fine-tuned + BiLSTM, public-corpus transfer; CV macro-F1 0.807, single model 0.739), confidence + audit review queue, review UI, explanations (integrated gradients, occlusion, nearest examples) with faithfulness metrics | Larger corpus, active learning |
| 3 Fine-tuning | 4-bit LoRA (Qwen2.5-1.5B on the laptop; 3B and 7B on Colab), instruction builder with IPC Q&A, scenario answers, refusals and a distilled query rewrite; base-vs-tuned comparison; in-app switch | Human-written answer targets; fewer wrong-section answers on scenarios when retrieval misses |
| 4 Explainability | Evidence-match band, citation checks, sentence attribution (legal claims vs other sentences), "how the sources were chosen" (floor, scope check, lexicon, rewrite, definition lookup), warnings on every legal answer | Detecting subtle legal errors, not just unsupported sentences |
| 5 Verification | Prolog element checklists for 7 sections (279, 304A, 304B, 323, 337, 338, 379), quote-checked two-run fact extraction (Groq, falling back to the free Gemini models in `EXTRA_MODELS` and then the local model when Groq's free limit is reached), rule-computed what-ifs | More sections, exceptions and Chapter IV defences |
| Case analysis | Upload a case file: quote-checked facts, possible IPC offences, element-by-element check (Prolog rules for the 7 encoded sections, model-assessed otherwise), similar past judgments with what the court held, a cited written analysis with explainability; local-only privacy mode. Never a verdict | Evaluation with lawyers on real files; more encoded sections |
| ZKML | Zero-knowledge proofs (EZKL) of fact-check verdicts (facts hidden) and of retrieval over all chunks, including the disclosed query rewrite; commitments manifest; Ed25519-signed answers; offline verifier ([docs/zk](zk/README.md)) | Proving the LLMs themselves (including the rewrite): a 1.5B+ transformer is still orders of magnitude too slow to prove on this hardware, and it would prove which model ran, not that the answer is legally right |
| Evaluation | IPC benchmark (1,056 items) with held-out sampling, baselines against three model families, paired significance tests, retrieval ablation, error analysis, counterfactual test of the rules ([benchmark](../benchmark/README.md), [reports](reports/)) | Legal review of the AI-written (silver) scenario and out-of-scope labels |

Known limitations are consolidated in [PLANNING.md](../PLANNING.md#known-limitations).

## Prerequisites

- Python 3.11
- Node.js 20+ (22 used in Docker)
- [SWI-Prolog](https://www.swi-prolog.org/download/stable) 9+ on `PATH` (needed by `pyswip` from Phase 6)
- Docker (optional, for `docker compose`)

## Configuration

```bash
cp .env.example .env
```

Fill in `GROQ_API_KEY` and `HF_TOKEN`. Path variables can stay empty to use the defaults
documented in `.env.example`. `.env` is git-ignored.

## Ingest the IPC corpus

The parsed act is committed as `data/corpus/ipc_chunks.jsonl` (bare-act text is public domain), and
the backend embeds it automatically when the index is empty — so this section is only needed to
rebuild from the PDF or re-index by hand:

```bash
python backend/scripts/ingest_corpus.py data/corpus/ipc_chunks.jsonl --reset   # no PDF needed
```

To parse from the source, download the official bare act PDF from India Code
(<https://www.indiacode.nic.in/bitstream/123456789/11091/1/the_indian_penal_code,_1860.pdf>) into
`data/raw_corpus/` (git-ignored; any filename), then from the repo root:

```bash
python backend/scripts/ingest_corpus.py data/raw_corpus/<file>.pdf --reset
```

The script checks the parse against the PDF's own *Arrangement of Sections* and prints how many
listed sections were found in the body (all 574 for the India Code PDF). `--dry-run --show 304A,302`
prints chunks without embedding; `--dump data/corpus/ipc_chunks.jsonl` regenerates the committed chunks. The index
persists in `data/chroma_db/`. Re-ingesting while the backend runs is safe.

After re-ingesting or changing the embedding model, re-check the refusal threshold:

```bash
python backend/scripts/calibrate_retrieval.py
```

## Annotate judgments (Module 1)

1. Put criminal judgments as plain `.txt` files in `data/inbox/` (git-ignored), then collect them:

   ```bash
   python backend/scripts/collect_judgments.py data/inbox
   ```

   Each file is normalized and screened. Duplicates (even reformatted copies), non-English text,
   non-criminal cases and very short files are rejected with a reason. Accepted judgments go to
   `data/raw_judgments/` with their detected court, year and cited IPC sections.
   Annotations already made in the team spreadsheet (columns `doc_id, case_name, sentence_id,
   sentence_text, label, annotator, notes`) are imported instead:

   ```bash
   python backend/scripts/import_annotations.py data/inbox/legaltech_dataset.xlsx --dry-run
   python backend/scripts/import_annotations.py data/inbox/legaltech_dataset.xlsx
   ```

   The sheet's own sentences become the case's segmentation; re-running skips cases already imported.
2. Open <http://localhost:5173/#annotate>, enter your name, pick a case and label each sentence
   (keys **1–6**, **↑/↓**, **n** for next unlabelled). Labels save immediately to
   `data/annotation_store/labels.jsonl`. Read `docs/annotation_guideline.md` first.
3. Measure agreement on double-annotated cases, then freeze a corpus version for training:

   ```bash
   python backend/scripts/annotation_agreement.py alice bob
   python backend/scripts/freeze_corpus.py v0.1 --notes "seed set"
   ```

   `freeze_corpus.py` writes `data/annotation_store/versions/<version>/manifest.json` (case ids,
   per-case and overall SHA-256, train/val/test split by case). `--verify <version>` re-checks a
   snapshot against its manifest.

Label definitions live in `backend/app/annotation/label_scheme.py`. After changing them, regenerate the
guideline: `python backend/scripts/build_guideline.py`. A test fails if the two disagree.

## BERT-assisted labelling (Module 2)

1. Train the rhetorical-role classifier (InLegalBERT, top 4 layers + head) on a frozen corpus
   version. It prints held-out test metrics and saves a checkpoint under `data/models/rrl/`
   (git-ignored). On CPU this takes about 15 minutes.

   ```bash
   python backend/scripts/train_rrl_classifier.py --version v0.1
   python backend/scripts/train_rrl_classifier.py --version v0.1 --cv 5   # case-grouped cross-validation (~1 h on CPU)
   ```

   **The model the Review tab uses now** is an ensemble (cross-validated macro-F1 0.807 vs 0.739 for
   the single fine-tuned model; `docs/reports/paper_analysis.md` §7):
   - the components:
     - InLegalBERT first trained on the public *InRhetoricalRoles* corpus (Kalamkar et al., 2022), then on our judgments;
     - a BiLSTM that reads the whole judgment in order;
   - the two are averaged and calibrated;
   - train it on a Colab L4 with `backend/scripts/colab/LegalGPT_RRL_final_Colab.ipynb` (uses `train_rrl_final.py`);
   - put the resulting `v0.3-ensemble-…` folder in `data/models/rrl/`. The app loads the newest checkpoint, ensemble or not;
   - set `REVIEW_TAU_CONF=0.9` in `.env` for it: about 2/3 of sentences are then accepted without review, ~97% correct on held-out judgments;
   - then run `build_explain_index.py`.
   The experiment notebooks are `LegalGPT_RRL_ablation_Colab.ipynb` and `LegalGPT_RRL_transfer_Colab.ipynb`.

2. Get new judgments. For example, fetch Supreme Court judgments from the CC-BY-4.0 dataset
   `labofsahil/Indian-Supreme-Court-Judgments` (only the chosen PDFs are downloaded), then collect them:

   ```bash
   python backend/scripts/fetch_sc_judgments.py --year 2023 --list-criminal 80
   python backend/scripts/fetch_sc_judgments.py --year 2023 --files 2023_10_993_1000_EN.pdf
   python backend/scripts/collect_judgments.py data/inbox
   ```

3. Predict and queue them for review. Low-confidence predictions (below `REVIEW_TAU_CONF`) and a
   random audit sample of confident ones (`REVIEW_AUDIT_RATE`) are flagged:

   ```bash
   python backend/scripts/run_assisted_labeling.py
   ```

   **Or skip steps 2–3 and upload from the app:** in the Review tab, drop `.txt` or text-based `.pdf`
   judgments on **Add judgments**. Each file is screened like `collect_judgments.py` does (duplicates,
   non-English and non-criminal texts are refused with the reason), segmented, labelled by the newest
   checkpoint in `data/models/rrl/` and queued; the panel shows progress and an **Open** link (about a
   minute per judgment on CPU, most of it loading the model the first time). Scanned PDFs without a text
   layer are refused. `POST /api/review/uploads?filename=…&uploader=…` with the file as the request body;
   `GET /api/review/uploads[/{job_id}]` for status.

4. Reviewers open <http://localhost:5173/#review>, check the highlighted sentences (Enter accepts,
   1–5 picks a role) and click **Sign off**. If more than `REVIEW_MAX_AUDIT_ERROR` of the audited
   confident predictions needed correction, the case is sent to full manual annotation instead;
   otherwise every sentence is added to the corpus with `source="bert_assisted"`.

## LoRA fine-tuning (Module 3)

Fine-tuning needs a CUDA GPU and runs from a separate environment, so the base install stays
CPU-only (~9 GB of disk for CUDA PyTorch + the base model):

```bash
python -m venv .venv-gpu
.venv-gpu\Scripts\python -m pip install --no-cache-dir -r backend/requirements-gpu.txt
```

Train a LoRA adapter on Qwen2.5-1.5B-Instruct (4-bit, fits a 4 GB GPU; ~12 minutes on an RTX 3050 Ti)
from a frozen corpus version, then compare it with the hosted model and the untuned base on held-out
questions through the same retriever:

```bash
.venv-gpu\Scripts\python backend/scripts/train_lora.py --version v0.2
.venv-gpu\Scripts\python backend/scripts/compare_models.py
```

The adapter (~37 MB, base model not copied), `examples.jsonl` (every training example with its
split) and `comparison.json` / `comparison.md` go to `data/models/lora/<version>-<time>/`.

For a blind rating of the two chat models, `blind_eval_answers.py` answers a question list with
both, writes a rating sheet with the answers shuffled into A/B (the key goes to a separate file), and
unblinds the scores once the sheet is filled in:

```bash
.venv-gpu\Scripts\python backend/scripts/blind_eval_answers.py generate --questions docs/eval/blind_questions.txt
.venv-gpu\Scripts\python backend/scripts/blind_eval_answers.py sheet
# fill in "ratings" (1-5 per criterion) in data/reports/blind_eval.sheet.json, without opening the key
.venv-gpu\Scripts\python backend/scripts/blind_eval_answers.py score --rater "your name"
```

### Model options in the chat

| In the model switch | What runs | Speed (RTX 3050 Ti laptop, 4 GB) | Set up with |
|---|---|---|---|
| **Base (Groq)** (default) | gpt-oss-120b on Groq's free tier | a few seconds | `GROQ_API_KEY` |
| **Local (LoRA)** | the 3B LoRA model (v0.7) on this machine | **~3 s** through llama.cpp; ~20 s through PyTorch 4-bit | llama.cpp: `LOCAL_MODEL_GGUF` + `LOCAL_LORA_GGUF`; PyTorch: `ADAPTER_PATH` (+ `LOCAL_LLM_DTYPE=4bit`) |
| **Local heavy (7B)** | the 7B LoRA model through llama.cpp, part on the GPU, part on the CPU | tens of seconds | `HEAVY_MODEL_GGUF` + `HEAVY_LORA_GGUF` |
| **Gemini …** / **OpenAI …** | hosted models, listed in `EXTRA_MODELS` | a few seconds | see "Other models" below |

**Local (LoRA) through llama.cpp vs PyTorch** (offline benchmark, the model writes its own rewrite;
`docs/reports/paper_analysis.md`, section 5):
- llama.cpp is 23x faster, with the same statute accuracy (100%) and safety (0% wrong punishments,
  0% civil questions as offences).
- It is more cautious on everyday scenarios: 45% answered with an acceptable section vs 60%, but
  only 15% cite a wrong section vs 32%; it says "I don't know" instead.
- The app uses llama.cpp when `LOCAL_MODEL_GGUF` is set. Remove it to use the PyTorch adapter.

**Notes:**
- A 4 GB GPU holds one local model at a time. Switching between Local and Local heavy unloads the
  other, so the first answer after a switch includes a few seconds of loading.
- `OFFLINE_MODE=true`: the local models also write the query rewrite themselves, so nothing is sent
  to Groq or any other service (fact-checking still uses the hosted model).
- Without it, the rewrite for a local model is written by the hosted model, which retrieves better.
- How the models were trained and converted, and the `.env` lines: `backend/scripts/colab/README.md`.

For GPU inference start the backend from the GPU environment:

```bash
cd backend
..\.venv-gpu\Scripts\python -m uvicorn app.main:app --port 8000
```

### Other models (Gemini, OpenAI, OpenRouter)

LegalGPT's pipeline (retrieval, citations, verification) can run on any model behind an
OpenAI-compatible chat API (`backend/app/core/openai_compat.py`). List them in `.env`, with keys:

```bash
EXTRA_MODELS=gemini:gemini-3.5-flash-lite,gemini:gemini-3.8-flash
GEMINI_API_KEY=...
```

They then appear in the chat's model switch. A model whose key is missing is listed in the
backend's startup notes and left out. Model names change: Google retires older models for new
keys (`gemini-2.5-flash` returns 404 for keys created in late 2026). The 404 message names the
replacement, and `GET https://generativelanguage.googleapis.com/v1beta/openai/models` (with the key
as a Bearer token) lists what the key can use. "High demand" (HTTP 503) replies are retried
three times, then reported.

| Provider | Key | Cost | How to get the key |
|---|---|---|---|
| `gemini` | `GEMINI_API_KEY` | free tier (rate-limited; Google may use free-tier prompts to improve its products) | [aistudio.google.com/apikey](https://aistudio.google.com/apikey): sign in, **Create API key** |
| `openai` | `OPENAI_API_KEY` | paid | platform.openai.com |
| `openrouter` | `OPENROUTER_API_KEY` | pay as you go; some `:free` models | openrouter.ai |

GitHub Models is not offered: GitHub retired it on 30 July 2026.

The same specs work in the benchmark runner (`benchmark/README.md`): `legalgpt+gemini:gemini-3.5-flash-lite`
is LegalGPT with Gemini as the LLM, `closed+gemini:gemini-3.5-flash-lite` is Gemini on its own. Run with
`--smoke` first: it answers 4 questions per system, so a wrong key or model name shows up before a
full run.

### Explanations in the Review tab

Press **w** (or click *Why this suggestion?*) on a sentence to see:
- **Word highlights:** Integrated Gradients over the input embeddings, 128 steps, [PAD]
  baseline. Green words push toward the role and red words push against it. *Why not…* explains
  any other role.
- **Context reliance:** the share of the evidence that came from the sentence's position and
  neighbours rather than the sentence itself. Above 50% is flagged, because neighbouring quoted
  precedent is a common cause of Ruling/Precedent mistakes.
- **Similar labelled sentences:** the nearest human-checked sentences by the classifier's own
  representation. Build the index once per checkpoint:
  `.venv-gpu\Scripts\python backend/scripts/build_explain_index.py`.

The API is `GET /api/review/cases/{case_id}/items/{sentence_id}/explain[?label=Ruling]`. A
faithfulness test (ERASER comprehensiveness and sufficiency; IG vs occlusion vs random) is in
`backend/scripts/eval_explanations.py` and reported in
[docs/reports/classifier_explanations.md](reports/classifier_explanations.md).

## Judgment citations and case analysis

**The judgment library.** Real judgments, split into passages and labelled by role (Facts, Law
Applied, Precedent, Argument, Ruling). It holds the annotated collection plus Supreme Court
judgments fetched from the CC-BY-4.0 dataset `labofsahil/Indian-Supreme-Court-Judgments`. The fetch
keeps only judgments that cite the IPC and writes a metadata file next to each.

```bash
python backend/scripts/fetch_sc_judgments.py --year 2023 --criminal 170 --library   # -> data/judgment_library/sc/
.venv-gpu\Scripts\python backend/scripts/build_judgment_index.py                    # labels roles, builds the index
```

The index is a separate Chroma collection (`judgments`), so the IPC index and its retrieval proof are
unchanged. Roles come from human labels where they exist and from the sentence-role ensemble
otherwise.

**In the chat**, after the IPC sources, up to `JUDGMENT_K` (2) passages are added: Ruling, Law
Applied or Precedent passages, one per judgment, above `JUDGMENT_MIN_SIMILARITY` (0.60). Rules:
- The answer may cite a judgment only for what that court said, and must name the case (prompt rule 7).
- Judgment chips show a scales icon and the case. The source window shows the court, the year and the passage's role.
- The retrieval proof covers the IPC sources only, and says so.
- Turn the feature off with `JUDGMENT_CITATIONS=false`.

**Case analysis** (its own tab; `POST /api/case/analyze`). Upload a `.txt` / `.pdf` (or paste text). The report:
1. The facts, each with a quote that is checked against the file. A fact whose quote is not found
   is shown but not used.
2. The IPC offences these facts may engage, with the section text. The section number must exist
   and its title must match the named offence, so a Motor Vehicles Act number is dropped.
3. An element-by-element check. The Prolog rule engine handles the 7 encoded sections; other
   sections are assessed by the model and labelled so.
4. Similar past judgments, and what the court held in each.
5. A written analysis citing only the numbered sources, with the chat's "Why this answer?".

It never says whether anyone is guilty. **Keep the file on this computer** runs every model step on
a local model; nothing is sent to Groq or Gemini, and nothing is stored either way.

## Neuro-symbolic verification (Module 5)

Needs [SWI-Prolog](https://www.swi-prolog.org/download/stable) on `PATH` (the backend still starts
without it; only `/api/verify` is disabled). Seven IPC sections are encoded as element checklists in
`backend/app/verification/rules/*.pl`: 279, 304A, 304B, 323, 337, 338, 379.

In the chat, pick a section under **Also verify facts against** (Legal mode) to check the facts in
your question after the answer. In the app's **Verify** tab, paste case facts and pick the charged section. The hosted model extracts
each element as true / false / unknown with a quote from the text (twice — only values both runs
agree on are kept), and Prolog returns **CONSISTENT** (every element established), **INCONSISTENT**
(an element contradicted) or **INSUFFICIENT** (an element not established), plus the other encoded
sections the facts fit. It checks extracted facts against encoded elements; it does not decide guilt.
Each result also says **what would change the verdict**: the facts still missing (as plain questions) for
INSUFFICIENT, and the contradicted elements for INCONSISTENT. This is computed exactly from the rules
(`app/verification/what_if.py`).

`POST /api/verify` with `{"case_text": "...", "cited_section": "304A"}`; `GET /api/verify/sections`.
To produce a sanity-check report on real annotated cases (written to `data/reports/`, git-ignored):

```bash
python backend/scripts/verify_case.py case20:304A case01:279 case31:323
```

To measure agreement with a labelled test set (15 fact patterns with expected verdicts in
`docs/eval/verification_cases.json`; about 8k Groq tokens per case):

```bash
python backend/scripts/eval_verification.py
```

## Verifiable results (ZKML)

Optional. Install EZKL and build the circuits once (about a minute; needs SWI-Prolog and the search
index; writes about 1.3 GB of proving keys to `data/zk/`, which is git-ignored):

```bash
pip install -r backend/requirements-zk.txt
python backend/scripts/zk_setup.py
```

The **Verify** tab then offers *Generate proof*, a zero-knowledge proof that the verdict follows
from the published rules, with the facts hidden. Each chat answer gets a **Signed** panel with
*Check signature* and *Prove the sources*. To check a downloaded proof offline from the published
files, run `python backend/scripts/zk_verify.py <file>`. Re-run `zk_setup.py` after changing the
corpus, the rules or the retrieval settings. Settings: `ZK_DIR` (default `data/zk`) and
`ATTEST_ANSWERS` (default `true`). Details, the API and what is *not* proved:
[docs/zk/README.md](zk/README.md).

## Run the backend

From the repo root, create a virtualenv and install dependencies (PyTorch is the CPU build):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r backend/requirements.txt
```

Then start the API from `backend/`:

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Check it: <http://localhost:8000/health> → `{"status":"ok"}`. Interactive docs: <http://localhost:8000/docs>.
On Windows, `--reload` can hang after a code change (the log says "Reloading..." but the old code keeps
serving); stop and restart uvicorn if that happens. `.env` changes always need a restart.

## Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The page calls `GET /health` on the backend and shows the result
(green `ok` when the backend is up). The backend URL comes from `VITE_API_BASE_URL`
(default `http://localhost:8000`).

## Run the tests

From the repo root (with the `.venv` from *Run the backend* active):

```bash
pytest -m "not integration"
```

This is what CI runs (`.github/workflows/ci.yml`, plus `npm run build` for the frontend).
`backend/tests/test_integration_e2e.py` drives the whole chain — question → retrieval over real IPC
text → answer → explanation → Prolog verification — with stub embedder and LLMs; it uses the real
SWI-Prolog engine when installed and a Python stand-in otherwise. To run the same chain live
(Groq, MiniLM, the real index, SWI-Prolog):

```bash
LEGALGPT_LIVE_E2E=1 pytest backend/tests/test_integration_e2e.py
```

(`pytest.ini` adds `backend/` to the import path, so `pytest tests` from inside `backend/` works too.)

Unit tests use a fake embedder, in-memory ChromaDB and a stub `LLMClient` — no network.
Live tests against Groq and the ingested index are marked `integration` and skip without
`GROQ_API_KEY`:

```bash
pytest -m integration
```

## API

`POST /api/chat` with `{"question": "...", "mode": "legal" | "general", "history": [...],
"model": "groq" | "adapter", "verify_section": "304A"}` (the last two optional) streams Server-Sent
Events: `meta` (with the answering model), `sources` (legal mode), `token`*, `explanation` (legal
mode, after the answer), `verification` (only with `verify_section`), optional `error`, `done`.
`GET /api/models` lists the models the toggle offers.
Legal mode answers only from retrieved IPC chunks and cites them as `[n]`; if nothing clears
`RETRIEVAL_MIN_SIMILARITY` it returns a scope-boundary refusal without calling the model.

The `explanation` event (shown in the app's **Why this answer?** panel) carries the evidence-match
score and band, citation checks (markers that point at no retrieved source), per-sentence
attribution to the supporting source excerpt, a support ratio, and plain-language warnings. The
evidence-match band says how closely the retrieved IPC text matches the question — **not** whether
the answer is correct. Thresholds: `EXPLAIN_BAND_HIGH`, `EXPLAIN_BAND_MEDIUM`,
`EXPLAIN_SUPPORT_THRESHOLD`.


## Repository layout

```text
backend/app/core/                 Module 0 — retrieval & chat platform
backend/app/annotation/           Module 1 — dataset pipeline
backend/app/labeling_assistant/   Module 2 — InLegalBERT-assisted annotation
backend/app/finetuning/           Module 3 — LoRA fine-tuning
backend/app/explainability/       Module 4 — explainability
backend/app/verification/         Module 5 — neuro-symbolic verification (Prolog)
backend/scripts/                  CLI entry points
backend/tests/                    pytest suite
frontend/                         React 19 + Vite + Tailwind
data/corpus/                      parsed IPC chunks (committed)
data/                             raw judgments, annotation store, ChromaDB, models (not committed)
```
