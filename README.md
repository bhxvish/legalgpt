# LegalGPT

Retrieval-augmented legal question answering for **Indian criminal law (IPC)**, with
an offline annotation + fine-tuning pipeline, an explainability layer, and a Prolog-based
verification service. See [PLANNING.md](PLANNING.md) for modules, phases and decisions.

> Status: **all 8 phases complete** (Phase 0–7). Not legal advice: answers are grounded in the
> IPC bare act only, and the IPC was replaced by the Bharatiya Nyaya Sanhita on 1 July 2024.

## Quick start (Docker)

From a clean checkout, with [Docker](https://docs.docker.com/get-docker/) running:

```bash
cp .env.example .env            # then set GROQ_API_KEY in .env
docker compose up --build
```

- Frontend: <http://localhost:5173> — Chat (Legal / General mode, **Base (Groq)** vs **Tuned (LoRA)**
  toggle, "also verify facts against" a section), Verify, Annotate, Review.
- Backend: <http://localhost:8000/docs>.

On the first start the backend downloads the embedding model and embeds the committed IPC chunks
(`data/corpus/ipc_chunks.jsonl`) into the `chroma_db` volume — legal-mode questions work about a
minute after `/health` turns green. SWI-Prolog is inside the backend image, so verification works
without a separate service. ChromaDB, the annotation store, collected judgments and the Hugging Face
cache live in named Docker volumes (`docker compose down -v` deletes them). The **Tuned (LoRA)**
toggle is only enabled once an adapter exists under `data/models/lora/` (see Module 3); on CPU it
takes about a minute per answer.

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
| 0 Retrieval & chat | IPC bare act, hierarchical chunks, ChromaDB + MiniLM, calibrated refusal floor, SSE chat with `[n]` citations, Legal/General modes | BNS/BNSS and IPC↔BNS mapping; judgments as a retrieval source; re-ranker |
| 1 Dataset | Collector with screening, 5-role scheme + guideline, append-only store, kappa tooling, frozen versions with hashes | Double annotation to actually measure agreement |
| 2 Assisted labelling | InLegalBERT classifier (CV macro-F1 0.664 ± 0.031), confidence + audit review queue, review UI | Larger corpus, active learning |
| 3 Fine-tuning | 4-bit LoRA on Qwen2.5-1.5B, instruction builder, base-vs-tuned comparison, in-app toggle | Human-written answer targets (the adapter copies sources and over-refuses) |
| 4 Explainability | Evidence-match band, citation checks, sentence attribution, warnings on every legal answer | Detecting subtle legal errors, not just unsupported sentences |
| 5 Verification | Prolog element checklists for 7 sections (279, 304A, 304B, 323, 337, 338, 379), quote-checked two-run fact extraction | More sections, exceptions and Chapter IV defences |
| — | — | **ZKML proofs of inference: not implemented** — proving a transformer forward pass in zero knowledge is still orders of magnitude too slow for a 1.5B+ model on this hardware, and it would prove which model ran, not that the answer is legally right. |

Known limitations are consolidated in [PLANNING.md](PLANNING.md#known-limitations).

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

The chat app offers the newest adapter next to the hosted model (the **Base / Tuned** toggle;
`LLM_BACKEND=adapter` makes it the default). For GPU inference start the backend from the GPU
environment:

```bash
cd backend
..\.venv-gpu\Scripts\python -m uvicorn app.main:app --port 8000
```

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

`POST /api/verify` with `{"case_text": "...", "cited_section": "304A"}`; `GET /api/verify/sections`.
To produce a sanity-check report on real annotated cases (written to `data/reports/`, git-ignored):

```bash
python backend/scripts/verify_case.py case20:304A case01:279 case31:323
```

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
