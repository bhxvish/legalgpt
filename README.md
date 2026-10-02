# LegalGPT

Retrieval-augmented legal question answering for **Indian criminal law (IPC)**, with
an offline annotation + fine-tuning pipeline, an explainability layer, and a Prolog-based
verification service. See [PLANNING.md](PLANNING.md) for modules, phases and decisions.

> Status: **Phase 2** — retrieval-augmented chat over the IPC bare act (Module 0) and the
> rhetorical-role annotation pipeline (Module 1).

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

Download the official bare act PDF from India Code
(<https://www.indiacode.nic.in/bitstream/123456789/11091/1/the_indian_penal_code,_1860.pdf>) into
`data/raw_corpus/` (git-ignored; any filename), then from the repo root:

```bash
python backend/scripts/ingest_corpus.py data/raw_corpus/<file>.pdf --reset
```

The script checks the parse against the PDF's own *Arrangement of Sections* and prints how many
listed sections were found in the body (all 574 for the India Code PDF). `--dry-run --show 304A,302`
prints chunks without embedding; `--dump chunks.jsonl` writes every chunk for inspection. The index
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

To serve the tuned model in the chat app, set `LLM_BACKEND=adapter` in `.env` (optionally
`ADAPTER_PATH`) and start the backend from the GPU environment:

```bash
cd backend
..\.venv-gpu\Scripts\python -m uvicorn app.main:app --port 8000
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

From the repo root:

```bash
pytest backend/tests
```

(`pytest.ini` adds `backend/` to the import path, so `pytest tests` from inside `backend/` works too.)

Unit tests use a fake embedder, in-memory ChromaDB and a stub `LLMClient` — no network.
Live tests against Groq and the ingested index are marked `integration` and skip without
`GROQ_API_KEY`:

```bash
pytest -m integration
```

## API

`POST /api/chat` with `{"question": "...", "mode": "legal" | "general", "history": [...]}`
streams Server-Sent Events: `meta`, `sources` (legal mode), `token`*, optional `error`, `done`.
Legal mode answers only from retrieved IPC chunks and cites them as `[n]`; if nothing clears
`RETRIEVAL_MIN_SIMILARITY` it returns a scope-boundary refusal without calling the model.

## Docker (local dev)

```bash
docker compose up --build
```

Backend on `:8000`, frontend on `:5173`, both with source mounted for hot reload. `./data`
is mounted into the backend container.

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
data/                             raw judgments, annotation store, ChromaDB (not committed)
```
