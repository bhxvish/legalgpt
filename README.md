# LegalGPT

Retrieval-augmented legal question answering for **Indian criminal law (IPC)**, with
an offline annotation + fine-tuning pipeline, an explainability layer, and a Prolog-based
verification service. See [PLANNING.md](PLANNING.md) for modules, phases and decisions.

> Status: **Phase 1** — retrieval-augmented chat over the IPC bare act (Module 0).

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
