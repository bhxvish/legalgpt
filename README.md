# LegalGPT

Retrieval-augmented legal question answering for **Indian criminal law (IPC)**, with
an offline annotation + fine-tuning pipeline, an explainability layer, and a Prolog-based
verification service. See [PLANNING.md](PLANNING.md) for modules, phases and decisions.

> Status: **Phase 0** — scaffolding only. The backend serves `/health`; the frontend shows it.

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
