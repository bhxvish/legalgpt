# LegalGPT — Planning

Domain-adapted, retrieval-augmented legal question-answering for **Indian criminal law (IPC)**.

- **Online core:** React 19 + Tailwind chat UI, FastAPI backend, ChromaDB retrieval over
  hierarchically-chunked legal text, Llama-3-8B via Groq, strict source-grounded prompting.
- **Offline research pipeline:** manual rhetorical-role annotation → InLegalBERT-assisted
  labelling → LoRA fine-tuning of an open LLM on the resulting corpus.
- **New runtime services:** explainability (confidence, citation validation, sentence
  attribution) and neuro-symbolic verification (Prolog PoC over 5–10 IPC sections).

## Modules

| # | Module | Package | Purpose |
|---|--------|---------|---------|
| 0 | Core Platform | `backend/app/core/` | Parser, embeddings, ChromaDB store, retriever, prompt builder, `LLMClient` (Groq, later Adapter), SSE chat router |
| 1 | Dataset Pipeline | `backend/app/annotation/` | Judgment collection, sentence segmentation, 5-label rhetorical-role scheme, versioned `AnnotationStore`, Cohen's kappa |
| 2 | BERT-Assisted Annotation | `backend/app/labeling_assistant/` | InLegalBERT classifier, confidence-based review selection, human review queue, evaluation |
| 3 | LoRA Fine-Tuning | `backend/app/finetuning/` | Instruction dataset from corpus, 4-bit LoRA training, `AdapterClient`, base-vs-tuned comparison |
| 4 | Explainability | `backend/app/explainability/` | Retrieval-relevance band, citation validation, sentence-level attribution, `explanation` SSE event |
| 5 | Neuro-Symbolic Verification | `backend/app/verification/` | LLM fact extraction → Prolog facts → verdict against hand-encoded IPC rules |

## Repository structure

```text
legalgpt/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app, mounts routers
│   │   ├── config.py               # env-driven settings (API keys, paths, thresholds)
│   │   ├── core/                   # Module 0 — reused RAG platform
│   │   │   ├── parser.py           # LegalDocumentParser
│   │   │   ├── retriever.py        # Retriever, QueryExpander
│   │   │   ├── vector_store.py     # VectorStore (ChromaDB wrapper)
│   │   │   ├── embeddings.py       # EmbeddingService
│   │   │   ├── prompt_builder.py   # PromptBuilder
│   │   │   ├── llm_client.py       # LLMClient interface, GroqClient, AdapterClient
│   │   │   └── chat_router.py      # ChatRouter (FastAPI router, SSE streaming)
│   │   ├── annotation/             # Module 1 — dataset pipeline
│   │   │   ├── collector.py        # JudgmentCollector, JudgmentRecord
│   │   │   ├── segmenter.py        # SentenceSegmenter
│   │   │   ├── label_scheme.py     # LabelScheme
│   │   │   ├── store.py            # AnnotationStore
│   │   │   └── agreement.py        # AgreementCalculator (Cohen's kappa)
│   │   ├── labeling_assistant/     # Module 2 — BERT-assisted annotation
│   │   │   ├── dataset.py          # RRLDataset
│   │   │   ├── classifier.py       # RRLClassifier (InLegalBERT)
│   │   │   ├── review_selector.py  # ReviewSelector
│   │   │   ├── review_queue.py     # HumanReviewQueue
│   │   │   └── evaluator.py        # ClassifierEvaluator, Metrics
│   │   ├── finetuning/             # Module 3 — LoRA pipeline
│   │   │   ├── instruction_builder.py
│   │   │   ├── lora_trainer.py     # LoRATrainer, TrainConfig
│   │   │   └── model_comparator.py # ModelComparator
│   │   ├── explainability/         # Module 4
│   │   │   ├── confidence_scorer.py
│   │   │   ├── citation_validator.py
│   │   │   ├── attribution_analyzer.py
│   │   │   └── explanation_builder.py
│   │   └── verification/           # Module 5 — neuro-symbolic
│   │       ├── fact_schema.py
│   │       ├── fact_extractor.py
│   │       ├── predicate_mapper.py
│   │       ├── prolog_engine.py    # pyswip wrapper
│   │       ├── verification_service.py
│   │       └── rules/              # .pl files, 5-10 IPC sections
│   ├── scripts/                    # one-off CLI entry points per phase
│   ├── tests/                      # pytest, mirrors app/ structure
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── components/             # ChatWindow, SourceModal, WhyPanel, VerifyPanel
│       └── ...
├── data/
│   ├── raw_judgments/              # collected Indian Kanoon text
│   └── annotation_store/           # versioned JSONL + manifest.json
├── docs/                           # annotation_guideline.md (Phase 2)
├── docker-compose.yml
├── PLANNING.md
└── README.md
```

> Note: the build plan's tree omits `verification_service.py` and `docs/`, but Phases 2 and 6
> require them, so they are listed here.

## Phase checklist

- [x] **Phase 0** — Scaffolding & environment setup
- [x] **Phase 1** — Module 0: Core retrieval & chat platform
- [ ] **Phase 2** — Module 1: Domain-specific dataset pipeline
- [ ] **Phase 3** — Module 2: InLegalBERT-assisted annotation
- [ ] **Phase 4** — Module 3: LoRA fine-tuning pipeline
- [ ] **Phase 5** — Module 4: Explainability
- [ ] **Phase 6** — Module 5: Neuro-symbolic verification
- [ ] **Phase 7** — Integration, testing & deployment

## Conventions (every phase)

- Python 3.11, type hints everywhere, pydantic models for API bodies, pytest alongside new code.
- Secrets only via env vars read in `backend/app/config.py`; every variable documented in `.env.example`.
- Dependency injection + stub `LLMClient` for unit tests; live Groq/HF calls only in clearly
  marked, skippable integration tests.
- Keep the LLD's class boundaries; flag any redesign explicitly.
- Scope: IPC criminal law only. Prolog KB: 5–10 sections. Fine-tuning: 4-bit + LoRA only.
- ZKML is out of scope for all phases (documented future work).
- No fabricated metrics — anything needing a real run is computed live.
- One commit per phase (or logical sub-step), e.g. `feat(module-1): add JudgmentCollector`.
- Each phase ends with a pass/fail report against its acceptance checklist.

## Decisions log

- **Repo root** is `legaltech/` itself (the plan's `legalgpt/` is a placeholder name).
- **PyTorch is CPU-only** in `backend/requirements.txt` through Phases 0–3. Revisit in
  Phase 4: either a separate `requirements-gpu.txt` with the CUDA wheel for the fine-tuning
  environment, or install the CUDA build directly on Colab without touching base requirements.
- **Local GPU** is an RTX 3050 Ti (4 GB), which is not enough for 7B LoRA training; expect
  Phase 4 training to run on Colab-class compute.
- **Not tracked in git:** `.env`, `data/annotation_store/` contents, `chroma_db/`, model
  checkpoints (`*.ckpt`, `*.safetensors`). Folder structure is kept via `.gitkeep`.
- **Single `.env` at the repo root**, read by `config.py` (via an absolute path) and by Vite
  (`envDir: '..'`). Empty values fall back to defaults (`env_ignore_empty`).
- **Top-level Python deps are pinned** to the versions verified in Phase 0 (torch 2.14.1+cpu,
  transformers 5.18, peft 0.21, trl 1.14, chromadb 1.5.9). Bump deliberately, not implicitly.
- **`httpx2` instead of `httpx`** for the FastAPI `TestClient` (Starlette deprecated plain httpx).
- **pyswip 0.3.3 works with SWI-Prolog 10.0.2** on the dev machine (assert + query smoke-tested).
- **Frontend calls the backend directly** at `VITE_API_BASE_URL` (CORS), no Vite proxy.

### Phase 1 (Module 0)

- **Generation model is `openai/gpt-oss-120b` on Groq, not Llama-3-8B.** Groq no longer serves any
  Llama-3 model for this key (available: gpt-oss-20b/120b, qwen3.8-27b); user chose "the best".
  Configurable via `GROQ_MODEL`. Its reasoning tokens arrive separately and are not streamed.
  Phase 4 impact: the base-vs-tuned comparison will be gpt-oss (Groq) vs the LoRA-tuned local model,
  not the same base model — note this when reporting ModelComparator results.
- **Corpus:** official India Code IPC PDF (not committed). Parsed with pdfminer.six (MIT) instead of
  pypdf (which split words, e.g. "deat h"); PyMuPDF rejected for its AGPL licence. Extraction uses
  layout: lines re-sorted top-to-bottom, superscript footnote numbers dropped by size, footnotes
  dropped as the bottom small-font run starting at a "1. Subs./Ins. by ..." line (illustrations
  share the footnote font size, so size alone is not enough).
- **The PDF's Arrangement of Sections is the source of truth** for which section numbers exist:
  it bounds where the body starts, admits headings with irregular formatting ("17 “Government”.—"),
  and keeps state-inserted sections (e.g. Chhattisgarh's 376F) inside their State Amendments node.
  All 574 listed sections are parsed; a test asserts this when the PDF is present.
- **State amendments are separate nodes** (`... > Section 304A > State Amendments`) so state-only law
  is never chunked together with the central section text. Repealed sections get small nodes.
- **Retrieval floor 0.55** (cosine, MiniLM) calibrated with `scripts/calibrate_retrieval.py`:
  worst in-scope top-1 0.646, best out-of-scope top-1 0.533 (anticipatory bail). Hit@6 16/16,
  hit@1 10/16. The same floor filters supporting chunks, hence the low end of the separating range.
- **Retriever additions (beyond the LLD):** explicit section references ("u/s 304A") also run a
  filtered lookup that bypasses the floor; follow-ups are searched both alone and with the previous
  user question prepended (merged), so "within how many years?" after a dowry-death question works.
- **Citation markers are canonicalized in the stream** to `[n]` (gpt-oss emits `【1】` and
  `[1†L2-L4]`), so the UI and Phase 5's CitationValidator see one format.

## Known limitations

_To be filled in during Phase 7._
