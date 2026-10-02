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
- [x] **Phase 2** — Module 1: Domain-specific dataset pipeline
- [x] **Phase 3** — Module 2: InLegalBERT-assisted annotation
- [x] **Phase 4** — Module 3: LoRA fine-tuning pipeline
- [x] **Phase 5** — Module 4: Explainability
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

### Phase 2 (Module 1)

- **Six labels, not five.** The LLD's Facts, Law Applied, Precedent, Argument, Ruling plus
  **None** for headers, cause titles and boilerplate (user decision). Phase 3 can train with
  None or filter it out. The guideline is generated from `LabelScheme`; a test enforces sync.
- **`LabeledSentence` gains `annotator` and `created_at`** (beyond the LLD's fields), needed for
  double annotation and Cohen's kappa. The store is an append-only log; the latest decision per
  sentence is gold, so adjudication = re-labelling. The UI shows annotators only their own
  labels so double annotation stays independent.
- **Segmentation is pinned per case** (`annotation_store/segments/<case>.json`) on first open, so
  later segmenter changes cannot shift sentence ids under existing labels.
- **Frozen versions are immutable** (`versions/<v>/labels.jsonl` + `manifest.json` with per-case
  and overall SHA-256 and a seeded by-case split). The digest covers content (text, label, source,
  reviewed), not who labelled or when, so identical data gives an identical digest.
- **Collector heuristics** (no extra dependencies): English = ≥90% Latin letters and ≥15% English
  function words; criminal = cites an IPC section, or ≥3 criminal-procedure terms; duplicate =
  same SHA-256 of whitespace/case-normalized text. Judgment texts are git-ignored.
- **Sentence splitting** distinguishes abbreviations that lead into something ("S.", "v.",
  "Smt.", "PW.") from acronyms that can end a sentence ("I.P.C.", "Cr.P.C."); initials of up to
  two letters ("M.K.") never split. Tuned against the team's seed segmentation: boundary F1
  **0.956** (P 0.967 / R 0.946) over 4,283 sentences; a test keeps it ≥ 0.94 when the seed
  corpus is present. Colon-introduced quotations ("reads as under:-") end a sentence, as the
  team segmented them.
- **Seed corpus imported, not hand-built in our UI.** The team supplied `legaltech_dataset.xlsx`
  (31 criminal judgments, 4,283 labelled sentences, labels FACT/LAW/PRECEDENT/ARGUMENT/RULING).
  `scripts/import_annotations.py` normalizes doc ids ("case 11" / "case11"), drops a repeated
  header row, pins the sheet's own sentences as each case's segmentation, screens the rebuilt
  text through JudgmentCollector (all 31 eligible), and records empty annotator cells as
  `team-sheet`. Frozen as **v0.1** (sha256 `44659d79…`, split 25/3/3 cases, seed 13).
- **The team's tie-break order** (Ruling > Argument > Precedent > Law Applied > Facts) and their
  LAW/FACT conventions (uncited "well settled" principles = Law Applied; witness testimony =
  Facts) were added to LabelScheme and the guideline, since the gold data follows them.
- **Acceptance check on real text was run in a scratch store**, not the real one: labelling an
  imported case in the real store would overwrite the team's gold labels (latest decision wins).
  case04's rebuilt text went through collect → our segmenter (26 sentences, same count as the
  team) → keyboard labelling of every sentence; κ vs the team's labels was 0.848 (approximate:
  2 of 26 sentence boundaries differ).

### Phase 3 (Module 2)

- **None dropped; five labels** everywhere (user decision), matching the seed corpus and the LLD.
  The frozen v0.1 manifest still lists six labels in its scheme snapshot; versions are immutable
  and v0.1's data never used None.
- **Seed size:** 31 cases (25 train / 3 val / 3 test), below the plan's 50+; noted as a limitation.
- **Compute:** CPU-only (Phase 0 decision). InLegalBERT (MIT, BERT-base, 110M params) trains with
  embeddings + the bottom 8 of 12 encoder layers frozen (28.9M trainable), class-weighted loss,
  3 epochs, ~4 min/epoch with length-grouped batches.
- **"Sortish" batching, not global length sorting.** Sorting the whole corpus by length made
  batches label-homogeneous wherever length correlates with the label, which a unit test exposed
  (a tiny model learned nothing). Batches are now sorted only within shuffled pools of 50 batches.
- **Held-out test (3 cases, 296 sentences), model `v0.1-20261002-191352`:** accuracy 0.669,
  macro-F1 0.642 (validation macro-F1 0.751). Per-class F1: Facts 0.79, Argument 0.65,
  Law Applied 0.64, Ruling 0.60, Precedent 0.54. Precedent↔Ruling and Facts→Ruling are the main
  confusions. With 3 test cases these numbers are noisy; see cross-validation below.
- **5-fold cross-validation grouped by case** (same config, v0.1, `data/models/rrl_cv/`):
  accuracy **0.712 ± 0.041**, macro-F1 **0.664 ± 0.031** (folds 0.606–0.693). The single held-out
  split (0.642) sits inside this range; ~0.66 macro-F1 is the honest estimate for this seed size.
- **New real judgments:** the user could not supply files, so `scripts/fetch_sc_judgments.py`
  pulls individual Supreme Court PDFs from the CC-BY-4.0 HF dataset
  `labofsahil/Indian-Supreme-Court-Judgments` via HTTP range requests into its yearly tars
  (~5 MB instead of 400 MB). It cuts the SCR editorial headnote (the reporter's summary, not the
  court's text), page numbers and running headers, and skips PDFs with a garbled text layer
  (Ravi Mandal v. State of Uttarakhand was skipped). Four 2023 criminal appeals were collected.
- **Segmenter fixed for PDF-derived text** after the first review pass showed fragments: lines
  wrapped at ~70 characters and page breaks mid-sentence were split. Headings are now recognised
  by shape (all caps, bracketed coram), not length, and clause markers ("(a)") join the next line.
  The four cases went from 1,725 fragments to 1,270 sentences; seed boundary F1 stays ≥ 0.94.
- **End-to-end on a real new case:** State of U.P. v. Sonu Kushwaha (2023 INSC 603): 79 sentences,
  30 flagged (24 low-confidence + 6 audit), reviewed in the Review tab by reviewer **"claude"**
  (17 corrections; 1 of 6 audits wrong = 17% < 20% limit), signed off and promoted as
  `bert_assisted` (30 reviewed, 49 model-accepted, annotator `model:<checkpoint>`). These labels
  are Claude's, not the team's; a teammate can re-review them (latest decision wins). The other
  three fetched cases are queued for the team.
- **Review UI race fixed:** fast keystrokes acted on a stale cursor and skipped sentences; key
  handling now reads synchronously updated refs. Sign-off stays disabled until every flagged
  sentence is checked, so a skip could never be promoted.

### Phase 4 (Module 3)

- **No existing fine-tuning notebook** in the repo to adapt; the pipeline was written fresh.
- **Compute and model (user decisions):** local RTX 3050 Ti (4 GB) and **Qwen2.5-1.5B-Instruct**
  (Apache-2.0, not gated). CUDA PyTorch (2.14.1+cu130) lives in a separate `.venv-gpu` from
  `backend/requirements-gpu.txt`; the base install stays CPU-only. The disk was full (2 GB free):
  the user approved `pip cache purge` (8.3 GB); the GPU env is installed with `--no-cache-dir`.
- **Training data (v0.2, 32 cases):** 124 instruction examples — facts→law (32),
  facts+arguments→ruling (31), grounded answers in the exact inference prompt format with [n]
  citations (32), grounded refusals pairing unrelated cases (29); split by case 100/12/12
  (26/3/3 cases), verified disjoint. Small: this demonstrates the pipeline more than it teaches law.
- **Training run `v0.2-20261002-203807`:** 4-bit NF4 QLoRA, r=16, α=32, all attention + MLP
  projections, lr 2e-4, 3 epochs (39 optimizer steps), loss on answers only, 12 min, peak GPU 3.06
  GB. Validation loss **1.292 (untuned) → 1.062 / 1.047 / 1.045** after epochs 1–3. Adapter 37 MB,
  saved without the base model.
- **Comparison** (6 held-out questions from the 3 test cases, same retrieved evidence for every
  model; report in `docs/reports/lora_v0.2_comparison.md`):

  | | Groq gpt-oss-120b | Qwen-1.5B base | Qwen-1.5B + LoRA |
  |---|---|---|---|
  | mean citation accuracy | 1.00 | 0.83 | 1.00 |
  | answers using `[n]` markers | 5/5 | 0/6 | 5/5 |
  | ungrounded sections | 0 | 1 | 0 |
  | refusals | 1 | 0 | 1 |
  | mean seconds | 1.1 | 8.7 | 11.9 |

  What fine-tuning changed: the base model never used the required citation format and once
  invented a section ("378(1) … obscene material"); the tuned model cites in the required format
  and stayed grounded. What it broke: it learned to **copy sources** (its targets were source
  sentences concatenated with markers) — degenerate listings, a state amendment pasted as the
  answer — and to **over-refuse** (refusals were 23% of examples): with `LLM_BACKEND=adapter` it
  answered "I don't know" to the punishment for death by negligence although §304A was source
  [1]. Better targets need human-written answers, not concatenated sources.
- **citation_accuracy measures grounding, not relevance or correctness:** citing a retrieved but
  irrelevant chunk counts as grounded (the tuned model cited §211/§292 for a CrPC appeal).
  `answers_using_markers` was added after the first run because base answers scored 1.0 without
  any markers.
- **Retriever fix found by the comparison:** "Section 378(1) of the Code of Criminal Procedure"
  had pulled IPC §378 (theft) through the explicit-section lookup; section numbers followed by
  another enactment (CrPC, BNS, "… Act") are no longer treated as IPC references.
- **Serving:** `LLM_BACKEND=adapter` switches ChatRouter to AdapterClient with no code change
  (verified end to end through `/api/chat`); run the backend from `.venv-gpu` for GPU inference.

### Phase 5 (Module 4)

- **Evidence-centric, no SHAP/LIME:** everything comes from retrieval similarities, the answer
  text and the source text. `explanation` is emitted after the last token (never delays the
  answer); a failure there is logged and skipped rather than turning a delivered answer into an
  error. Refusals get an explanation too: the retriever now reports the best candidate similarity
  even when nothing cleared the floor, so every legal response has a non-null relevance and band.
- **Bands calibrated, not the spec's example 0.75/0.5:** High ≥ 0.70, Medium ≥ 0.60 (answerable
  questions scored 0.65–0.86, median 0.73; floor 0.55). With 0.75 most good matches would read
  "Medium". Docstring, `BAND_NOTE` and the panel all state the band is evidence match, not answer
  correctness.
- **Attribution against source sentences, not chunks** (MiniLM truncates at ~256 tokens; sentence
  level also yields the supporting excerpt). Support threshold **0.55**, measured on real answers:
  90% of sentences scored against their own sources clear it vs 10% against sources about unrelated
  offences. Sources about a *neighbouring* offence are not separable this way, so attribution
  catches off-topic/unsupported sentences, not subtle legal errors.
- **When the cited source supports a sentence, it is credited** even if another source scores a
  little higher (e.g. s.304 vs the cited s.304A); a sentence whose citation does not support it is
  flagged separately.
- **Found while testing on real answers:** a correct "I don't know" was being attributed and
  flagged as unsupported (fixed: the not-covered sentence is excluded); and the tuned model's
  over-refusals went unflagged (fixed: "the model said the sources do not cover this, but the
  evidence match is high/medium" — it fires on the adapter's dowry-death and theft answers).
- **Citation normalisation moved to `app/core/citations.py`** (chat stream, validator and
  comparator share it; avoids a chat_router ↔ explainability import cycle).

## Known limitations

_Collected as they are found; consolidated in Phase 7._

- **Fact-pattern retrieval** (also seen in Phase 5): "The accused drove a lorry rashly and hit a
  scooter…" scored 0.54 and was refused; short questions about the same offence retrieve §279/§337
  fine.

- **Retrieval floor vs long fact patterns:** questions written as fact narratives pull generic
  chunks (IPC §1/§2, state-amendment boilerplate) at 0.6–0.7 similarity, above the 0.55 floor,
  because the Phase 1 calibration set only had short questions. A CrPC appeal therefore reaches the
  model with irrelevant IPC sources instead of being refused. Needs a fact-pattern calibration set
  and probably down-weighting of definitional / state-amendment chunks.
- **Tuned model quality:** see Phase 4 — copies sources and over-refuses; not a drop-in
  replacement for the hosted model.

- **Seed corpus (v0.1):** 31 cases, single annotation pass. No double annotation yet, so
  inter-annotator agreement (Cohen's kappa) on real data is **unmeasured**; the guideline is not yet
  validated the way the LLD intends. The seed never uses **None**: headers and boilerplate were
  forced into the five roles, so new UI annotations that use None will differ in distribution.
- **Label skew:** Facts 38%, Precedent 23%, Ruling 21%, Argument 10%, Law Applied 9% — Phase 3 should
  report per-class metrics, not only accuracy. Precedent-heavy cases (case01, case08, case31) partly
  consist of long quotations from earlier judgments.
- **Sheet gaps:** case18 and case21 skip 2 and 5 sentence ids, so their rebuilt text is missing
  those sentences. Court and decision year are unknown/approximate for imported cases (the sheet
  has no cover page; year comes from the title).
- **Codes:** one seed case (case04) also cites the BNS/BNSS, which replaced the IPC in July 2024;
  the corpus and Prolog scope remain IPC-only.
