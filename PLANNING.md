# LegalGPT — Planning

Domain-adapted, retrieval-augmented legal question-answering for **Indian criminal law (IPC)**.

- **Online core:** React 19 + Tailwind chat UI, FastAPI backend, ChromaDB retrieval over
  hierarchically-chunked legal text, a hosted LLM via Groq (planned Llama-3-8B; gpt-oss-120b since
  Groq retired Llama-3, see Phase 1), strict source-grounded prompting.
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
│   │   │   ├── segmenter.py        # AssistedSegmenter (PDF-tuned; assisted path only)
│   │   │   ├── titles.py           # cause-title detection for the Review list
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
- [x] **Phase 6** — Module 5: Neuro-symbolic verification
- [x] **Phase 7** — Integration, testing & deployment

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
  *Moved out of Module 1 afterwards:* the Module 2 spec says the manual module must not change, so
  `annotation/segmenter.py`, `collector.py` and `AnnotateView.jsx` were restored to their Module 1
  state; the retuned segmenter is `labeling_assistant/segmenter.py` (AssistedSegmenter) and title
  detection `labeling_assistant/titles.py`, used only by the assisted path. Queued cases pin their
  segments, so an escalated case is annotated manually against the same sentences. The only
  remaining change to Module 1 is the five-label scheme (None dropped, a user decision).
- **Review queue corrupted case files under fast reviewing** (found re-checking the Module 2
  acceptance list): accepting 103 sentences with Enter held down sent parallel saves that
  interleaved in one shared temp file (corrupt JSON, 160 failed requests) and could drop each
  other's corrections. Fixed with one lock per queue folder and a unique temp file per save;
  regression test with 8 threads. The real queue files were checked and were intact.
- **End-to-end on a real new case:** State of U.P. v. Sonu Kushwaha (2023 INSC 603): 79 sentences,
  30 flagged (24 low-confidence + 6 audit), reviewed in the Review tab by reviewer **"claude"**
  (17 corrections; 1 of 6 audits wrong = 17% < 20% limit), signed off and promoted as
  `bert_assisted` (30 reviewed, 49 model-accepted, annotator `model:<checkpoint>`). These labels
  are Claude's, not the team's; a teammate can re-review them (latest decision wins). The other
  three fetched cases are queued for the team.
- **Review UI race fixed:** fast keystrokes acted on a stale cursor and skipped sentences; key
  handling now reads synchronously updated refs. Sign-off stays disabled until every flagged
  sentence is checked, so a skip could never be promoted.
- **Classifier v2 (2026-10-03): context + full fine-tuning + calibration**, to cut review load.
  Changes: each sentence is read with its neighbours and its position in the judgment (BERT's
  second segment); all 12 layers trained at 256 tokens for 5 epochs (bf16 on the RTX 3050 Ti,
  ~4 min per model); only human-checked labels trained/evaluated on (unchecked assisted labels kept
  as context only); temperature scaling fitted on validation cases so confidence tracks accuracy.
  Same 5-fold case-grouped CV procedure for both (each fold holds out 3 training cases for epoch
  choice and calibration), v0.1, 4,283 sentences:

  | | accuracy | macro-F1 | must review at bar 0.70 | skipped labels right |
  |---|---|---|---|---|
  | A: old setup (8 layers frozen, 128 tokens, 3 epochs) | 0.708 ± 0.033 | 0.659 ± 0.030 | 44.2% | 84.7% |
  | C: context + full fine-tune | **0.781 ± 0.038** | **0.744 ± 0.040** | **18.7%** | 83.8% |

  C beat A on every fold (macro-F1 +0.06 to +0.12). Single split (v0.1 test case09/10/22): A 0.686 /
  0.653, + full fine-tune 0.740 / 0.718, + context 0.767 / 0.753. The gain is at the default bar:
  for skipped labels to be ~89% right, both need ~57% reviewed (A at bar 0.8, C at 0.9) — C is right
  more often but its most confident tail is not more reliable. Results: `data/models/rrl_cv/
  v0.1-cv5-{A-baseline,C-context-full}-20261003.json`.
- **Production checkpoint `v0.2-20261003-135812`** (C, trained on v0.2's human-checked labels; T=2.83)
  is now the newest, so `run_assisted_labeling.py` uses it. Its own test split (case10/17/22)
  is weak — 0.594 / 0.629, vs 0.620 / 0.635 for A trained identically — because of **case17**
  (Abdul Subhan, 2006): the team labelled the court's own evaluation of evidence as Ruling, and both
  models call it Precedent (A 50% accuracy on the case, C 37%, with C's errors at 0.88–0.95
  confidence). On case10/case22, unseen by both, C is better (73% vs 65%, 84% vs 80%). Kept C on the
  strength of the CV; its failure mode is a whole case wrong with confidence, which the audit sample
  is there to catch (case17 would be escalated) — keep audit_rate ≥ 10%. Worth checking with the
  team how court reasoning is labelled across cases (Ruling vs Precedent).

- **Upload in the Review tab (2026-10-08):** JudgmentIntake runs the assisted-labelling pipeline
  for an uploaded .txt / text-based .pdf: screening (JudgmentCollector), AssistedSegmenter, the newest
  classifier, ReviewSelector, HumanReviewQueue. One background worker processes uploads in order
  (the classifier is CPU-heavy and not thread-safe); the panel polls job status. Every sentence gets a
  suggested role, but nothing is promoted without a reviewer's sign-off, as before. Re-uploading text
  collected earlier but never queued reuses that case; text already queued or labelled is refused.
  Measured on a 12-page PDF: 205 sentences annotated and queued in 49 s on CPU, model load included.
  PDF clean-up (ligatures, page numbers, running headers) moved from fetch_sc_judgments.py into
  labeling_assistant/pdf_text.py so both use it. Upload jobs are kept in memory: after a restart the
  status list is empty, but the queued cases remain.

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

### Phase 6 (Module 5)

- **Seven sections, element checklists:** 279, 304A, 304B, 323, 337, 338, 379 — the offences that
  dominate the corpus (road accidents, hurt, theft, dowry death). `element(Section, Predicate,
  Required, Description)` is the single source; named predicates (`negligent_death/1`, `theft/1`, …)
  are readable aliases. One shared schema of 19 predicates; a test asserts schema and rules match.
- **Explicit `unknown`, no negation-as-failure:** the spec's `not(intent_to_kill(Case))` would read
  "the facts do not say" as "false" and satisfy s.304A's "no intention" element by default. Facts
  are `fact(Case, Predicate, true|false|unknown)`; an element is satisfied / violated / missing,
  and the verdict is INCONSISTENT (any violation) > INSUFFICIENT (any missing) > CONSISTENT.
- **Bugs caught by the hand-verified rule tests:** (1) section facts in a second .pl file replaced the
  first file's (static predicates) — fixed with `multifile`; (2) `verdict(C, S, consistent)` with the
  verdict already bound skipped the cut-guarded violated/missing clauses and succeeded for any
  section with no facts (`theft(nobody)` was true) — the verdict is now computed, then unified.
- **Extraction guards:** evidence quotes must appear in the text (elisions "…" allowed in order),
  otherwise the value is downgraded to unknown; one repair round on invalid JSON / schema errors;
  extraction runs twice and keeps only values both runs agree on — gpt-oss varied between runs even
  at temperature 0 (case01 flipped from three established elements to none). Rash vs negligent is
  not treated as a disagreement (every encoded section that needs one accepts either).
- **Engine safety:** one process-wide lock around pyswip (not thread-safe; FastAPI uses a thread
  pool); facts asserted in `session()` and always retracted in `finally` (tested with a verdict that
  raises, and for leakage between consecutive verifications); only schema predicates/values and a
  validated case id reach Prolog.
- **Real cases (report: `data/reports/verification_examples.md`, for a team member to sanity-check):**
  case20 (G. Manickam) s.279 **CONSISTENT** with a quote per element (convicted by both lower
  courts); s.304A **INSUFFICIENT** — the facts never state absence of intention/knowledge;
  case01 (Vijay Kumar, acquitted) s.279 INSUFFICIENT — no eyewitness says the accused drove, which
  is the reason the court acquitted; case07 (Ram Suresh Tiwari) s.304A INCONSISTENT because the
  extractor classed an "accidental fire from his country-made pistol" as accidental rather than
  negligent — a legal judgment worth a human look.

### Phase 7 (integration, testing & deployment)

- **One chat request, whole chain:** `POST /api/chat` takes `model` (`groq` | `adapter`, listed by
  `GET /api/models`) and `verify_section`. Order of events: `meta`, `sources`, `token`*,
  `explanation` (every legal response, refusals included), then `verification` only when asked.
  Verification failures (no SWI-Prolog, extraction error, rate limit) arrive as
  `{"error": ...}` in that event and never turn a delivered answer into an error.
- **Base-vs-tuned toggle, not a restart:** both LLMClients are built at startup (the adapter
  only if one exists under `LORA_DIR`; it loads lazily on first use). `LLM_BACKEND` now only picks
  the default. Verified in the browser: the same question answered by `openai/gpt-oss-120b`, then
  by `Qwen/Qwen2.5-1.5B-Instruct+lora:v0.2-20261002-203807`, each tagged with its model.
- **Fact extraction always uses Groq,** whichever model answers: it needs valid JSON for all 19
  predicates, which the 1.5B adapter does not produce reliably.
- **Clean-checkout index:** the IPC PDF is git-ignored and India Code was unreachable from here, so
  the parsed chunks are committed (`data/corpus/ipc_chunks.jsonl`, 685 chunks, 0.6 MB; bare-act text
  is public domain under Copyright Act s.52(1)(q)) and byte-identical to a fresh parse of the PDF.
  On startup the backend embeds them into an empty collection (~45 s on CPU) — no manual ingest step.
- **Docker:** named volumes for ChromaDB, AnnotationStore, raw judgments and the Hugging Face cache;
  `data/corpus` and `data/models` are read-only bind mounts. **No Prolog sidecar** (deviation from
  the plan's "optional"): pyswip embeds SWI-Prolog in-process, the backend image already installs
  `swi-prolog-nox`, and a sidecar would need a network protocol around the engine for no benefit.
  Source bind mounts and `--reload` were dropped: the images are self-contained, development uses
  the local venv.
- **Race found by the demo:** the startup index check and the first chat request opened ChromaDB's
  PersistentClient at the same moment ("Could not connect to tenant default_tenant"; the first demo
  question failed). `VectorStore` now opens it under a lock; a test with 8 threads fails without it.
- **`docker compose up` from a clean clone: verified** (after Docker Desktop's WSL service hung once
  and needed killing as administrator — a host problem). Fresh `git clone` + `.env` + `docker compose
  up --build`, with no pre-existing volumes: backend healthy, "index ready: 685 chunks" logged ~1 min
  later, SWI-Prolog loaded in the container, frontend on :5173, the Tuned toggle disabled (no adapter
  in a clean clone). Demo against it: answers, s.379 CONSISTENT, s.304A INSUFFICIENT and the
  refusal all worked before Groq's free-tier **daily** cap (200k tokens) ran out from the day's testing.
  App INFO logs were invisible under uvicorn (no "index ready" signal); now shown.
- **Rate limits found by the demo:** Groq's free tier allows 8k tokens/minute; one answer plus the
  two extraction runs can exceed it, and the second demo question's verification failed with 429.
  GroqClient now retries up to 5 times honouring `retry-after`.
- **Tests:** `test_integration_e2e.py` runs the real parser, retriever, prompt builder, explanation
  layer and verification service over real IPC text with stub embedder/LLMs (and the real
  PrologEngine when SWI-Prolog is installed, a Python stand-in otherwise); `LEGALGPT_LIVE_E2E=1`
  runs the same chain through `create_app()` with Groq, MiniLM, the real index and SWI-Prolog.
  CI (`.github/workflows/ci.yml`): `pytest -m "not integration"` + frontend build. Not run on
  GitHub at first (no remote yet); since the repo was pushed (2026-10-03) every push runs it on
  GitHub Actions, and the first runs passed (backend tests with SWI-Prolog, frontend build).

## Known limitations

**Scope**
- IPC only. The IPC was replaced by the Bharatiya Nyaya Sanhita on 1 July 2024; offences committed
  after that date fall under the BNS, which is not indexed (one seed case, case04, already cites it).
- Retrieval is over the bare act only: no judgments, no CrPC/Evidence Act, so procedure, bail and
  sentencing practice are out of reach.
- Not legal advice. The UI and the demo say so; the evidence-match band is not answer correctness.

**Retrieval**
- **Fact-pattern questions:** the 0.55 floor was calibrated on short questions. Narratives pull
  generic chunks (§1/§2, state-amendment boilerplate) above the floor, so an off-scope narrative can
  reach the model with irrelevant sources; and the relevant punishment section can be missed — in the
  demo, the bicycle-theft narrative retrieved §378's illustrations and §381 but not §379, and the
  answer (correctly) said the punishment was not in its sources. Needs a fact-pattern calibration set
  and down-weighting of definitional / state-amendment chunks.
- **Follow-up context leaks into unrelated questions:** in a conversation, "How do I file my income
  tax return online?" after a question on s.304 retrieved s.304/302/304A at 0.62 instead of being
  refused (alone it scores 0.24 and is refused). The model still answered "I don't know", but the
  refusal should not depend on the model.
- MiniLM is a general-purpose embedder; no legal-domain embedding model or re-ranker was evaluated.

**Answers and explanation**
- Attribution catches off-topic or unsupported sentences, not subtle legal errors: a sentence about
  a neighbouring offence (§304 vs §304A) scores like a supported one.
- `citation_accuracy` (Phase 4) measures grounding, not relevance or correctness.

**Tuned model**
- Trained on 124 examples from 26 cases: it cites in the required format but copies sources and
  over-refuses (Phase 4) — in Docker it answered "I don't know" to s.304A and s.379 questions with the
  right section as source [1] (the Why panel flags it). Not a drop-in replacement for the hosted model.
  In the CPU Docker image it runs in bfloat16 (~3 GB; same speed as float32 on this CPU, 5 tokens/s):
  first answer ~2 min including the model load, then ~20–30 s. Its 3 GB base model downloads at
  ~0.2 MB/s inside Docker here, so `HF_CACHE_DIR` mounts the host's Hugging Face cache instead.

**Verification**
- 7 sections (by design). s.337/338 do not exclude cases where the victim died; s.323 ignores the
  s.334 grave-provocation case; exceptions and general defences (Chapter IV) are not encoded.
- The verdict is only as good as the extracted facts; the quote check proves a quote exists, not
  that it supports the value (case01's first run used the father's hearsay as evidence of driving).
  Extraction is legal judgement in disguise (case07: "accidental fire" classed as not negligent).
- Each verification costs two hosted-model calls; on the Groq free tier several in a row hit the
  per-minute token limit (retried, but slow).

**Dataset and annotation**
- 31 seed cases plus 1 BERT-assisted case (3 more queued for the team); single annotation pass, so inter-annotator agreement on real data
  is **unmeasured** and the guideline is not validated the way the LLD intends. The 79
  `bert_assisted` labels were reviewed by Claude, not the team.
- Label skew: Facts 38%, Precedent 23%, Ruling 21%, Argument 10%, Law Applied 9%. Classifier
  macro-F1 0.664 ± 0.031 (5-fold, case-grouped) on this small set; 0.744 ± 0.040 with context and
  full fine-tuning (Phase 3, classifier v2). A whole unusual case can be labelled wrong with high
  confidence (case17: court reasoning labelled Ruling by the team, predicted Precedent); the audit
  sample, not the confidence bar, is what catches it.
- case18 and case21 skip 2 and 5 sentence ids in the team sheet; court and year are approximate for
  imported cases.

**Deployment**
- Local development/demo only: no authentication, rate limiting, HTTPS or multi-user isolation;
  the annotation store is a single append-only file.
- First `docker compose up` downloads MiniLM and embeds the corpus (~1–2 min); answers need
  internet access to Groq.

**Future work (not implemented)**
- **ZKML proofs of inference** — out of scope: proving a transformer forward pass in zero knowledge
  is still orders of magnitude too slow and memory-hungry for a 1.5B+ model on this hardware, and it
  would prove *which* model ran, not that the legal answer is right.
- BNS/BNSS corpus and IPC↔BNS section mapping; judgments as a second retrieval source.
- Double annotation with measured kappa; human-written answer targets for LoRA training.
- More Prolog sections, exceptions and Chapter IV defences.

## Final acceptance (checked 2026-10-03)

- [x] **Fresh clone + `docker compose up`, only `.env` set** — verified from a fresh clone with empty
  volumes: the index builds itself from the committed corpus (685 chunks), verification runs in the
  container, the frontend serves. A full 5/5 demo run on Docker was cut short by Groq's daily token
  cap; the same commit ran 5/5 against the local backend.
- [x] **Grounded answer with working citations (Module 0)** — "punishment for culpable homicide not
  amounting to murder" answered from s.304 (evidence match High 0.76); clicking [1] opens s.304.
- [x] **Real team-labelled batch in AnnotationStore (Module 1)** — 31 cases / 4,283 sentences from the
  team sheet (`legaltech_dataset.xlsx`; 8 of them carry the sheet's own annotator name "example"),
  frozen as v0.1 and v0.2 with hashes.
- [x] **InLegalBERT trained, metrics saved (Module 2)** — `data/models/rrl/v0.1-20261002-191352/metrics.json`
  (test macro-F1 0.642) and `data/models/rrl_cv/v0.1-cv5-20261002-200419.json` (5-fold, 0.664 ± 0.031).
- [x] **LoRA checkpoint + real base-vs-tuned report (Module 3)** — adapter `v0.2-20261002-203807`;
  `docs/reports/lora_v0.2_comparison.md`.
- [x] **Band + "Why this answer?" on every legal answer (Module 4)** — checked in the UI on an answer
  and on an out-of-scope question; refusals carry an explanation too (demo question 5).
- [x] **Real Prolog verdict for a genuine case in VerifyPanel (Module 5)** — case20 (G. Manickam,
  convicted under s.279/304A), s.279: INSUFFICIENT, three elements satisfied with quotes
  ("Goodshed Road, Coimbatore", "due to rash and negligent driving of the lorry driver…"),
  "endangered human life" not established. The Phase 6 report run gave CONSISTENT for the same
  input: the verdict depends on run-to-run extraction (see Known limitations).
- [x] **`demo.py` runs cleanly end to end** — exit 0, 5/5 ok (s.379 CONSISTENT, s.304A INSUFFICIENT,
  s.304B CONSISTENT, private defence High, income tax refused). On the Groq free tier, verified
  questions take 35–100 s because of rate-limit waits; `--no-verify` runs in seconds.
- [x] **README and PLANNING state implemented vs deferred, ZKML explicit** — README "Implemented vs
  future work" table; PLANNING "Future work".
