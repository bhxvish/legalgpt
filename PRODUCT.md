# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The primary audience is a research panel and the reviewers of the accompanying paper: people
evaluating whether a legal-AI system for Indian criminal law can be trusted, usually through a short
live demo and the written results. Everyday users (citizens describing what happened, in English or
Hindi; law students; annotators labelling judgments) are the scenarios the system is built and
measured for, not yet a deployed user base.

## Product Purpose

LegalGPT answers questions about Indian criminal law from the statute text itself, with every claim
cited, and analyses case files without ever deciding guilt. It exists to show that such a system can
be measured and checked rather than merely asserted. Success is trust through evidence: every
answer, citation and headline number can be traced to its source and re-checked.

## Positioning

Its claims are measured and verifiable, not stated: a published benchmark with paired significance
tests, citation-faithfulness and prompt-injection evaluations, signed answers and zero-knowledge
proofs of retrieval, a Prolog rule engine checking offence elements, and one command that rebuilds
every result. Commercial legal-AI products assert reliability; this system shows its work.

## Operating Context

- Presented to a panel in a short live demo, alongside a research paper
  (`docs/reports/paper_analysis.md`); the Overview tab is the presenting surface.
- Free model tiers (Groq, Gemini) with daily quotas, plus local fine-tuned models via llama.cpp;
  quota exhaustion is a normal operating condition the UI must handle.
- Judgments come from public sources (Supreme Court 2020-2023, CC-BY-4.0); statutes from India Code.

## Capabilities and Constraints

- Tabs: Overview (the pipeline at a glance), Label (InLegalBERT suggests sentence roles, people check;
  blind samples), Chat (Legal / General; Law: Auto / IPC / BNS; English and Hindi), Case analysis
  (facts, offences, element checks, similar judgments, cited analysis), Verify (Prolog rule engine).
- Answers come only from retrieved sources (IPC, 1860 or BNS, 2023, plus judgment passages) and
  cite them `[n]`; out-of-scope (civil, procedural, special-law) questions are declined.
- Case analysis never states or predicts a verdict (implemented: prompt rules, an output guard and a
  verdict classifier).
- Binding constraint confirmed by the user: free model providers by default; paid models (OpenAI) only
  when the user chooses them, never as silent fallbacks.
- Terminology: IPC (Indian Penal Code, 1860), BNS (Bharatiya Nyaya Sanhita, 2023, in force from
  1 July 2024), sentence roles (Facts, Law Applied, Precedent, Argument, Ruling), modules M0-M5 as
  numbered in the paper.

## Evidence on Hand

- Measured results with sources: `docs/reports/paper_analysis.md`, `docs/reports/system_results.json`,
  reproducible via `backend/scripts/reproduce.py` (output `docs/reports/reproduced.md`).
- Benchmarks and data: `benchmark/`, `data/corpus/` (IPC and BNS chunks), the judgment library.
- AI-generated data is labelled as such in the repository (e.g. machine-translated Hindi questions,
  AI-written verdict-guard sentences, AI labels in the annotation store). No testimonials, users,
  customers or deployment claims exist; none may be invented.

## Product Principles

1. Evidence over assertion: every number on screen names its source; every answer cites its text.
2. Know the limits out loud: decline what the sources do not cover, and say why.
3. Never decide guilt: inform and structure, do not judge.
4. Free and reproducible by default: no hidden spending, no result that cannot be rebuilt.
5. Explainable to a panel in minutes: the whole system must be presentable at a glance.
