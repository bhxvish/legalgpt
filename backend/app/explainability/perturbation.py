"""Perturbation explanations: change the input, watch the output. They need no access to the model's
insides, so they work for hosted models (a black box) as well as local ones.

word_importance   remove each word of the question and recompute (no model call): the scope check's
                  crime probability, and the similarity to each cited section. The words whose removal
                  lowers them most are the ones that tied the question to that law (the lexicon's
                  everyday-to-statute mappings show up here: removing "stole" removes "theft").
source_influence  answer again with one source left out (one model call per source): a source whose
                  removal changes the sections the answer relies on was decisive for the conclusion.
fact_influence    answer again with one fact of the question left out (same sources): a fact whose
                  removal changes the conclusion is one the conclusion depends on. Compared with the
                  key facts the reasoning trace names, it gives a faithfulness score for the trace.

Every model-based check runs a control first: the unchanged input answered again. A change that the
control also shows is the model's run-to-run variation, not the removed part's influence.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from app.core.models import Message, SourceEvidence
from app.core.prompt_builder import NOT_COVERED_REPLY

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+[A-Za-z]{0,2}|[ऀ-ॿ]+")
STOPWORDS = set("""a an the and or but if of to in on at by for from with without into onto as is was were be been
being am are it its this that these those there here i me my we our you your he him his she her they them
their what which who whom whose when where why how do does did done have has had can could should would
will shall may might must not no so than then too very just also any some such only own same other
""".split())


def by_section(sources: list[SourceEvidence]) -> list[SourceEvidence]:
    """One statute source per section: its main text rather than a state amendment's."""
    out: dict[str, SourceEvidence] = {}
    for s in sources:
        if s.kind != "statute" or not s.section:
            continue
        if s.section not in out or ("State Amendments" in out[s.section].citation_path and "State Amendments" not in s.citation_path):
            out[s.section] = s
    return list(out.values())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))


def word_importance(question: str, sources: list[SourceEvidence], retriever: Any, rewrite_text: str | None = None,
                    max_words: int = 40) -> dict:
    """Per content word: how much its removal lowers the crime probability and each section's match."""
    from app.core.lay_terms import statutory_terms

    statute = by_section(sources)[:6]
    stored = retriever.store.get_embeddings([s.chunk_id for s in statute]) if hasattr(retriever.store, "get_embeddings") else {}
    if len(stored) < len(statute):  # a store without the method, or chunks not found: embed their text
        missing = [s for s in statute if s.chunk_id not in stored]
        for s, e in zip(missing, retriever.embedder.embed([s.text for s in missing])):
            stored[s.chunk_id] = e
    words = [(m.start(), m.end(), m.group()) for m in _WORD.finditer(question)
             if m.group().lower() not in STOPWORDS][:max_words]
    variants = [question] + [(question[:a] + question[b:]).replace("  ", " ") for a, b, _ in words]
    exp = retriever.expander
    search = list(retriever.embedder.embed([exp.search_query(v) for v in variants]))
    scope = getattr(retriever, "scope", None)
    plain = list(retriever.embedder.embed([exp.expand(v) for v in variants])) if scope is not None else None
    base_terms = set(statutory_terms(question))

    def sims(vec) -> dict[int, float]:
        return {s.marker: _cosine(vec, stored[s.chunk_id]) for s in statute if s.chunk_id in stored}

    base_sims = sims(search[0])
    base_p = scope.probability(plain[0]) if scope is not None else None
    out = []
    for i, (a, b, w) in enumerate(words, start=1):
        s = sims(search[i])
        lost_terms = sorted(base_terms - set(statutory_terms(variants[i])))
        out.append({
            "word": w, "start": a, "end": b,
            "scope_drop": None if base_p is None else round(base_p - scope.probability(plain[i]), 4),
            "match_drop": {str(m): round(base_sims[m] - s[m], 4) for m in base_sims},
            "lexicon": lost_terms,  # statutory wording this word triggered (lay_terms.py)
        })
    rewrite_gain = {}
    if rewrite_text:
        with_rw = retriever.embedder.embed([exp.search_query(f"{question}\n{rewrite_text}")])[0]
        rewrite_gain = {str(m): round(_cosine(with_rw, stored[sid]) - base_sims[m], 4)
                        for m, sid in ((s.marker, s.chunk_id) for s in statute) if sid in stored and m in base_sims}
    return {
        "words": out,
        "base": {"scope_probability": None if base_p is None else round(base_p, 4),
                 "match": {str(m): round(v, 4) for m, v in base_sims.items()}},
        "sections": [{"marker": s.marker, "section": s.section, "title": s.title} for s in statute],
        "rewrite_gain": rewrite_gain,  # how much the AI rewrite raised each section's match
        "method": "occlusion: each word removed in turn; no model call",
    }


# --- model-based checks ---------------------------------------------------------------------------------

def _answer(question: str, sources: list[SourceEvidence], llm: Any, prompt_builder: Any, law: str) -> str:
    messages = prompt_builder.build(question, sources, "legal", [], law=law)
    return llm.generate(messages)


def _outcome(answer: str, sources: list[SourceEvidence]) -> dict:
    from app.explainability.bias import cited_sections

    refused = NOT_COVERED_REPLY.lower() in (answer or "").lower()
    return {"refused": refused, "cited": [] if refused else cited_sections(answer, [s.model_dump() for s in sources])}


def _change(base: dict, new: dict) -> dict:
    lost = [s for s in base["cited"] if s not in new["cited"]]
    gained = [s for s in new["cited"] if s not in base["cited"]]
    return {"lost": lost, "gained": gained, "refusal_changed": base["refused"] != new["refused"],
            "changed": bool(lost or gained or base["refused"] != new["refused"])}


def source_influence(question: str, answer: str, sources: list[SourceEvidence], llm: Any, prompt_builder: Any,
                     law: str = "ipc", max_sources: int = 6) -> dict:
    """Leave-one-source-out: which sources the conclusion depends on."""
    base = _outcome(answer, sources)
    control = _outcome(_answer(question, sources, llm, prompt_builder, law), sources)
    noise = _change(base, control)
    noisy = set(noise["lost"]) | set(noise["gained"])
    rows = []
    for s in by_section(sources)[:max_sources]:  # a section leaves with all its chunks (main text, amendments)
        kept = [x for x in sources if not (x.kind == "statute" and x.section == s.section)]
        markers = [x.marker for x in sources if x.kind == "statute" and x.section == s.section]
        out = _outcome(_answer(question, kept, llm, prompt_builder, law), kept)
        ch = _change(base, out)
        real = [x for x in ch["lost"] + ch["gained"] if x not in noisy] or (ch["refusal_changed"] and not noise["refusal_changed"])
        rows.append({"marker": s.marker, "markers": markers, "section": s.section, "title": s.title, **ch,
                     "decisive": bool(real), "cited_by_answer": s.section in base["cited"]})
    return {"baseline": base, "control": {**control, **noise}, "sources": rows,
            "method": "leave one section out (all its chunks); the unchanged prompt is answered again as a control"}


def split_facts(question: str, max_facts: int = 5) -> list[str]:
    """The question's clauses: sentences, split again at commas and 'and' when long."""
    parts = []
    for sent in re.split(r"(?<=[.!?।])\s+", question.strip()):
        if len(sent.split()) > 14:
            parts += [p.strip() for p in re.split(r",\s+|\s+and\s+(?=\w+\s+\w+)", sent) if p.strip()]
        else:
            parts.append(sent.strip())
    # the question itself ("Which offence is this?") is not a fact
    facts = [p.rstrip(".") for p in parts if p and not re.match(r"(?i)^(which|what|is|can|how|whether|who)\b", p)]
    return [f for f in facts if len(f.split()) >= 3][:max_facts]


def fact_influence(question: str, sources: list[SourceEvidence], llm: Any, prompt_builder: Any, law: str = "ipc",
                   key_facts: list[str] | None = None, answer: str | None = None) -> dict:
    """Leave-one-fact-out with the same sources; agreement with the trace's key facts."""
    keys = [k for k in key_facts or [] if k]
    # the question's other clauses, minus those overlapping a key fact (already tested through it)
    others = [c for c in split_facts(question)
              if not any(_squash(k) in _squash(c) or _squash(c) in _squash(k) for k in keys)]
    facts = list(dict.fromkeys(keys + others))[:6]
    base = _outcome(answer, sources) if answer else _outcome(_answer(question, sources, llm, prompt_builder, law), sources)
    control = _outcome(_answer(question, sources, llm, prompt_builder, law), sources)
    noise = _change(base, control)
    noisy = set(noise["lost"]) | set(noise["gained"])
    key = {_squash(k) for k in key_facts or []}
    rows = []
    for f in facts:
        reduced = re.sub(r"\s{2,}", " ", question.replace(f, " ")).strip(" ,.") or question
        out = _outcome(_answer(reduced, sources, llm, prompt_builder, law), sources)
        ch = _change(base, out)
        real = [x for x in ch["lost"] + ch["gained"] if x not in noisy] or (ch["refusal_changed"] and not noise["refusal_changed"])
        sq = _squash(f)  # a clause that contains a key fact (or is part of one) counts as named
        named_by_trace = any(k and (k in sq or sq in k) for k in key)
        rows.append({"fact": f, "named_by_trace": named_by_trace, **ch, "influential": bool(real)})
    named = [r for r in rows if r["named_by_trace"]]
    others = [r for r in rows if not r["named_by_trace"]]
    score = None
    if rows:
        agree = sum(r["influential"] == r["named_by_trace"] for r in rows)
        score = round(agree / len(rows), 3)
    return {"baseline": base, "control": {**control, **noise}, "facts": rows,
            "faithfulness": {
                "agreement": score,  # share of facts where "the trace says it matters" == "removing it changes the result"
                "key_facts_confirmed": sum(r["influential"] for r in named), "key_facts": len(named),
                "unnamed_but_influential": sum(r["influential"] for r in others), "unnamed": len(others)},
            "method": "leave one fact out, same sources; the unchanged question is answered again as a control"}


def _squash(s: str) -> str:
    return re.sub(r"\W+", " ", s or "").lower().strip()
