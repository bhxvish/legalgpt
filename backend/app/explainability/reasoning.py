"""Reasoning trace: how the model got from the question to its conclusion, in the IRAC form lawyers use.

  Issue        the legal question the facts raise
  Rule         each section considered: does it apply, and why (or why not), with the sources [n]
  Application  each key fact matched to an element of a section, and whether it holds
  Conclusion   the outcome, and the facts it depends on (key_facts, quoted from the question)

The model writes it after the answer, from the same numbered sources, as JSON; the server checks it:
source numbers and sections that are not among the sources are removed and counted, so the trace
cannot cite law the answer did not have. This is the model's own account of its reasoning, which may
not be what actually drove the answer (models do not always say what they think); perturbation.py
tests it - key facts the trace names should change the conclusion when they are removed.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.core.models import Message, SourceEvidence
from app.core.prompt_builder import PromptBuilder

REASONING_PROMPT = """You explain how a legal answer was reached, step by step, in the IRAC form.
Use ONLY the numbered SOURCES, the QUESTION and the ANSWER given. Return ONLY a JSON object:
{"issue": "the legal question the facts raise, one sentence",
 "rules": [{"section": "section number from a source", "applies": true or false,
            "why": "one sentence: which words of the section match or do not match the facts", "sources": [n]}],
 "application": [{"fact": "a fact from the question, in its own words", "element": "the element of the section it bears on",
                  "section": "section number", "holds": "yes" or "no" or "unclear", "sources": [n]}],
 "conclusion": "the outcome the answer reached, one or two sentences",
 "key_facts": ["short phrases copied word for word from the QUESTION that the conclusion depends on"]}
Rules: list every section the answer relied on, and any source section the answer did not use with
applies=false and why. 2 to 6 application steps. key_facts: at most 4, each copied exactly from the
question. Do not add facts, sections or sources that are not given. Do not say anyone is guilty."""


def _json(reply: str) -> dict | None:
    reply = re.sub(r"^```(?:json)?|```$", "", (reply or "").strip(), flags=re.M).strip()
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(reply[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\sऀ-ॿ]", " ", s or "").lower()).strip()


def build_trace(question: str, answer: str, sources: list[SourceEvidence], llm: Any, tries: int = 2) -> dict:
    """The model's IRAC trace, validated against the sources and the question."""
    statute = [s for s in sources if s.kind == "statute"]
    messages = [Message("system", REASONING_PROMPT),
                Message("user", f"SOURCES:\n{PromptBuilder.format_sources(sources)}\n\nQUESTION: {question}\n\nANSWER:\n{answer}")]
    data = None
    for _ in range(tries):
        gen = getattr(llm, "generate_brief", None) or llm.generate
        data = _json(gen(messages))
        if data:
            break
    if not data:
        return {"error": "The model did not return a reasoning trace. Try again or choose another model."}
    markers = {s.marker for s in sources}
    sections = {s.section for s in statute}
    from app.explainability.perturbation import by_section

    titles = {s.section: s.title for s in by_section(sources)}  # the main text's title, not a state amendment's
    removed = 0

    def clean_markers(xs) -> list[int]:
        nonlocal removed
        out = []
        for x in xs if isinstance(xs, list) else []:
            try:
                n = int(x)
            except (TypeError, ValueError):
                removed += 1
                continue
            if n in markers:
                out.append(n)
            else:
                removed += 1
        return out

    rules = []
    for r in data.get("rules", []) if isinstance(data.get("rules"), list) else []:
        sec = str(r.get("section", "")).upper().removeprefix("SECTION ").strip()
        if sec not in sections:
            removed += 1
            continue
        rules.append({"section": sec, "title": titles.get(sec, ""), "applies": bool(r.get("applies")),
                      "why": str(r.get("why", ""))[:400], "sources": clean_markers(r.get("sources"))})
    application = []
    for a in data.get("application", []) if isinstance(data.get("application"), list) else []:
        sec = str(a.get("section", "")).upper().removeprefix("SECTION ").strip()
        if sec and sec not in sections:
            removed += 1
            continue
        holds = str(a.get("holds", "unclear")).lower()
        application.append({"fact": str(a.get("fact", ""))[:300], "element": str(a.get("element", ""))[:300],
                            "section": sec, "holds": holds if holds in {"yes", "no", "unclear"} else "unclear",
                            "sources": clean_markers(a.get("sources"))})
    q = _norm(question)
    key_facts, unquoted = [], []
    for k in data.get("key_facts", []) if isinstance(data.get("key_facts"), list) else []:
        k = str(k).strip()
        (key_facts if k and _norm(k) in q else unquoted).append(k)
    return {
        "issue": str(data.get("issue", ""))[:500],
        "rules": rules,
        "application": application[:8],
        "conclusion": str(data.get("conclusion", ""))[:800],
        "key_facts": key_facts[:4],
        "unquoted_key_facts": unquoted[:4],  # named by the model but not found word for word in the question
        "removed": removed,  # source numbers or sections the model gave that were not among the sources
        "model": getattr(llm, "model_id", ""),
        "note": ("The model's own account of its reasoning, checked against the sources. Whether it is faithful is "
                 "tested by removing the key facts and sources it names (the influence checks)."),
    }
