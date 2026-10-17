"""Case analysis: a structured, cited legal analysis of a case file. Never a verdict.

Steps (each streamed to the UI as it finishes):
  1. facts     the model reads the case file and returns a neutral summary, the key facts, each
               with a quote, and the IPC offences that may be engaged. A quote that is not in the
               file is flagged, and its fact is not used further.
  2. law       the IPC text of those offences (the chat's retriever, steered by them).
  3. elements  element-by-element check of each offence. For the 7 sections encoded as Prolog
               rules it is the rule engine (verification_service: facts extracted twice, quote-checked).
               Other sections are assessed by the model against the section text, and labelled so.
  4. similar   past judgments whose facts are closest (Facts passages of the judgment library),
               with what the court held there (their Ruling passage).
  5. analysis  a short written analysis citing only the numbered sources (IPC and the similar
               judgments) and the facts, with the chat's explainability (sentence attributions).
The system states what the law requires and what the facts show or leave open. It does not say
whether anyone is guilty: that depends on evidence tested in court.
"""

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from app.core.citations import normalize_stream
from app.core.models import Message, SourceEvidence
from app.core.prompt_builder import LEGAL_SYSTEM_PROMPT, PromptBuilder
from app.core.query_rewriter import Rewrite

DISCLAIMER = ("Legal analysis, not legal advice and not a prediction of the outcome. It shows what the law requires "
              "and what this file shows or leaves open; a court decides on the evidence. Check every point against "
              "the cited sources.")
MAX_CHARS_HOSTED, MAX_CHARS_LOCAL = 24000, 6000
MAX_SECTIONS = 4

FACTS_PROMPT = """You read a case file about a possible criminal matter in India and extract information. \
Return ONLY a JSON object, no other text:
{"summary": "2-3 neutral sentences: who, what, when, where",
 "facts": [{"fact": "one short factual statement", "quote": "the exact words from the case file that state it"}],
 "offences": [{"name": "offence name", "section": "IPC section number, e.g. 304A"}]}
Rules: 5 to 12 facts, each quote copied word for word from the file (a short phrase is enough). \
Offences: only sections of the Indian Penal Code, 1860 (not other Acts such as the Motor Vehicles Act) \
that these facts may engage, at most 4, [] if none. \
Do not say whether anyone is guilty, and do not add facts that are not in the file."""

ELEMENTS_PROMPT = """You check one offence of the Indian Penal Code against the facts of a case file. \
Use only the SECTION TEXT and the FACTS given. Return ONLY a JSON object:
{"elements": [{"element": "one ingredient of the offence, from the section text",
               "status": "shown" or "contradicted" or "not established",
               "facts": ["F1", ...], "explanation": "one sentence"}]}
"shown": the facts state it. "contradicted": the facts state the opposite. "not established": the \
facts are silent or unclear. Cite fact numbers (F1, F2, ...). Do not decide guilt."""

ANALYSIS_TASK = """CASE FILE SUMMARY: {summary}

FACTS FROM THE FILE:
{facts}

TASK: Write a short legal analysis (2-4 paragraphs) of these facts, using ONLY the SOURCES. \
Say which offences the facts may engage and why, which elements the facts appear to satisfy and which \
still need to be established, and how the cited judgments treated similar facts. Cite sources inline \
as [n]. Do NOT say whether the accused is guilty or predict the outcome; end by naming the key facts \
that the evidence would need to establish."""


def _json_object(reply: str) -> dict | None:
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL)
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(reply[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[“”\"'‘’]", "", text.lower())).strip()


@dataclass
class CaseFacts:
    summary: str
    facts: list[dict] = field(default_factory=list)  # {"id": "F1", "fact", "quote", "verified"}
    offences: list[tuple[str, str]] = field(default_factory=list)

    @property
    def verified(self) -> list[dict]:
        return [f for f in self.facts if f["verified"]]

    def facts_block(self) -> str:
        return "\n".join(f"{f['id']}. {f['fact']} (\"{f['quote']}\")" for f in self.verified)


def ask_json(llm: Any, messages: list[Message], tries: int = 2) -> dict:
    """A JSON reply. Reasoning models get their brief mode (generate_brief: a long hidden reasoning
    can use up the token budget and leave no answer); one retry if the reply has no JSON object."""
    ask = getattr(llm, "generate_brief", None) or llm.generate
    for _ in range(tries):
        data = _json_object(ask(messages))
        if data:
            return data
    return {}


_STOP = {"section", "sections", "punishment", "offence", "offences", "under", "with", "from", "into", "being", "person"}


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3 and w not in _STOP}


def matches_title(name: str, title: str) -> bool:
    """The model's offence name shares a word with the IPC section's title. "Driving under the
    influence" (Motor Vehicles Act s.185) shares none with IPC s.185 (illegal purchase at an auction)."""
    return bool(_stems(name) & _stems(title))


def extract_facts(text: str, llm: Any, valid_sections: dict[str, str] | None = None) -> CaseFacts:
    """valid_sections: IPC section -> title. An offence whose number is not an IPC section, or whose
    name has nothing in common with that section's title (e.g. a Motor Vehicles Act section offered
    by number), is dropped."""
    data = ask_json(llm, [Message("system", FACTS_PROMPT), Message("user", f"CASE FILE:\n{text}")])
    norm_text = _norm(text)
    facts, n = [], 0
    for f in data.get("facts") or []:
        if not isinstance(f, dict) or not str(f.get("fact", "")).strip():
            continue
        n += 1
        quote = str(f.get("quote", "")).strip()
        facts.append({"id": f"F{n}", "fact": str(f["fact"]).strip()[:300], "quote": quote[:300],
                      "verified": bool(quote) and _norm(quote) in norm_text})
    offences: list[tuple[str, str]] = []
    for o in data.get("offences") or []:
        if isinstance(o, dict):
            m = re.match(r"\s*(?:section\s*)?(\d{1,3}[A-Z]{0,2})\b", str(o.get("section", "")), re.I)
            name = re.sub(r"[\s;()]+", " ", str(o.get("name", ""))).strip()[:80]
            sec = m.group(1).upper() if m else ""
            ok = valid_sections is None or (sec in valid_sections and matches_title(name, valid_sections[sec]))
            if sec and name and sec not in {s for _, s in offences} and ok:
                offences.append((name, sec))
    return CaseFacts(str(data.get("summary", "")).strip()[:800], facts, offences[:MAX_SECTIONS])


class CaseAnalyzer:
    def __init__(self, retriever: Any, explainer: Any = None, verifier: Any = None, judgments: Any = None,
                 verifier_factory: Any = None, valid_sections: dict[str, str] | None = None) -> None:
        """verifier: the app's VerificationService (hosted fact extraction with fallbacks);
        verifier_factory(llm): one that uses only `llm`, for local-only analyses."""
        self.retriever, self.explainer, self.verifier, self.judgments = retriever, explainer, verifier, judgments
        self.verifier_factory = verifier_factory
        self.valid_sections = valid_sections

    def analyze(self, text: str, llm: Any, local_only: bool = False) -> Iterator[tuple[str, Any]]:
        limit = MAX_CHARS_LOCAL if local_only else MAX_CHARS_HOSTED
        truncated = len(text) > limit
        text = text[:limit]
        yield "step", {"id": "facts", "status": "running"}
        facts = extract_facts(text, llm, self.valid_sections)
        yield "facts", {"summary": facts.summary, "facts": facts.facts,
                        "offences": [{"name": n, "section": s} for n, s in facts.offences],
                        "truncated": truncated, "chars_read": len(text)}
        if not facts.summary and not facts.verified:
            yield "error", {"message": "The model could not read facts from this file. Try a hosted model or a shorter, clearer text."}
            return

        yield "step", {"id": "law", "status": "running"}
        rewrite = Rewrite(True, facts.offences) if facts.offences else None
        result = self.retriever.retrieve_with_diagnostics(facts.summary or text[:2000], rewrite=rewrite)
        statute = list(result.sources)
        yield "law", {"sources": [s.model_dump() for s in statute]}

        yield "step", {"id": "elements", "status": "running"}
        checks = []
        verifier = self.verifier_factory(llm) if (local_only and self.verifier_factory) else self.verifier
        encoded = {s.section for s in verifier.sections} if verifier is not None else set()
        for name, sec in facts.offences:
            if sec in encoded:
                try:
                    r = verifier.verify(text, sec).to_dict()
                    checks.append({"section": sec, "title": r["title"], "method": "rules", "status": r["status"], "result": r})
                    continue
                except Exception as e:  # fall back to the model check below
                    checks.append({"section": sec, "title": name, "method": "rules", "error": str(e)})
            sec_sources = [s for s in statute if s.section == sec]
            if not sec_sources:
                checks.append({"section": sec, "title": name, "method": "model", "elements": [],
                               "note": f"The text of section {sec} was not retrieved, so it was not checked."})
                continue
            checks.append(self._model_check(name, sec, sec_sources, facts, llm))
        yield "elements", {"checks": checks}

        yield "step", {"id": "similar", "status": "running"}
        similar = self._similar_cases(facts)
        yield "similar", {"cases": similar}

        yield "step", {"id": "analysis", "status": "running"}
        judgment_sources = [c["source"] for c in similar if c.get("source")]
        sources = statute + [SourceEvidence(**{**s, "marker": len(statute) + i + 1}) for i, s in enumerate(judgment_sources)]
        yield "sources", [s.model_dump() for s in sources]
        task = ANALYSIS_TASK.format(summary=facts.summary, facts=facts.facts_block() or "(none verified)")
        messages = [Message("system", LEGAL_SYSTEM_PROMPT),
                    Message("user", f"SOURCES:\n{PromptBuilder.format_sources(sources)}\n\n{task}")]
        answer = []
        for piece in normalize_stream(llm.stream(messages)):
            answer.append(piece)
            yield "token", {"text": piece}
        if self.explainer is not None:
            try:
                rec = self.explainer.build(facts.summary, sources, "".join(answer), llm.model_id,
                                           best_similarity=result.best_similarity, retrieval_floor=result.min_similarity,
                                           refused=False)
                yield "explanation", rec.to_dict()
            except Exception:  # explainability is best-effort; the analysis already reached the user
                pass
        yield "done", {"disclaimer": DISCLAIMER, "model": llm.model_id, "local_only": local_only}

    def _model_check(self, name: str, sec: str, sec_sources: list, facts: CaseFacts, llm: Any) -> dict:
        section_text = "\n".join(s.text for s in sec_sources)[:3000]
        data = ask_json(llm, [Message("system", ELEMENTS_PROMPT),
                              Message("user", f"SECTION TEXT (IPC section {sec}):\n{section_text}\n\nFACTS:\n{facts.facts_block()}")])
        known = {f["id"] for f in facts.verified}
        elements = []
        for e in data.get("elements") or []:
            if not isinstance(e, dict) or not str(e.get("element", "")).strip():
                continue
            status = str(e.get("status", "")).lower()
            status = "satisfied" if status == "shown" else "violated" if status == "contradicted" else "missing"
            elements.append({"element": str(e["element"])[:200], "status": status,
                             "facts": [f for f in e.get("facts") or [] if f in known],
                             "explanation": str(e.get("explanation", ""))[:300]})
        statuses = {e["status"] for e in elements}
        overall = "INCONSISTENT" if "violated" in statuses else "INSUFFICIENT" if "missing" in statuses or not elements else "CONSISTENT"
        return {"section": sec, "title": name, "method": "model", "status": overall, "elements": elements,
                "markers": [s.marker for s in sec_sources]}

    def _similar_cases(self, facts: CaseFacts, k: int = 3) -> list[dict]:
        if self.judgments is None or not self.judgments.available():
            return []
        from app.judgments.library import chunk_meta

        query = facts.summary + " " + "; ".join(f"{n} (section {s})" for n, s in facts.offences)
        out = []
        for chunk, sim in self.judgments.search(query, roles=("Facts",), k=k, min_similarity=0.45):
            court, year, _ = chunk_meta(chunk)
            held = self.judgments.search(query, roles=("Ruling",), k=1, min_similarity=0.0, within_case=chunk.doc_id)
            case = {"case_id": chunk.doc_id, "title": chunk.title, "court": court, "year": year,
                    "similarity": round(sim, 3), "facts_passage": chunk.text, "facts_citation": chunk.citation_path}
            if held:
                case["source"] = self.judgments.to_sources(held, 0)[0].model_dump()
            out.append(case)
        return out
