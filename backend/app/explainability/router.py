"""/api/explain/*: explanations of a chat answer, computed on request (each is extra work the answer
itself does not need). The browser sends back what it already has: the question, the answer and its
numbered sources; nothing is stored here.

  POST /api/explain/words      which words of the question tied it to each section (no model call)
  POST /api/explain/reasoning  the model's reasoning as Issue / Rule / Application / Conclusion
  POST /api/explain/sources    leave-one-source-out: which sources the conclusion depends on
  POST /api/explain/facts      leave-one-fact-out, and how far the reasoning trace's key facts agree
  POST /api/explain/bias       the question again with gender, religion, caste, income, region or age changed: does the result change?
  POST /api/explain/sensitive  is this question gender- or caste-sensitive (the bias check then runs by itself)?
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.case_analysis.router import ipc_sections
from app.core.models import SourceEvidence
from app.explainability import bias as bias_mod
from app.explainability.perturbation import fact_influence, source_influence, word_importance
from app.explainability.reasoning import build_trace

logger = logging.getLogger(__name__)


class Answered(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    answer: str = Field("", max_length=20000)
    sources: list[dict[str, Any]] = Field(default_factory=list, max_length=12)
    model: str = ""
    law: Literal["auto", "ipc", "bns"] = "auto"


class WordsRequest(Answered):
    search_text: str | None = Field(None, max_length=4000)  # the English text searched (a Hindi question)
    rewrite_text: str | None = Field(None, max_length=2000)


class FactsRequest(Answered):
    key_facts: list[str] = Field(default_factory=list, max_length=6)


class BiasRequest(Answered):
    attribute: bias_mod.Attribute = "gender"
    refused: bool = False
    control: bool = True  # if they differ, run both versions a second time, to tell bias from run-to-run variation


class SensitiveRequest(BaseModel):
    text: str = Field("", max_length=200_000)
    sections: list[str] = Field(default_factory=list, max_length=20)
    law: Literal["ipc", "bns"] = "ipc"


@lru_cache(1)
def ipc_offences() -> dict[str, str]:
    return bias_mod.same_offence(ipc_sections() or {})


def parse_sse(events) -> dict[str, Any]:
    """The chat pipeline's events, collected: {event: data}, tokens joined into "answer"."""
    out: dict[str, Any] = {"answer": ""}
    for raw in events:
        name, data = "", ""
        for line in raw.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data += line[6:]
        try:
            value = json.loads(data) if data else None
        except json.JSONDecodeError:
            value = data
        if name == "token":
            out["answer"] += (value or {}).get("text", "")
        elif name:
            out[name] = value
    return out


class ExplainRouter:
    def __init__(self, chat_router: Any) -> None:
        self.chat = chat_router
        self.router = APIRouter(prefix="/api/explain", tags=["explain"])
        self.router.add_api_route("/results", self.results, methods=["GET"])
        for path, fn in (("/words", self.words), ("/reasoning", self.reasoning), ("/sources", self.sources),
                         ("/facts", self.facts), ("/bias", self.bias), ("/sensitive", self.sensitive)):
            self.router.add_api_route(path, fn, methods=["POST"])

    def results(self) -> dict:
        """The measured explainability results (scripts/make_explain_report.py) for the Explainability page."""
        from app.config import REPO_ROOT

        path = REPO_ROOT / "docs" / "reports" / "explainability_results.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    # helpers ----------------------------------------------------------------------------------------
    def _llm(self, name: str) -> Any:
        from app.case_analysis.router import StickyFallback

        name = name or self.chat.default_model
        if name not in self.chat.models:
            raise HTTPException(422, f"unknown model {name!r}")
        llm = self.chat.models[name]
        if getattr(llm, "is_local", False):
            return llm
        backups = [m for k, m in self.chat.models.items() if m is not llm and not getattr(m, "is_local", False)
                   and (k == "groq" or str(getattr(m, "model_id", "")).startswith("gemini:"))]
        return StickyFallback([llm] + backups)

    @staticmethod
    def _sources(raw: list[dict]) -> list[SourceEvidence]:
        try:
            return [SourceEvidence(**s) for s in raw]
        except Exception as e:
            raise HTTPException(422, f"the sources are malformed: {e}") from e

    def _law(self, req: Answered) -> str:
        from app.core.bns import choose_law

        law, _ = choose_law(req.question, req.law)
        return "bns" if law == "bns" and getattr(self.chat, "bns_retriever", None) is not None else "ipc"

    def _retriever(self, law: str) -> Any:
        return self.chat.bns_retriever if law == "bns" and self.chat.bns_retriever is not None else self.chat.retriever

    # endpoints ------------------------------------------------------------------------------------------
    def words(self, req: WordsRequest) -> dict:
        sources = self._sources(req.sources)
        if not [s for s in sources if s.kind == "statute"]:
            raise HTTPException(422, "the answer has no statute sources to explain")
        return word_importance(req.search_text or req.question, sources, self._retriever(self._law(req)), req.rewrite_text)

    def reasoning(self, req: Answered) -> dict:
        sources = self._sources(req.sources)
        if not req.answer or not sources:
            raise HTTPException(422, "a reasoning trace needs an answer and its sources")
        return build_trace(req.question, req.answer, sources, self._llm(req.model))

    def sources(self, req: Answered) -> dict:
        sources = self._sources(req.sources)
        if not req.answer or not sources:
            raise HTTPException(422, "source influence needs an answer and its sources")
        return source_influence(req.question, req.answer, sources, self._llm(req.model), self.chat.prompt_builder, self._law(req))

    def facts(self, req: FactsRequest) -> dict:
        sources = self._sources(req.sources)
        if not sources:
            raise HTTPException(422, "fact influence needs the answer's sources")
        return fact_influence(req.question, sources, self._llm(req.model), self.chat.prompt_builder, self._law(req),
                              key_facts=req.key_facts, answer=req.answer or None)

    def sensitive(self, req: SensitiveRequest) -> dict:
        caste = bias_mod.CASTE_WORDS.search(req.text or "")
        if caste:
            return {"reason": f"it mentions \u201c{caste.group(0)}\u201d", "attribute": "caste"}
        return {"reason": bias_mod.sensitivity(req.text, req.sections, req.law), "attribute": "gender"}

    def run_question(self, question: str, llm: Any, law: str, tries: int = 3) -> tuple[bias_mod.Run, list[dict]]:
        """One pass through the chat pipeline. A run whose query rewrite failed (a busy or rate-limited
        model) searched without it, so comparing it with another run would show a difference the swap
        did not cause: it is retried, and reported as an error if it keeps failing."""
        import time

        from app.core.llm_client import LLMClientError

        retriever = self._retriever("bns" if law == "bns" else "ipc")
        expected = self.chat.rewriter is not None and not retriever.expander.section_refs(question)
        for attempt in range(tries):
            got = parse_sse(self.chat._legal(question, [], llm, law))
            retrieval = got.get("retrieval") or {}
            if not expected or retrieval.get("rewrite") is not None or "translation" in retrieval and retrieval["translation"].get("failed"):
                break
            if attempt + 1 == tries:
                raise LLMClientError("The query rewrite failed (the model is busy or rate-limited), so this run cannot be "
                                     "compared fairly. Try again in a minute.")
            time.sleep(5 * (attempt + 1))
        sources = got.get("sources") or []
        refused = bool((got.get("meta") or {}).get("refused"))
        run = bias_mod.Run(text=question, answer=got["answer"], refused=refused,
                           retrieved=[s["section"] for s in sources if s.get("kind", "statute") == "statute"],
                           cited=[] if refused else bias_mod.relied_sections(got["answer"], sources),
                           model=(got.get("meta") or {}).get("model", ""))
        return run, sources

    def bias(self, req: BiasRequest) -> dict:
        llm = self._llm(req.model)
        cf = bias_mod.counterfactual(req.question, req.attribute)
        if cf.mode == "none":
            return {"verdict": "not_applicable", "swaps": [],
                    "summary": f"The question names no {req.attribute} (no words or names to swap), so there is nothing to compare."}
        first = lambda: self.run_question(cf.first, llm, req.law)
        second = lambda: self.run_question(cf.second, llm, req.law)
        if cf.mode == "swap" and (req.answer or req.refused):
            given = bias_mod.Run(text=req.question, answer=req.answer, refused=req.refused,
                                 retrieved=[s["section"] for s in req.sources if s.get("kind", "statute") == "statute"],
                                 cited=[] if req.refused else bias_mod.relied_sections(req.answer, req.sources))
            (original, orig_sources), (swapped, sw_sources) = (given, req.sources), second()
        else:
            (original, orig_sources), (swapped, sw_sources) = bias_mod.both(first, second)
        titles = {s["section"]: s.get("title", "") for s in orig_sources + sw_sources if s.get("section")}
        law = self._law(req)
        offences = ipc_offences() if law == "ipc" else None

        def judge(control=None, repeat=None):
            return bias_mod.compare(req.attribute, original, swapped, cf.swaps, law, control, titles, cf.mode, repeat, offences)

        report = judge()
        if req.control and report.verdict == "possible_bias":  # only a difference needs the second runs
            (control, _), (repeat, _) = bias_mod.both(first, second)
            report = judge(control, repeat)
        report.sensitive_because = bias_mod.sensitivity(req.question, original.cited + swapped.cited, law)
        return report.to_dict()
