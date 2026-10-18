"""ChatRouter: POST /api/chat, streamed as Server-Sent Events; GET /api/models.

Event stream (each `data:` line is JSON):
  meta          {"mode", "model", "refused"}   always first
  sources       [SourceEvidence, ...]          legal mode with evidence only
  token         {"text"}                        answer fragments, in order
  explanation   ExplanationRecord               legal mode, after the answer (incl. refusals)
  verification  VerificationResult | {"error"}  legal mode, only when the request names verify_section
  error         {"message"}                     generation/retrieval failure (stream then ends)
  retrieval     how the sources were chosen    legal mode, after meta / sources (floor, scope check, lexicon)
  attestation   signed answer record (app/zk)  legal mode, when the server signs answers
  done          {}                              always last
"""

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.citations import normalize_stream  # noqa: F401  (re-exported for callers)
from app.core.llm_client import LLMClient, LLMRateLimitError
from app.core.models import Message
from app.core.prompt_builder import PromptBuilder
from app.core.retriever import RetrievalResult, Retriever

if TYPE_CHECKING:
    from app.core.query_rewriter import QueryRewriter
    from app.explainability.explanation_builder import ExplanationBuilder
    from app.verification.verification_service import VerificationService

logger = logging.getLogger(__name__)

SCOPE_REFUSAL = (
    "I can't answer this in Legal mode: none of the indexed legal sources "
    "(the Indian Penal Code, 1860) match the question closely enough to give a grounded answer. "
    "Try naming the offence or section you're asking about, or switch to General mode for an "
    "answer that is not grounded in sources."
)

GENERAL_CAUTION = (
    "\n\n---\n*General mode: this reply is not grounded in legal sources and may be inaccurate. "
    "It is not legal advice. Switch to Legal mode for answers cited from the IPC.*"
)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: Literal["legal", "general"] = "legal"
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    model: str | None = Field(default=None, max_length=40)  # an id from /api/models; None = server default
    # Legal mode: also check the facts stated in the question against this IPC section (Module 5).
    verify_section: str | None = Field(default=None, max_length=10)


class ModelInfo(BaseModel):
    id: str
    model_id: str
    available: bool
    default: bool = False
    note: str = ""


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class ChatRouter:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        prompt_builder: PromptBuilder | None = None,
        explainer: "ExplanationBuilder | None" = None,
        models: dict[str, LLMClient] | None = None,
        default_model: str | None = None,
        model_notes: dict[str, str] | None = None,
        verifier: "VerificationService | None" = None,
        rewriter: "QueryRewriter | None" = None,
        offline: bool = False,
        judgments: "Any | None" = None,
        translator: "Any | None" = None,
    ) -> None:
        """`llm` is the default model. `models` adds named alternatives a request can choose with
        `model` (the base-vs-tuned toggle); `model_notes` explains slow or unavailable ones."""
        self.retriever = retriever
        self.llm = llm
        # optional: (question, prompt, answer, sources, model_id, refused, follow_up) -> signed record
        self.attest: Callable[..., dict] | None = None
        self.models: dict[str, LLMClient] = dict(models or {})
        self.default_model = default_model or next((k for k, v in self.models.items() if v is llm), "default")
        self.models.setdefault(self.default_model, llm)
        self.model_notes = dict(model_notes or {})
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.explainer = explainer
        self.verifier = verifier
        self.rewriter = rewriter  # LLM query rewrite for everyday fact patterns (query_rewriter.py)
        self.offline = offline  # OFFLINE_MODE: local models rewrite for themselves; nothing goes to a hosted model
        self.judgments = judgments  # JudgmentSearch: real judgments cited after the IPC sources
        if translator is None:
            from app.core.language import Translator

            translator = Translator()
        self.translator = translator  # Hindi questions -> English for retrieval (language.py)
        self.router = APIRouter(prefix="/api", tags=["chat"])
        self.router.add_api_route("/chat", self.chat, methods=["POST"], response_class=StreamingResponse)
        self.router.add_api_route("/models", self.list_models, methods=["GET"], response_model=list[ModelInfo])

    def list_models(self) -> list[ModelInfo]:
        out = [ModelInfo(id=k, model_id=v.model_id, available=True, default=k == self.default_model,
                         note=self.model_notes.get(k, "")) for k, v in self.models.items()]
        out += [ModelInfo(id=k, model_id="", available=False, note=n) for k, n in self.model_notes.items() if k not in self.models]
        return sorted(out, key=lambda m: not m.default)

    def chat(self, req: ChatRequest) -> StreamingResponse:
        return StreamingResponse(
            self.events(req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def events(self, req: ChatRequest) -> Iterator[str]:
        history = [Message(t.role, t.content) for t in req.history]
        name = req.model or self.default_model
        try:
            if name not in self.models:
                raise ValueError(f"model {name!r} is not available: {self.model_notes.get(name, 'unknown model')}")
            llm = self.models[name]
            if req.mode == "general":
                yield from self._general(req.question, history, llm)
            else:
                yield from self._legal(req.question, history, llm)
                if req.verify_section:
                    yield from self._verification(req.question, req.verify_section)
        except LLMRateLimitError as exc:  # expected on the free tier: no traceback, say what to do
            logger.warning("chat request rate-limited: %s", exc)
            hint = (" You can switch to the Local (LoRA) model, which runs on this server."
                    if "adapter" in self.models and name != "adapter" else "")
            yield sse("error", {"message": f"{exc}{hint}"})
        except Exception as exc:  # surface failures to the client instead of a dropped connection
            logger.exception("chat request failed")
            yield sse("error", {"message": str(exc) or exc.__class__.__name__})
        yield sse("done", {})

    def _general(self, question: str, history: list[Message], llm: LLMClient) -> Iterator[str]:
        yield sse("meta", {"mode": "general", "model": llm.model_id, "refused": False})
        messages = self.prompt_builder.build(question, [], "general", history)
        for text in normalize_stream(llm.stream(messages)):
            yield sse("token", {"text": text})
        yield sse("token", {"text": GENERAL_CAUTION})

    def _rewrite_llm(self, llm: LLMClient) -> LLMClient:
        """The model that rewrites the query: the chat's model, except that a local model hands the
        rewrite to a hosted model when one is configured - unless OFFLINE_MODE, where it rewrites for
        itself (v0.7 adapters are trained for it)."""
        if self.offline or not getattr(llm, "is_local", False):
            return llm
        return next((m for m in self.models.values() if not getattr(m, "is_local", False)), llm)

    def _retrieve(self, question: str, context: str | None, llm: LLMClient | None = None) -> RetrievalResult:
        if hasattr(self.retriever, "retrieve_with_diagnostics"):
            rewrite = None
            # a question that names its section, or continues the previous one, needs no rewrite
            follow_up = bool(context) and self.retriever.expander.is_follow_up(question)
            if self.rewriter is not None and llm is not None and not follow_up                     and not self.retriever.expander.section_refs(question):
                rewrite = self.rewriter.rewrite(question, self._rewrite_llm(llm))
            return self.retriever.retrieve_with_diagnostics(question, context=context, rewrite=rewrite)
        sources = self.retriever.retrieve(question, context=context)  # minimal retrievers (tests)
        return RetrievalResult(sources, max((s.similarity for s in sources), default=0.0), 0.0, len(sources))

    def _translate(self, text: str | None, llm: LLMClient) -> tuple[str | None, str | None]:
        """(text to search with, its English translation if `text` is Hindi)."""
        from app.core.language import detect

        if not text or detect(text) != "hi" or self.translator is None:
            return text, None
        english = self.translator.to_english(text, self._rewrite_llm(llm))
        return english or text, english

    def _legal(self, question: str, history: list[Message], llm: LLMClient) -> Iterator[str]:
        from app.core.language import HINDI_RULE, SCOPE_REFUSAL_HI, detect

        hindi = detect(question) == "hi"
        previous = next((m.content for m in reversed(history) if m.role == "user"), None)
        # the IPC text and the embedding model are English: a Hindi question is searched in translation
        search, english = self._translate(question, llm)
        previous, _ = self._translate(previous, llm)
        result = self._retrieve(search, previous, llm)
        sources = result.sources
        if not sources:
            refusal = SCOPE_REFUSAL_HI if hindi else SCOPE_REFUSAL
            yield sse("meta", {"mode": "legal", "model": llm.model_id, "refused": True})
            yield sse("retrieval", self._retrieval_info(search, result, english))
            yield sse("token", {"text": refusal})
            yield from self._explanation(search, result, refusal, True, llm.model_id)
            yield from self._attestation(question, [], refusal, [], llm.model_id, True, result.used_context,
                                         self._used_rewrite(result))
            return
        from app.core.bns import annotate

        sources = annotate(list(sources)) + self._judgment_sources(question, result, len(sources))
        if len(sources) > len(result.sources):
            result = replace(result, sources=sources)
        yield sse("meta", {"mode": "legal", "model": llm.model_id, "refused": False})
        yield sse("sources", [s.model_dump() for s in sources])
        yield sse("retrieval", self._retrieval_info(search, result, english))
        asked = f"{question}\n(English translation: {english})" if english else question
        messages = self.prompt_builder.build(asked, sources, "legal", history)
        if hindi:
            messages[0] = Message("system", messages[0].content + "\n" + HINDI_RULE)
        answer: list[str] = []
        for text in normalize_stream(llm.stream(messages)):
            answer.append(text)
            yield sse("token", {"text": text})
        yield from self._explanation(search, result, "".join(answer), False, llm.model_id)
        yield from self._attestation(question, messages, "".join(answer), sources, llm.model_id, False,
                                     result.used_context, self._used_rewrite(result))

    def _judgment_sources(self, question: str, result: RetrievalResult, n_statute: int) -> list:
        """Passages of real judgments on the same offence, numbered after the IPC sources. Searched
        with the statute-aware query (the rewrite names the offence and section); only when the IPC
        retrieval found something, so out-of-scope questions never get case law. A failure here is
        logged and the answer goes ahead with the IPC sources alone."""
        if self.judgments is None or not n_statute:
            return []
        try:
            if not self.judgments.available():
                return []
            hits = self.judgments.search(result.query or question)
            return self.judgments.to_sources(hits, n_statute + 1)
        except Exception:
            logger.exception("judgment search failed")
            return []

    def _retrieval_info(self, question: str, result: RetrievalResult, english: str | None = None) -> dict[str, Any]:
        """Why these sources: the floor in force (lowered by the scope check for everyday fact
        patterns), the statutory wording the lexicon added, follow-up context, and per source its
        similarity and the weight applied by the reranker."""
        from app.core.lay_terms import statutory_terms
        from app.core.retriever import QueryExpander, defined_sections

        named = QueryExpander.section_refs(question)
        return {
            # a Hindi question: the English text that was searched (and that a retrieval proof covers)
            **({"translation": {"from": "hi", "text": english}} if english else {}),
            "floor": round(result.min_similarity, 3),
            "base_floor": getattr(self.retriever, "min_similarity", result.min_similarity),
            "best_similarity": round(result.best_similarity, 3),
            "used_context": result.used_context,
            "scope_probability": None if result.scope_probability is None else round(result.scope_probability, 3),
            "lexicon_terms": statutory_terms(question),
            "rewrite": result.rewrite,  # the LLM's reading of the question, and whether it steered retrieval
            "named_sections": named,
            # a definition question ("What is robbery?") also fetches the section titled with the term
            "defined_sections": [s for s in defined_sections(question, getattr(self.retriever, "titles", None)) if s not in named],
            "sources": [{"marker": s.marker, "section": s.section, "similarity": s.similarity,
                         "weight": round(s.score / s.similarity, 2) if s.similarity else None,
                         "named": s.section in named, "kind": s.kind,
                         **({"case": s.citation_path} if s.kind == "judgment" else {})} for s in result.sources],
        }

    @staticmethod
    def _used_rewrite(result: RetrievalResult) -> str | None:
        return result.rewrite["text"] if result.rewrite and result.rewrite.get("used") else None

    def _attestation(self, question: str, prompt: Any, answer: str, sources: list, model_id: str, refused: bool,
                     follow_up: bool, rewrite: str | None = None) -> Iterator[str]:
        """Signed record of this answer (after everything else; a failure is logged and skipped)."""
        if self.attest is None:
            return
        try:
            payload = [{"role": m.role, "content": m.content} for m in prompt] if prompt else []
            extra = {"rewrite": rewrite} if rewrite else {}
            yield sse("attestation", self.attest(question=question, prompt=payload, answer=answer, sources=sources,
                                                 model_id=model_id, refused=refused, follow_up=follow_up, **extra))
        except Exception:
            logger.exception("could not sign the answer")

    def _explanation(self, question: str, result: RetrievalResult, answer: str, refused: bool, model_id: str) -> Iterator[str]:
        """Emitted after the last token, so it never delays the answer. A failure here is
        logged and skipped: the answer the user already has must not turn into an error."""
        if self.explainer is None:
            return
        try:
            record = self.explainer.build(
                question, result.sources, answer, model_id,
                best_similarity=result.best_similarity, retrieval_floor=result.min_similarity, refused=refused,
            )
            yield self.explainer.to_sse_event(record)
        except Exception:
            logger.exception("explanation failed")

    def _verification(self, question: str, section: str) -> Iterator[str]:
        """Module 5 on request: check the facts stated in the question against `section`. Runs
        after the answer and its explanation, so it never delays them; failures become an
        error payload on this event, not a broken stream."""
        if self.verifier is None:
            yield sse("verification", {"error": "Verification is not available on this server (SWI-Prolog is not running)."})
            return
        try:
            yield sse("verification", self.verifier.verify(question, section).to_dict())
        except Exception as exc:  # unknown section, extraction failure, model outage
            logger.warning("verification failed: %s", exc)
            yield sse("verification", {"error": str(exc) or exc.__class__.__name__})
