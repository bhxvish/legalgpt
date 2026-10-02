"""ChatRouter: POST /api/chat, streamed as Server-Sent Events; GET /api/models.

Event stream (each `data:` line is JSON):
  meta          {"mode", "model", "refused"}   always first
  sources       [SourceEvidence, ...]          legal mode with evidence only
  token         {"text"}                        answer fragments, in order
  explanation   ExplanationRecord               legal mode, after the answer (incl. refusals)
  verification  VerificationResult | {"error"}  legal mode, only when the request names verify_section
  error         {"message"}                     generation/retrieval failure (stream then ends)
  done          {}                              always last
"""

import json
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.citations import normalize_stream  # noqa: F401  (re-exported for callers)
from app.core.llm_client import LLMClient
from app.core.models import Message
from app.core.prompt_builder import PromptBuilder
from app.core.retriever import RetrievalResult, Retriever

if TYPE_CHECKING:
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
    ) -> None:
        """`llm` is the default model. `models` adds named alternatives a request can choose with
        `model` (the base-vs-tuned toggle); `model_notes` explains slow or unavailable ones."""
        self.retriever = retriever
        self.llm = llm
        self.models: dict[str, LLMClient] = dict(models or {})
        self.default_model = default_model or next((k for k, v in self.models.items() if v is llm), "default")
        self.models.setdefault(self.default_model, llm)
        self.model_notes = dict(model_notes or {})
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.explainer = explainer
        self.verifier = verifier
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
        try:
            name = req.model or self.default_model
            if name not in self.models:
                raise ValueError(f"model {name!r} is not available: {self.model_notes.get(name, 'unknown model')}")
            llm = self.models[name]
            if req.mode == "general":
                yield from self._general(req.question, history, llm)
            else:
                yield from self._legal(req.question, history, llm)
                if req.verify_section:
                    yield from self._verification(req.question, req.verify_section)
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

    def _retrieve(self, question: str, context: str | None) -> RetrievalResult:
        if hasattr(self.retriever, "retrieve_with_diagnostics"):
            return self.retriever.retrieve_with_diagnostics(question, context=context)
        sources = self.retriever.retrieve(question, context=context)  # minimal retrievers (tests)
        return RetrievalResult(sources, max((s.similarity for s in sources), default=0.0), 0.0, len(sources))

    def _legal(self, question: str, history: list[Message], llm: LLMClient) -> Iterator[str]:
        previous = next((m.content for m in reversed(history) if m.role == "user"), None)
        result = self._retrieve(question, previous)
        sources = result.sources
        if not sources:
            yield sse("meta", {"mode": "legal", "model": llm.model_id, "refused": True})
            yield sse("token", {"text": SCOPE_REFUSAL})
            yield from self._explanation(question, result, SCOPE_REFUSAL, True, llm.model_id)
            return
        yield sse("meta", {"mode": "legal", "model": llm.model_id, "refused": False})
        yield sse("sources", [s.model_dump() for s in sources])
        messages = self.prompt_builder.build(question, sources, "legal", history)
        answer: list[str] = []
        for text in normalize_stream(llm.stream(messages)):
            answer.append(text)
            yield sse("token", {"text": text})
        yield from self._explanation(question, result, "".join(answer), False, llm.model_id)

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
