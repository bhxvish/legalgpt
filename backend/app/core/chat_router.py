"""ChatRouter: POST /api/chat, streamed as Server-Sent Events.

Event stream (each `data:` line is JSON):
  meta         {"mode", "model", "refused"}   always first
  sources      [SourceEvidence, ...]          legal mode with evidence only
  token        {"text"}                        answer fragments, in order
  explanation  ExplanationRecord               legal mode, after the answer (incl. refusals)
  error        {"message"}                     generation/retrieval failure (stream then ends)
  done         {}                              always last
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


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class ChatRouter:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        prompt_builder: PromptBuilder | None = None,
        explainer: "ExplanationBuilder | None" = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.explainer = explainer
        self.router = APIRouter(prefix="/api", tags=["chat"])
        self.router.add_api_route("/chat", self.chat, methods=["POST"], response_class=StreamingResponse)

    def chat(self, req: ChatRequest) -> StreamingResponse:
        return StreamingResponse(
            self.events(req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def events(self, req: ChatRequest) -> Iterator[str]:
        history = [Message(t.role, t.content) for t in req.history]
        try:
            if req.mode == "general":
                yield from self._general(req.question, history)
            else:
                yield from self._legal(req.question, history)
        except Exception as exc:  # surface failures to the client instead of a dropped connection
            logger.exception("chat request failed")
            yield sse("error", {"message": str(exc) or exc.__class__.__name__})
        yield sse("done", {})

    def _general(self, question: str, history: list[Message]) -> Iterator[str]:
        yield sse("meta", {"mode": "general", "model": self.llm.model_id, "refused": False})
        messages = self.prompt_builder.build(question, [], "general", history)
        for text in normalize_stream(self.llm.stream(messages)):
            yield sse("token", {"text": text})
        yield sse("token", {"text": GENERAL_CAUTION})

    def _retrieve(self, question: str, context: str | None) -> RetrievalResult:
        if hasattr(self.retriever, "retrieve_with_diagnostics"):
            return self.retriever.retrieve_with_diagnostics(question, context=context)
        sources = self.retriever.retrieve(question, context=context)  # minimal retrievers (tests)
        return RetrievalResult(sources, max((s.similarity for s in sources), default=0.0), 0.0, len(sources))

    def _legal(self, question: str, history: list[Message]) -> Iterator[str]:
        previous = next((m.content for m in reversed(history) if m.role == "user"), None)
        result = self._retrieve(question, previous)
        sources = result.sources
        if not sources:
            yield sse("meta", {"mode": "legal", "model": self.llm.model_id, "refused": True})
            yield sse("token", {"text": SCOPE_REFUSAL})
            yield from self._explanation(question, result, SCOPE_REFUSAL, refused=True)
            return
        yield sse("meta", {"mode": "legal", "model": self.llm.model_id, "refused": False})
        yield sse("sources", [s.model_dump() for s in sources])
        messages = self.prompt_builder.build(question, sources, "legal", history)
        answer: list[str] = []
        for text in normalize_stream(self.llm.stream(messages)):
            answer.append(text)
            yield sse("token", {"text": text})
        yield from self._explanation(question, result, "".join(answer), refused=False)

    def _explanation(self, question: str, result: RetrievalResult, answer: str, refused: bool) -> Iterator[str]:
        """Emitted after the last token, so it never delays the answer. A failure here is
        logged and skipped: the answer the user already has must not turn into an error."""
        if self.explainer is None:
            return
        try:
            record = self.explainer.build(
                question, result.sources, answer, self.llm.model_id,
                best_similarity=result.best_similarity, retrieval_floor=result.min_similarity, refused=refused,
            )
            yield self.explainer.to_sse_event(record)
        except Exception:
            logger.exception("explanation failed")
