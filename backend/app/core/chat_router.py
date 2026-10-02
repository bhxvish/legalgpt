"""ChatRouter: POST /api/chat, streamed as Server-Sent Events.

Event stream (each `data:` line is JSON):
  meta     {"mode", "model", "refused"}   always first
  sources  [SourceEvidence, ...]          legal mode with evidence only
  token    {"text"}                        answer fragments, in order
  error    {"message"}                     generation/retrieval failure (stream then ends)
  done     {}                              always last
"""

import json
import logging
import re
from collections.abc import Iterable, Iterator
from typing import Any, Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.llm_client import LLMClient
from app.core.models import Message, SourceEvidence
from app.core.prompt_builder import PromptBuilder
from app.core.retriever import Retriever

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


# Models differ in how they write citations: gpt-oss emits 【1】 and [1†L2-L4]. Markers are
# canonicalized to [n] so the UI (and Phase 5's CitationValidator) see one format.
_CANONICAL = str.maketrans({chr(0x3010): "[", chr(0x3011): "]", chr(0xFF3B): "[", chr(0xFF3D): "]", chr(0x202F): " ", chr(0x00A0): " "})  # CJK and full-width brackets, narrow and no-break spaces
_MARKER = re.compile(r"\[(\d+)(?:[" + chr(0x2020) + chr(0x2021) + r"][^\]]*)?\]")  # [1], [1<dagger>L2-L4], [1<double dagger>...]
_MAX_PENDING = 32  # longest bracket we hold back waiting for its "]"


def _canonical_markers(text: str) -> str:
    return _MARKER.sub(lambda m: f"[{m.group(1)}]", text)


def normalize_stream(fragments: Iterable[str]) -> Iterator[str]:
    """Canonicalize citation markers in a token stream. A marker can be split across fragments
    ("[", "1†L2", "-L4]"), so text from an unclosed "[" is held back until it closes."""
    pending = ""
    for fragment in fragments:
        text = pending + fragment.translate(_CANONICAL)
        open_at = text.rfind("[")
        if open_at != -1 and "]" not in text[open_at:] and len(text) - open_at <= _MAX_PENDING:
            text, pending = text[:open_at], text[open_at:]
        else:
            pending = ""
        if text:
            yield _canonical_markers(text)
    if pending:
        yield _canonical_markers(pending)


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class ChatRouter:
    def __init__(self, retriever: Retriever, llm: LLMClient, prompt_builder: PromptBuilder | None = None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.prompt_builder = prompt_builder or PromptBuilder()
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

    def _legal(self, question: str, history: list[Message]) -> Iterator[str]:
        previous = next((m.content for m in reversed(history) if m.role == "user"), None)
        sources: list[SourceEvidence] = self.retriever.retrieve(question, context=previous)
        if not sources:
            yield sse("meta", {"mode": "legal", "model": self.llm.model_id, "refused": True})
            yield sse("token", {"text": SCOPE_REFUSAL})
            return
        yield sse("meta", {"mode": "legal", "model": self.llm.model_id, "refused": False})
        yield sse("sources", [s.model_dump() for s in sources])
        messages = self.prompt_builder.build(question, sources, "legal", history)
        for text in normalize_stream(self.llm.stream(messages)):
            yield sse("token", {"text": text})
