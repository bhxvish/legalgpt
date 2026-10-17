"""POST /api/case/analyze: send a case file (.txt / .pdf, base64 in JSON) or its text; the analysis streams
back as Server-Sent Events (step, facts, law, elements, similar, sources, token, explanation, done).

local_only=true runs every model step on a model of this machine (Local (LoRA) / Local heavy), so
the case file is not sent to any hosted service; the judgment and IPC searches are local anyway.
Nothing is stored: the text exists only for the duration of the request.
"""

import base64
import binascii
import json
import logging
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.case_analysis.analyzer import DISCLAIMER, CaseAnalyzer

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MIN_CHARS = 200


class AnalyzeRequest(BaseModel):
    """The case text, or a .txt / .pdf file (base64; the browser reads it locally)."""

    text: str = Field("", max_length=200_000)
    filename: str = ""
    file_base64: str = Field("", max_length=12_000_000)
    model: str = ""
    local_only: bool = False


def ipc_sections() -> dict[str, str] | None:
    """IPC section -> title, from the corpus (offences outside it are dropped from an analysis)."""
    try:
        from app.config import get_settings
        from app.core.corpus import load_chunks

        out: dict[str, str] = {}
        for c in load_chunks(get_settings().corpus_chunks_path):
            if c.section and c.title:
                out.setdefault(c.section, c.title)
        return out
    except Exception:
        return None


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class CaseRouter:
    def __init__(self, chat_router: Any, verifier_factory: Any = None) -> None:
        self.chat = chat_router
        self.analyzer = CaseAnalyzer(chat_router.retriever, chat_router.explainer, chat_router.verifier,
                                     getattr(chat_router, "judgments", None), verifier_factory, ipc_sections())
        self.router = APIRouter(prefix="/api/case", tags=["case analysis"])
        self.router.add_api_route("/analyze", self.analyze, methods=["POST"])
        self.router.add_api_route("/info", self.info, methods=["GET"])

    def info(self) -> dict[str, Any]:
        local = [k for k, m in self.chat.models.items() if getattr(m, "is_local", False)]
        return {"disclaimer": DISCLAIMER, "local_models": local, "default_model": self.chat.default_model,
                "encoded_sections": [s.section for s in self.chat.verifier.sections] if self.chat.verifier else [],
                "judgments": bool(self.analyzer.judgments and self.analyzer.judgments.available())}

    async def analyze(self, req: AnalyzeRequest) -> StreamingResponse:
        text = req.text
        if req.file_base64:
            try:
                data = base64.b64decode(req.file_base64, validate=True)
            except (binascii.Error, ValueError) as e:
                raise HTTPException(422, "the file could not be decoded") from e
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "the file is larger than 8 MB")
            name = (req.filename or "").lower()
            if name.endswith(".pdf"):
                from app.labeling_assistant.pdf_text import pdf_to_judgment_text

                try:
                    text = pdf_to_judgment_text(data)
                except Exception as e:
                    raise HTTPException(422, f"could not read the PDF: {e}") from e
            elif name.endswith(".txt"):
                text = data.decode("utf-8", errors="replace")
            else:
                raise HTTPException(415, "upload a .txt or .pdf file")
        text = (text or "").strip()
        if len(text) < MIN_CHARS:
            raise HTTPException(422, f"the case text is too short (at least {MIN_CHARS} characters)")
        name = req.model or self.chat.default_model
        if name not in self.chat.models:
            raise HTTPException(422, f"unknown model {name!r}")
        llm = self.chat.models[name]
        local_only = req.local_only
        if local_only and not getattr(llm, "is_local", False):
            raise HTTPException(422, "local-only analysis needs a local model (Local (LoRA) or Local heavy)")
        return StreamingResponse(self._stream(text, llm, local_only), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def _stream(self, text: str, llm: Any, local_only: bool) -> Iterator[str]:
        try:
            for event, data in self.analyzer.analyze(text, llm, local_only=local_only):
                yield sse(event, data)
        except Exception as e:  # report on the stream: the response has already started
            logger.exception("case analysis failed")
            yield sse("error", {"message": f"{type(e).__name__}: {e}"})
