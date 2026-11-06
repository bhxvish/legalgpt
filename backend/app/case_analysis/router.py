"""POST /api/case/analyze: send a case file (.txt / .pdf, base64 in JSON) or its text; the analysis streams
back as Server-Sent Events (step, facts, law, elements, similar, sources, token, explanation, done).

local_only=true runs every model step on a model of this machine (Local (LoRA) / Local heavy), so
the case file is not sent to any hosted service; the judgment and IPC searches are local anyway.
Nothing is stored: the text exists only for the duration of the request.

POST /api/case/ask: a follow-up question about a finished analysis. The browser sends the analysis
back with the question (nothing was kept here); the answer streams as sources, token, explanation, done,
and passes the same output guard as the analysis (answer_replace / guard when a sentence was removed).
"""

import base64
import binascii
import json
import logging
from collections.abc import Iterator
from typing import Any, Literal

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


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class AskRequest(BaseModel):
    """A question about a finished analysis. The server kept nothing, so the browser sends the analysis
    back: summary, facts, offences, element checks, the analysis text and its numbered sources."""

    question: str = Field(min_length=1, max_length=2000)
    context: dict[str, Any]
    history: list[Turn] = Field(default_factory=list, max_length=16)
    model: str = ""
    local_only: bool = False


class CaseBiasRequest(AnalyzeRequest):
    """The same file or text as the analysis, plus what the analysis found (offences, element checks)."""

    attribute: Literal["gender", "religion", "caste", "income", "region", "age"] = "gender"
    original: dict[str, Any] | None = None
    control: bool = True


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


class StickyFallback:
    """For one analysis: try the chosen hosted model, and if it refuses (quota, a 403 from the
    provider, an outage) switch to the next free hosted model and stay with it for the rest of the
    analysis, so the steps do not mix models. `note` says what happened."""

    is_local = False

    def __init__(self, clients: list) -> None:
        self.clients, self.i, self.note = clients, 0, ""

    @property
    def model_id(self) -> str:
        return self.clients[self.i].model_id

    def _call(self, method: str, *args):
        from app.core.llm_client import LLMClientError

        while True:
            try:
                return getattr(self.clients[self.i], method)(*args)
            except LLMClientError as e:
                if self.i + 1 >= len(self.clients):
                    raise
                failed = self.clients[self.i].model_id
                self.i += 1
                self.note = f"{failed} refused ({str(e)[:120]}); continued with {self.model_id}"
                logger.warning("case analysis: %s", self.note)

    def generate(self, messages):
        return self._call("generate", messages)

    def generate_brief(self, messages):
        c = self.clients[self.i]
        return self._call("generate_brief" if hasattr(c, "generate_brief") else "generate", messages)

    def stream(self, messages):
        try:
            first = self.clients[self.i]
            yield from first.stream(messages)
        except Exception as e:  # nothing streamed yet in practice: fall back to a whole answer
            from app.core.llm_client import LLMClientError

            if not isinstance(e, LLMClientError) or self.i + 1 >= len(self.clients):
                raise
            self.i += 1
            self.note = f"{first.model_id} refused; continued with {self.model_id}"
            yield self.clients[self.i].generate(messages)


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class CaseRouter:
    def __init__(self, chat_router: Any, verifier_factory: Any = None) -> None:
        self.chat = chat_router
        self.analyzer = CaseAnalyzer(chat_router.retriever, chat_router.explainer, chat_router.verifier,
                                     getattr(chat_router, "judgments", None), verifier_factory, ipc_sections())
        self.router = APIRouter(prefix="/api/case", tags=["case analysis"])
        self.router.add_api_route("/analyze", self.analyze, methods=["POST"])
        self.router.add_api_route("/ask", self.ask, methods=["POST"])
        self.router.add_api_route("/bias", self.bias, methods=["POST"])
        self.router.add_api_route("/info", self.info, methods=["GET"])

    def info(self) -> dict[str, Any]:
        local = [k for k, m in self.chat.models.items() if getattr(m, "is_local", False)]
        return {"disclaimer": DISCLAIMER, "local_models": local, "default_model": self.chat.default_model,
                "encoded_sections": [s.section for s in self.chat.verifier.sections] if self.chat.verifier else [],
                "judgments": bool(self.analyzer.judgments and self.analyzer.judgments.available())}

    async def analyze(self, req: AnalyzeRequest) -> StreamingResponse:
        text = self._case_text(req)
        llm = self._pick_llm(req.model, req.local_only)
        return StreamingResponse(self._stream(self.analyzer.analyze(text, llm, local_only=req.local_only)),
                                 media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def bias(self, req: CaseBiasRequest) -> dict:
        """The case file with one attribute (gender, caste, age...) changed, through facts, offences and element checks;
        compared with the original analysis (from the browser) and, with `control`, with both versions
        analysed again. See app/explainability/bias.py."""
        from app.explainability import bias as bias_mod

        text = self._case_text(req)
        cf = bias_mod.counterfactual(text, req.attribute)
        if cf.mode == "none":
            return {"verdict": "not_applicable", "swaps": [],
                    "summary": f"The file names no {req.attribute} (no words or names to swap), so there is nothing to compare."}
        llm = self._pick_llm(req.model, req.local_only)
        o = req.original or {}
        original = bias_mod.Run(text=text, answer=o.get("analysis", ""),
                                cited=[str(x.get("section")) for x in o.get("offences", [])],
                                elements={f"s.{c.get('section')}": str(c.get("status") or "") for c in o.get("checks", [])
                                          if c.get("status")})
        first = lambda: self._assess(cf.first, llm, req.local_only)
        second = lambda: self._assess(cf.second, llm, req.local_only)
        if not original.cited or cf.mode == "added":  # with an added sentence, the first version is new too
            original, swapped = bias_mod.both(first, second)
        else:
            swapped = second()
        titles = dict(self.analyzer.valid_sections or {})
        offences = bias_mod.same_offence(titles)

        def judge(control=None, repeat=None):
            return bias_mod.compare(req.attribute, original, swapped, cf.swaps, "ipc", control, titles, cf.mode, repeat, offences)

        report = judge()
        if req.control and report.verdict == "possible_bias":  # only a difference needs the second runs
            report = judge(*bias_mod.both(first, second))
        report.sensitive_because = bias_mod.sensitivity(text, original.cited + swapped.cited)
        return report.to_dict()

    def _assess(self, text: str, llm: Any, local_only: bool, tries: int = 3) -> Any:
        """Steps 1-3 of an analysis (facts, offences, element checks) as a bias_mod.Run; the similar
        judgments and the written analysis are not needed to compare two versions. A reading that found
        no offences is usually a failed model call, and would look like a difference the swap did not
        cause, so it is tried again."""
        for _ in range(tries):
            run = self._assess_once(text, llm, local_only)
            if run.cited:
                break
        return run

    def _assess_once(self, text: str, llm: Any, local_only: bool) -> Any:
        from app.explainability import bias as bias_mod

        run = bias_mod.Run(text=text)
        for event, data in self.analyzer.analyze(text, llm, local_only=local_only):
            if event == "facts":
                run.cited = [str(o["section"]) for o in data.get("offences", [])]
            elif event == "elements":
                run.elements = {f"s.{c['section']}": str(c.get("status") or "") for c in data["checks"] if c.get("status")}
                break
            elif event in ("similar", "error"):
                break
        run.model = getattr(llm, "model_id", "")
        return run

    def _case_text(self, req: Any) -> str:
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
        return text

    async def ask(self, req: AskRequest) -> StreamingResponse:
        if not req.context.get("sources"):
            raise HTTPException(422, "there is no analysis with sources to ask about")
        try:
            from app.core.models import SourceEvidence

            [SourceEvidence(**s) for s in req.context["sources"]]
        except Exception as e:
            raise HTTPException(422, f"the analysis sources are malformed: {e}") from e
        from app.core.models import Message

        llm = self._pick_llm(req.model, req.local_only)
        history = [Message(t.role, t.content) for t in req.history]
        return StreamingResponse(self._stream(self.analyzer.follow_up(req.question, req.context, history, llm)),
                                 media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def _pick_llm(self, model: str, local_only: bool) -> Any:
        name = model or self.chat.default_model
        if name not in self.chat.models:
            raise HTTPException(422, f"unknown model {name!r}")
        llm = self.chat.models[name]
        if local_only and not getattr(llm, "is_local", False):
            raise HTTPException(422, "local-only analysis needs a local model (Local (LoRA) or Local heavy)")
        if not local_only and not getattr(llm, "is_local", False):
            # free hosted models to fall back on (Groq, Gemini); paid providers are never added
            backups = [m for k, m in self.chat.models.items() if m is not llm and not getattr(m, "is_local", False)
                       and (k == "groq" or str(getattr(m, "model_id", "")).startswith("gemini:"))]
            llm = StickyFallback([llm] + backups)
        from app.core.llamacpp_client import LlamaCppClient, RoomyLocal

        return RoomyLocal(llm) if isinstance(llm, LlamaCppClient) else llm

    def _stream(self, events: Iterator[tuple[str, Any]]) -> Iterator[str]:
        try:
            for event, data in events:
                yield sse(event, data)
        except Exception as e:  # report on the stream: the response has already started
            logger.exception("case analysis failed")
            yield sse("error", {"message": f"{type(e).__name__}: {e}"})
