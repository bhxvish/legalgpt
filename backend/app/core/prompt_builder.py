"""PromptBuilder: turns a question + retrieved evidence into chat messages."""

from typing import Literal

from app.core.models import Message, SourceEvidence

Mode = Literal["legal", "general"]

NOT_COVERED_REPLY = "I don't know — the retrieved sources do not cover this question."

LEGAL_SYSTEM_PROMPT = f"""You are LegalGPT, an assistant for Indian criminal law (the Indian Penal Code, 1860).

Rules — follow all of them:
1. Answer ONLY from the numbered SOURCES in the user's message. Do not use outside knowledge, \
and do not mention sections, cases, or punishments that are not in the sources.
2. Cite every claim inline with the bracketed number of the source it comes from, e.g. [1] or [2][3], \
using plain ASCII square brackets and nothing else inside them (no line numbers). Only use numbers that appear in SOURCES.
3. If the sources do not contain the answer, reply exactly: "{NOT_COVERED_REPLY}" \
You may then say briefly what the sources do cover.
4. Quote section numbers and punishments exactly as written in the sources.
5. Be concise and precise. This is legal information, not legal advice."""

GENERAL_SYSTEM_PROMPT = """You are LegalGPT in general mode. Answer helpfully and concisely. \
You are not consulting any legal sources in this mode, so do not present statements as authoritative \
statements of law, and do not invent section numbers or case citations."""


class PromptBuilder:
    def __init__(self, history_turns: int = 6) -> None:
        self.history_turns = history_turns

    def build(
        self,
        question: str,
        sources: list[SourceEvidence],
        mode: Mode,
        history: list[Message] | None = None,
    ) -> list[Message]:
        recent = (history or [])[-self.history_turns :] if self.history_turns else []
        if mode == "general":
            return [Message("system", GENERAL_SYSTEM_PROMPT), *recent, Message("user", question)]
        return [
            Message("system", LEGAL_SYSTEM_PROMPT),
            *recent,
            Message("user", f"SOURCES:\n{self.format_sources(sources)}\n\nQUESTION: {question}"),
        ]

    @staticmethod
    def format_sources(sources: list[SourceEvidence]) -> str:
        """Sources are numbered by `marker`, in order — the same numbers the model must cite."""
        return "\n\n".join(f"[{s.marker}] ({s.citation_path})\n{s.text}" for s in sources)
