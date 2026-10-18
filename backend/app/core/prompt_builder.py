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
3. Questions that describe what happened: the sources are statute text, so they never mention the \
specific facts (phones, apps, places, people). Applying an offence defined in the sources to the \
described facts IS answering from the sources. Name the offence and its section, say which of its \
elements the facts appear to satisfy, and say which facts would still need to be established (for \
example, dishonest intention). Do not refuse because the facts themselves are not in the sources.
4. Only if no source defines an offence or rule that could apply to the question, reply exactly: \
"{NOT_COVERED_REPLY}" You may then say briefly what the sources do cover.
5. Quote section numbers and punishments exactly as written in the sources.
6. Be concise and precise. This is legal information, not legal advice.
7. Sources marked JUDGMENT are passages of real court decisions, not statute. Cite one only for what \
that court said or decided, and name the case (e.g. "In X v. Y (SC, 2023), the court held ... [4]"). \
Never present a judgment as the text of a section, never name a case that is not in SOURCES, and \
do not use a judgment that is about a different offence."""

GENERAL_SYSTEM_PROMPT = """You are LegalGPT in general mode. Answer helpfully and concisely. \
You are not consulting any legal sources in this mode, so do not present statements as authoritative \
statements of law, and do not invent section numbers or case citations."""


BNS_NAME = ("the Bharatiya Nyaya Sanhita, 2023 (BNS), which replaced the Indian Penal Code from 1 July 2024; "
            "give BNS section numbers")


def legal_system_prompt(law: str = "ipc") -> str:
    """The legal-mode system prompt for the IPC (default) or the BNS."""
    if law != "bns":
        return LEGAL_SYSTEM_PROMPT
    return LEGAL_SYSTEM_PROMPT.replace("the Indian Penal Code, 1860", BNS_NAME, 1)


class PromptBuilder:
    def __init__(self, history_turns: int = 6) -> None:
        self.history_turns = history_turns

    def build(
        self,
        question: str,
        sources: list[SourceEvidence],
        mode: Mode,
        history: list[Message] | None = None,
        law: str = "ipc",
    ) -> list[Message]:
        recent = (history or [])[-self.history_turns :] if self.history_turns else []
        if mode == "general":
            return [Message("system", GENERAL_SYSTEM_PROMPT), *recent, Message("user", question)]
        return [
            Message("system", legal_system_prompt(law)),
            *recent,
            Message("user", f"SOURCES:\n{self.format_sources(sources)}\n\nQUESTION: {question}"),
        ]

    @staticmethod
    def format_sources(sources: list[SourceEvidence]) -> str:
        """Sources are numbered by `marker`, in order — the same numbers the model must cite."""
        def head(s: SourceEvidence) -> str:
            if s.kind == "judgment":
                court = s.court or "court not recorded"  # so the model cannot guess "the Supreme Court"
                return f"[{s.marker}] (JUDGMENT: {s.citation_path}; {court}; {s.role} passage)"
            return f"[{s.marker}] ({s.citation_path})"

        return "\n\n".join(f"{head(s)}\n{s.text}" for s in sources)
