"""IpcQaBuilder: grounded question-answer examples built from the IPC itself.

The v0.2 adapter only ever saw judgment sentences as sources (plus many refusals), so faced with the
IPC sections the chat actually retrieves it answered "I don't know" almost every time. These examples
use exactly the inference setting: a short question, the sources the real Retriever returns for it
(the right section plus its neighbours), and an answer taken from that section's own text with its
[n] marker - correct by construction, no hosted model needed. About 10% are refusals where the asked
section is genuinely not among the sources.

EVAL_SECTIONS are never used for training, so a comparison on them measures generalisation.
"""

import random
import re
from collections.abc import Sequence

from app.core.models import Chunk
from app.core.prompt_builder import LEGAL_SYSTEM_PROMPT, NOT_COVERED_REPLY, PromptBuilder
from app.finetuning.instruction_builder import InstructionExample

EVAL_SECTIONS = (
    "279", "302", "304", "304A", "304B", "306", "307", "323", "324", "326", "337", "338", "341", "342", "354",
    "363", "379", "380", "383", "392", "406", "411", "420", "447", "457", "498A", "500", "506", "509", "376",
)
_HEADING = re.compile(r"^Section\s+\S+\.\s*[^\n]*\n")
_SENTENCE_END = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")


def main_chunk(chunks: Sequence[Chunk], section: str) -> Chunk | None:
    """The section's own text (not its State Amendments or Illustrations)."""
    for c in chunks:
        if c.section == section and c.citation_path.endswith(f"Section {section}"):
            return c
    return None


def body_of(chunk: Chunk, max_chars: int = 520) -> str:
    """The provision's text without its heading line, cut at a sentence end within max_chars."""
    text = re.sub(r"\s+", " ", _HEADING.sub("", chunk.text, count=1)).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    ends = [m.start() for m in _SENTENCE_END.finditer(cut)]
    return (cut[: ends[-1]] if ends else cut.rsplit(" ", 1)[0]).rstrip(" ;,") + " …"


_OFFENCE = re.compile(r"^Whoever(?P<comma>,?)\s+(?P<conduct>.+?),?\s+shall be punished with (?P<punishment>[^.;]+)", re.S)
_PLAIN = (  # statutory punishment wording -> plain English, applied in order
    (r"imprisonment of either description", "imprisonment (simple or rigorous)"),
    (r"for a term which shall not be less than ([\w ]+?) but which may extend to", r"of at least \1 and up to"),
    (r"for a term which may extend to", "of up to"),
    (r"for a term of not less than", "of at least"),
    (r",? and shall also be liable to fine", ", and a fine"),
    (r",? and shall be liable to fine", ", and a fine"),
    (r"or with fine which may extend to", "or a fine of up to"),
    (r"or with fine", "or a fine"),
    (r"or with both", "or both"),
    (r"or with imprisonment", "or imprisonment"),
    (r"with death", "death"),
)


def plain_punishment(text: str) -> str:
    out = re.sub(r"\s+", " ", text).strip().rstrip(",")
    for pattern, repl in _PLAIN:
        out = re.sub(pattern, repl, out)
    return out


def offence_parts(chunk: Chunk, max_conduct: int = 420) -> tuple[str, str] | None:
    """(conduct, plain punishment) for sections of the form "Whoever <conduct>, shall be punished
    with <punishment>", else None."""
    m = _OFFENCE.match(body_of(chunk, max_chars=4000))
    if not m:
        return None
    conduct = (", " if m.group("comma") else " ") + re.sub(r"\s+", " ", m.group("conduct")).strip()  # "who, being ..."
    if len(conduct) > max_conduct:
        conduct = conduct[:max_conduct].rsplit(" ", 1)[0].rstrip(",;") + " …"
    return conduct, plain_punishment(m.group("punishment"))


def lower_first(s: str) -> str:
    return s[:1].lower() + s[1:] if s[:1].isupper() and not s[1:2].isupper() else s


class IpcQaBuilder:
    def __init__(self, retriever, chunks: Sequence[Chunk], seed: int = 13, refusal_share: float = 0.1,
                 plain: bool = True) -> None:
        """`plain`: explain offence sections in plain words (v0.4); False reproduces v0.3's quoting."""
        self.retriever = retriever
        self.plain = plain
        self.chunks = list(chunks)
        self.rng = random.Random(seed)
        self.refusal_share = refusal_share

    def question(self, section: str, title: str, body: str) -> tuple[str, str]:
        """(question, kind) - kind "punishment" or "provision"."""
        t = title.strip().rstrip(".")
        if "shall be punished" in body and self.rng.random() < 0.6:
            topic = re.sub(r"^punishment for\s+", "", t, flags=re.I)
            if topic.lower() != t.lower():
                return f"What is the punishment for {lower_first(topic)}?", "punishment"
            return f"What is the punishment under Section {section} of the IPC?", "punishment"
        return self.rng.choice([
            f"What does Section {section} of the IPC provide?",
            f"Explain Section {section} of the Indian Penal Code.",
            f"What does the IPC say about {lower_first(t)}?",
        ]), "provision"

    @staticmethod
    def answer(section: str, title: str, body: str, marker: int, kind: str, parts: tuple[str, str] | None = None) -> str:
        """Plain-language answer built from the section's own text (correct by construction).
        Offence sections ("Whoever ..., shall be punished with ...") are explained: who commits the
        offence and the punishment in plain words, each cited. Other sections are quoted."""
        t = title.strip().rstrip(".")
        if parts is not None:
            conduct, punishment = parts
            if kind == "punishment":
                return (f"Under Section {section} of the IPC ({t}) [{marker}], the punishment is {punishment}.\n"
                        f"It applies to anyone who{conduct} [{marker}].")
            return (f"Section {section} of the IPC ({t}) [{marker}] makes it an offence for anyone who{conduct}.\n"
                    f"Punishment: {punishment} [{marker}].")
        if kind == "punishment" and body.lower().startswith("whoever"):
            return f"Under Section {section} of the IPC [{marker}], {lower_first(body)}"
        return f"Section {section} of the IPC ({t}) provides: {body} [{marker}]"

    def build(self, sections: Sequence[str]) -> list[InstructionExample]:
        held_out = set(EVAL_SECTIONS)
        cases: list[tuple[str, str, list]] = []  # (section, question, sources)
        out: list[InstructionExample] = []
        for section in sections:
            if section in held_out:
                continue
            chunk = main_chunk(self.chunks, section)
            if chunk is None or not chunk.title:
                continue
            body = body_of(chunk)
            if len(body) < 40:
                continue
            q, kind = self.question(section, chunk.title, body)
            sources = self.retriever.retrieve(q)
            hit = next((s for s in sources if s.chunk_id == chunk.chunk_id), None)
            if hit is None:  # the retriever would not give the model the right section: skip
                continue
            cases.append((section, q, sources))
            parts = offence_parts(chunk) if self.plain else None
            out.append(self._example(section, q, sources, self.answer(section, chunk.title, body, hit.marker, kind, parts),
                                     "ipc_answer"))
        n_refusals = round(len(out) * self.refusal_share / (1 - self.refusal_share))
        for section, q, _ in self.rng.sample(cases, min(n_refusals, len(cases))):
            # sources retrieved for a different section's question that do not contain this section
            for _ in range(10):
                other_section, _, other_sources = self.rng.choice(cases)
                if other_section != section and all(s.section != section for s in other_sources):
                    out.append(self._example(section, q, other_sources, NOT_COVERED_REPLY, "ipc_refusal"))
                    break
        return out

    @staticmethod
    def _example(section: str, question: str, sources, output: str, task: str) -> InstructionExample:
        return InstructionExample(
            instruction=f"SOURCES:\n{PromptBuilder.format_sources(sources)}\n\nQUESTION: {question}",
            input="",
            output=output,
            source_case_id=f"ipc:{section}",
            task=task,
            system=LEGAL_SYSTEM_PROMPT,
        )
