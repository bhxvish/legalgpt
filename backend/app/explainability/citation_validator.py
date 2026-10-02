"""CitationValidator: do the answer's [n] markers point at sources that were actually retrieved?"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.core.citations import normalize_stream
from app.core.models import SourceEvidence

_MARKER = re.compile(r"\[(\d+)\]")


@dataclass
class CitationCheck:
    markers: list[int]  # every marker in order of appearance (repeats kept)
    resolved: list[int]  # distinct markers that match a retrieved source
    unresolved: list[int]  # distinct markers with no such source: the model cited something not retrieved
    uncited_sources: list[int] = field(default_factory=list)  # retrieved but never cited

    @property
    def ok(self) -> bool:
        return bool(self.markers) and not self.unresolved


class CitationValidator:
    @staticmethod
    def extract_markers(answer: str) -> list[int]:
        """Markers in the format PromptBuilder asks for ("[1]", "[2][3]"); variant brackets the
        models emit (【1】, [1†L2-L4]) are canonicalized first, as in the chat stream."""
        return [int(m) for m in _MARKER.findall("".join(normalize_stream([answer])))]

    def validate(self, answer: str, sources: Sequence[SourceEvidence]) -> CitationCheck:
        markers = self.extract_markers(answer)
        available = {s.marker for s in sources}
        distinct = list(dict.fromkeys(markers))
        return CitationCheck(
            markers=markers,
            resolved=[m for m in distinct if m in available],
            unresolved=[m for m in distinct if m not in available],
            uncited_sources=sorted(available - set(markers)),
        )
