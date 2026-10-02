"""ExplanationBuilder: the façade ChatRouter calls after generation.

Everything here comes from artifacts the chat flow already has — the retrieved evidence with its
similarities, the answer text, the model id — not from post-hoc feature attribution.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.models import SourceEvidence
from app.core.prompt_builder import NOT_COVERED_REPLY
from app.explainability.attribution_analyzer import Attribution, AttributionAnalyzer
from app.explainability.citation_validator import CitationCheck, CitationValidator
from app.explainability.confidence_scorer import BAND_NOTE, Band, ConfidenceScorer


@dataclass
class SourceView:
    marker: int
    citation_path: str
    section: str
    title: str
    similarity: float
    cited: bool  # the answer cites [marker]
    supports: list[int]  # answer sentences whose best match is this source


@dataclass
class ExplanationRecord:
    question: str
    model_id: str
    refused: bool
    relevance: float  # best question–evidence similarity (also for refusals)
    band: Band
    band_note: str  # what the band does NOT mean
    retrieval_floor: float
    sources: list[SourceView]
    citation_check: CitationCheck | None
    attributions: list[Attribution]
    support_ratio: float | None
    support_threshold: float
    warnings: list[str] = field(default_factory=list)  # plain-language red flags, most serious first

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExplanationBuilder:
    def __init__(
        self,
        analyzer: AttributionAnalyzer,
        scorer: ConfidenceScorer | None = None,
        validator: CitationValidator | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.scorer = scorer or ConfidenceScorer()
        self.validator = validator or CitationValidator()

    def build(
        self,
        question: str,
        sources: Sequence[SourceEvidence],
        answer: str,
        model_id: str,
        best_similarity: float | None = None,
        retrieval_floor: float = 0.0,
        refused: bool = False,
    ) -> ExplanationRecord:
        """`best_similarity` is the retriever's best candidate score; for a refusal (no sources
        cleared the floor) it is the only relevance evidence there is."""
        relevance = max((s.similarity for s in sources), default=best_similarity or 0.0)
        band = self.scorer.band(relevance)
        if refused or not sources:
            return ExplanationRecord(
                question, model_id, True, round(relevance, 4), band, BAND_NOTE, retrieval_floor, [], None, [], None,
                self.analyzer.support_threshold,
                [f"No indexed IPC text matched the question closely enough (best match {relevance:.2f}, "
                 f"needed {retrieval_floor:.2f}), so no answer was generated."],
            )

        check = self.validator.validate(answer, sources)
        # The "I don't know" sentence is a statement about the sources, not a claim drawn from
        # them; attributing it would flag a correct refusal as an unsupported answer.
        claims = re.sub(re.escape(NOT_COVERED_REPLY), " ", answer, flags=re.I)
        attributions = self.analyzer.attribute(claims, sources)
        ratio = self.analyzer.support_ratio(attributions) if attributions else None  # nothing to attribute
        supports: dict[int, list[int]] = {s.marker: [] for s in sources}
        for a in attributions:
            if a.supported and a.source_marker in supports:
                supports[a.source_marker].append(a.sentence_idx)
        views = [
            SourceView(s.marker, s.citation_path, s.section, s.title, round(s.similarity, 4),
                       s.marker in set(check.markers), supports[s.marker])
            for s in sources
        ]
        not_covered = NOT_COVERED_REPLY.lower() in answer.lower()
        return ExplanationRecord(
            question, model_id, not_covered, round(relevance, 4), band, BAND_NOTE, retrieval_floor, views, check,
            attributions, ratio, self.analyzer.support_threshold,
            self._warnings(band, check, attributions, not_covered),
        )

    @staticmethod
    def _warnings(band: Band, check: CitationCheck, attributions: Sequence[Attribution], not_covered: bool) -> list[str]:
        w: list[str] = []
        if check.unresolved:
            w.append(f"The answer cites {', '.join(f'[{m}]' for m in check.unresolved)}, which "
                     f"{'is' if len(check.unresolved) == 1 else 'are'} not among the retrieved sources.")
        unsupported = [a for a in attributions if not a.supported]
        if unsupported:
            w.append(f"{len(unsupported)} of {len(attributions)} answer sentences are not supported by any retrieved source.")
        mis = [a for a in attributions if a.supported and a.cited_supported is False]
        if mis:
            w.append(f"{len(mis)} sentence(s) cite a source that does not support them (a different source does).")
        if not check.markers and not not_covered:
            w.append("The answer cites no sources, so its claims cannot be traced to the retrieved text.")
        if not_covered and band != "Low":
            w.append(f"The model said the sources do not cover this, but the retrieved text matches the question "
                     f"({band.lower()} evidence match). Open the sources to check whether the answer is actually there.")
        if band == "Low":
            w.append("The retrieved legal text only loosely matches the question.")
        return w

    @staticmethod
    def to_sse_event(rec: ExplanationRecord) -> str:
        return f"event: explanation\ndata: {json.dumps(rec.to_dict(), ensure_ascii=False)}\n\n"
