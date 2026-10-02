"""ConfidenceScorer: how well did the retrieved evidence match the question?

IMPORTANT — what this score is and is not:
    It measures how closely the retrieved legal text matches the *question* (embedding
    similarity between question and sources). It does NOT measure whether the *answer* is
    correct, complete or legally sound. A High band means "we found closely matching IPC text";
    the model can still misread it. Use CitationValidator and AttributionAnalyzer to check the
    answer against that text, and a lawyer to check the law.
"""

from dataclasses import dataclass
from typing import Literal

Band = Literal["High", "Medium", "Low"]

BAND_NOTE = (
    "Evidence match measures how closely the retrieved legal text matches your question. "
    "It is not a measure of whether the answer is correct."
)


@dataclass(frozen=True)
class ConfidenceScorer:
    """Bands default to thresholds calibrated on this system (MiniLM + IPC chunks; see
    backend/scripts/calibrate_retrieval.py): answerable questions had top-1 similarity
    0.65–0.86 (median 0.73), and the retrieval floor is 0.55 — so the generic 0.75/0.5 split
    would call most good matches "Medium"."""

    high: float = 0.70
    medium: float = 0.60

    def __post_init__(self) -> None:
        if not 0 <= self.medium <= self.high <= 1:
            raise ValueError("thresholds must satisfy 0 <= medium <= high <= 1")

    @staticmethod
    def similarity(distance: float, space: str = "cosine") -> float:
        """Normalize a raw ChromaDB distance to a [0, 1] similarity for the collection's metric.

        - cosine: distance = 1 - cos               -> similarity = 1 - distance
        - ip:     distance = 1 - dot (unit vectors) -> similarity = 1 - distance
        - l2:     distance = squared L2 = 2 - 2cos (unit vectors) -> similarity = 1 - distance / 2
        Negative cosines (opposed vectors) are clamped to 0: for retrieval they mean "no match".
        """
        if space in ("cosine", "ip"):
            sim = 1.0 - distance
        elif space == "l2":
            sim = 1.0 - distance / 2.0
        else:
            raise ValueError(f"unknown distance space {space!r}")
        return min(1.0, max(0.0, sim))

    def band(self, score: float) -> Band:
        """'High' / 'Medium' / 'Low' evidence match. Not an accuracy guarantee (see module doc)."""
        if score >= self.high:
            return "High"
        if score >= self.medium:
            return "Medium"
        return "Low"
