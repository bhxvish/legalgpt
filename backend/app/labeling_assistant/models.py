"""Light-weight records shared by the classifier and the review queue (no torch import, so the
web app can serve the review queue without loading the ML stack)."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Prediction:
    sentence_id: str
    label: str
    probabilities: dict[str, float]
    confidence: float  # max probability
    margin: float  # top probability minus the second
    case_id: str = ""
    idx: int = -1
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Prediction":
        return cls(**d)
