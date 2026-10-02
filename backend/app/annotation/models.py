"""Records for Module 1 (dataset pipeline)."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

SelectionStatus = Literal["pending", "eligible", "rejected"]
LabelSource = Literal["manual", "bert_assisted"]


@dataclass
class JudgmentRecord:
    case_id: str
    court: str
    decision_year: int | None
    raw_text: str
    sections_cited: list[str] = field(default_factory=list)
    selection_status: SelectionStatus = "pending"
    rejection_reason: str = ""
    source_path: str = ""
    sha256: str = ""  # of the normalized text; used for duplicate detection

    def index_entry(self) -> dict[str, Any]:
        """Everything except the text, for data/raw_judgments/index.jsonl."""
        entry = asdict(self)
        entry.pop("raw_text")
        return entry


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class LabeledSentence:
    """One label decision. AnnotationStore keeps every decision (append-only); the latest one
    for a sentence wins. `annotator` (an addition to the LLD's fields) is what makes double
    annotation and Cohen's kappa possible."""

    sentence_id: str
    case_id: str
    idx: int
    text: str
    label: str
    source: LabelSource = "manual"
    reviewed: bool = False
    annotator: str = ""
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LabeledSentence":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def sentence_id(case_id: str, idx: int) -> str:
    return f"{case_id}:{idx:04d}"
