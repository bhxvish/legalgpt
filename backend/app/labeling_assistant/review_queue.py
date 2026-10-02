"""HumanReviewQueue: humans check flagged model predictions before they enter the corpus.

One JSON file per case under `<annotation_store>/review_queue/`. A case moves
open -> promoted (labels appended to the AnnotationStore with source="bert_assisted")
     -> escalated (too many audited predictions were wrong: the case goes to full manual
        annotation in the Annotate tab and nothing from the model is promoted).
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from app.annotation.label_scheme import LabelScheme
from app.annotation.models import LabeledSentence, utc_now
from app.annotation.store import AnnotationStore
from app.labeling_assistant.models import Prediction
from app.labeling_assistant.review_selector import ReviewBatch

ItemKind = Literal["mandatory", "audit", "auto"]
ItemStatus = Literal["pending", "accepted", "corrected"]
CaseStatus = Literal["open", "promoted", "escalated"]


@dataclass
class QueueItem:
    prediction: Prediction
    kind: ItemKind
    status: ItemStatus = "pending"
    final_label: str | None = None
    reviewer: str = ""
    reviewed_at: str = ""

    @property
    def needs_review(self) -> bool:
        return self.kind != "auto"


@dataclass
class QueueCase:
    case_id: str
    items: list[QueueItem]
    model: str = ""
    tau_conf: float = 0.0
    audit_rate: float = 0.0
    status: CaseStatus = "open"
    created_at: str = field(default_factory=utc_now)
    decided_at: str = ""
    decided_by: str = ""
    decision_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QueueCase":
        items = [QueueItem(**{**it, "prediction": Prediction.from_dict(it["prediction"])}) for it in d["items"]]
        return cls(**{**d, "items": items})


@dataclass
class SignOffResult:
    promoted: bool
    status: CaseStatus
    audit_error_rate: float
    reason: str


class ReviewIncomplete(ValueError):
    """Sign-off was attempted before every flagged prediction was reviewed."""


class HumanReviewQueue:
    def __init__(self, store: AnnotationStore, max_audit_error: float = 0.2) -> None:
        self.store = store
        self.max_audit_error = max_audit_error
        self.root = store.root / "review_queue"

    # ------------------------------------------------------------ persistence

    def _path(self, case_id: str) -> Path:
        return self.root / f"{case_id}.json"

    def load(self, case_id: str) -> QueueCase:
        path = self._path(case_id)
        if not path.exists():
            raise KeyError(f"no review queue for case {case_id!r}")
        return QueueCase.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _save(self, case: QueueCase) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self._path(case.case_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(case.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self._path(case.case_id))  # atomic: a crash never leaves half a file

    def case_ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json")) if self.root.exists() else []

    # ---------------------------------------------------------------- queue

    def enqueue(self, batch: ReviewBatch, model: str = "") -> QueueCase:
        if self._path(batch.case_id).exists():
            raise FileExistsError(f"case {batch.case_id!r} is already queued")
        items = (
            [QueueItem(p, "mandatory") for p in batch.mandatory]
            + [QueueItem(p, "audit") for p in batch.audit]
            + [QueueItem(p, "auto") for p in batch.auto]
        )
        items.sort(key=lambda it: it.prediction.idx)
        if self.store.load_segments(batch.case_id) is None:
            # pin now, so an escalated case is annotated against exactly these sentences
            self.store.save_segments(batch.case_id, [it.prediction.text for it in items], segmenter=f"review:{model}")
        case = QueueCase(batch.case_id, items, model=model, tau_conf=batch.tau_conf, audit_rate=batch.audit_rate)
        self._save(case)
        return case

    def apply_correction(self, sentence_id: str, label: str, reviewer: str) -> QueueItem:
        """Record the reviewer's label. Giving the model's own label counts as accepting it."""
        if not LabelScheme.validate(label):
            raise ValueError(f"unknown label {label!r}")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        case = self.load(sentence_id.rsplit(":", 1)[0])
        if case.status != "open":
            raise ValueError(f"case {case.case_id} is already {case.status}")
        item = next((it for it in case.items if it.prediction.sentence_id == sentence_id), None)
        if item is None:
            raise KeyError(f"sentence {sentence_id!r} is not in the queue")
        item.final_label = label
        item.status = "accepted" if label == item.prediction.label else "corrected"
        item.reviewer, item.reviewed_at = reviewer.strip(), utc_now()
        self._save(case)
        return item

    def audit_error_rate(self, case_id: str) -> float:
        """Share of reviewed audit items (confident predictions sampled for checking) that the
        reviewer had to correct. 0.0 when there is nothing audited yet."""
        audited = [it for it in self.load(case_id).items if it.kind == "audit" and it.status != "pending"]
        return sum(it.status == "corrected" for it in audited) / len(audited) if audited else 0.0

    def sign_off(self, case_id: str, reviewer: str) -> bool:
        return self.decide(case_id, reviewer).promoted

    def decide(self, case_id: str, reviewer: str) -> SignOffResult:
        """sign_off() with the reason. Requires every flagged item to be reviewed. Escalates the
        case if audit_error_rate exceeds max_audit_error, otherwise promotes all its labels."""
        case = self.load(case_id)
        if case.status != "open":
            raise ValueError(f"case {case_id} is already {case.status}")
        pending = [it for it in case.items if it.needs_review and it.status == "pending"]
        if pending:
            raise ReviewIncomplete(f"{len(pending)} flagged sentence(s) still need review")
        rate = self.audit_error_rate(case_id)
        case.decided_at, case.decided_by = utc_now(), reviewer.strip()
        if rate > self.max_audit_error:
            case.status = "escalated"
            case.decision_note = (
                f"{rate:.0%} of the spot-checked confident predictions were wrong (limit "
                f"{self.max_audit_error:.0%}), so the model's unchecked labels can't be trusted for this "
                "case. Nothing was promoted; annotate it fully in the Annotate tab."
            )
            self._save(case)
            return SignOffResult(False, case.status, rate, case.decision_note)

        if self.store.load_case(case_id):
            raise ValueError(f"case {case_id} already has labels in the annotation store")
        if self.store.load_segments(case_id) is None:  # normally pinned at enqueue time
            self.store.save_segments(case_id, [it.prediction.text for it in case.items], segmenter=f"review:{case.model}")
        self.store.append_many([
            LabeledSentence(
                sentence_id=it.prediction.sentence_id,
                case_id=case_id,
                idx=it.prediction.idx,
                text=it.prediction.text,
                label=it.final_label or it.prediction.label,
                source="bert_assisted",
                reviewed=it.status != "pending",
                annotator=it.reviewer or f"model:{case.model}",
            )
            for it in case.items
        ])
        case.status = "promoted"
        reviewed = sum(it.status != "pending" for it in case.items)
        case.decision_note = (
            f"Promoted {len(case.items)} sentences ({reviewed} human-reviewed, "
            f"{len(case.items) - reviewed} accepted from the model); audit error rate {rate:.0%}."
        )
        self._save(case)
        return SignOffResult(True, case.status, rate, case.decision_note)
