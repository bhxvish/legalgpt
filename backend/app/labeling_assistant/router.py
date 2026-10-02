"""ReviewRouter: JSON API behind the Review tab (/api/review/...)."""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.annotation.collector import JudgmentCollector
from app.labeling_assistant.review_queue import HumanReviewQueue, QueueCase, ReviewIncomplete
from app.labeling_assistant.titles import detect_title


class ReviewerBody(BaseModel):
    reviewer: str = Field(min_length=1, max_length=40)


class CorrectionBody(ReviewerBody):
    label: str


class ReviewRouter:
    def __init__(self, queue: HumanReviewQueue, collector: JudgmentCollector) -> None:
        self.queue = queue
        self.collector = collector
        self.router = APIRouter(prefix="/api/review", tags=["review"])
        self.router.add_api_route("/cases", self.list_cases, methods=["GET"])
        self.router.add_api_route("/cases/{case_id}", self.get_case, methods=["GET"])
        self.router.add_api_route("/cases/{case_id}/items/{sentence_id}", self.correct, methods=["PUT"])
        self.router.add_api_route("/cases/{case_id}/sign-off", self.sign_off, methods=["POST"])

    def _load(self, case_id: str) -> QueueCase:
        try:
            return self.queue.load(case_id)
        except KeyError:
            raise HTTPException(404, f"case {case_id!r} is not in the review queue") from None

    def _summary(self, case: QueueCase) -> dict[str, Any]:
        flagged = [it for it in case.items if it.needs_review]
        try:
            record = self.collector.load(case.case_id)
            title = record.title or detect_title(record.raw_text)
        except KeyError:
            title = ""
        return {
            "case_id": case.case_id,
            "title": title,
            "status": case.status,
            "model": case.model,
            "sentences": len(case.items),
            "flagged": len(flagged),
            "reviewed": sum(it.status != "pending" for it in flagged),
            "corrected": sum(it.status == "corrected" for it in case.items),
            "audit_error_rate": self.queue.audit_error_rate(case.case_id),
            "max_audit_error": self.queue.max_audit_error,
            "tau_conf": case.tau_conf,
            "decision_note": case.decision_note,
        }

    def list_cases(self) -> list[dict[str, Any]]:
        return [self._summary(self.queue.load(c)) for c in self.queue.case_ids()]

    def get_case(self, case_id: str) -> dict[str, Any]:
        case = self._load(case_id)
        return {
            **self._summary(case),
            "items": [
                {
                    "sentence_id": it.prediction.sentence_id,
                    "idx": it.prediction.idx,
                    "text": it.prediction.text,
                    "predicted": it.prediction.label,
                    "confidence": it.prediction.confidence,
                    "probabilities": it.prediction.probabilities,
                    "kind": it.kind,
                    "status": it.status,
                    "final_label": it.final_label,
                    "reviewer": it.reviewer,
                }
                for it in case.items
            ],
        }

    def correct(self, case_id: str, sentence_id: str, body: CorrectionBody) -> dict[str, Any]:
        if not sentence_id.startswith(f"{case_id}:"):
            raise HTTPException(422, "sentence does not belong to this case")
        self._load(case_id)
        try:
            item = self.queue.apply_correction(sentence_id, body.label, body.reviewer)
        except KeyError as e:
            raise HTTPException(404, str(e)) from None
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        return {"sentence_id": sentence_id, "status": item.status, "final_label": item.final_label,
                **{k: v for k, v in self._summary(self.queue.load(case_id)).items() if k in ("reviewed", "flagged", "audit_error_rate", "corrected")}}

    def sign_off(self, case_id: str, body: ReviewerBody) -> dict[str, Any]:
        self._load(case_id)
        try:
            result = self.queue.decide(case_id, body.reviewer)
        except ReviewIncomplete as e:
            raise HTTPException(409, str(e)) from None
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        return {"promoted": result.promoted, "status": result.status,
                "audit_error_rate": result.audit_error_rate, "reason": result.reason}
