"""ReviewRouter: JSON API behind the Review tab (/api/review/...)."""

import hashlib
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.annotation.collector import JudgmentCollector
from app.labeling_assistant.intake import MAX_BYTES, JudgmentIntake, UploadRejected
from app.labeling_assistant.review_queue import HumanReviewQueue, QueueCase, ReviewIncomplete
from app.labeling_assistant.titles import detect_title


def is_blind_sample(case_id: str, rate: float) -> bool:
    """About `rate` of all judgments, chosen by a stable hash of the case id (never by the model or
    a person), are labelled blind first - suggestions hidden - so the model's accuracy can be measured
    on labels it did not influence."""
    return int(hashlib.sha1(case_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < rate


class ReviewerBody(BaseModel):
    reviewer: str = Field(min_length=1, max_length=40)


class CorrectionBody(ReviewerBody):
    label: str


class ReviewRouter:
    def __init__(self, queue: HumanReviewQueue, collector: JudgmentCollector, intake: JudgmentIntake | None = None,
                 blind_rate: float = 0.1) -> None:
        self.queue = queue
        self.blind_rate = blind_rate
        self.collector = collector
        self.intake = intake
        self.router = APIRouter(prefix="/api/review", tags=["review"])
        self.router.add_api_route("/cases", self.list_cases, methods=["GET"])
        self.router.add_api_route("/cases/{case_id}", self.get_case, methods=["GET"])
        self.router.add_api_route("/cases/{case_id}/items/{sentence_id}", self.correct, methods=["PUT"])
        self.router.add_api_route("/cases/{case_id}/sign-off", self.sign_off, methods=["POST"])
        self.router.add_api_route("/cases/{case_id}/items/{sentence_id}/explain", self.explain, methods=["GET"])
        self.router.add_api_route("/uploads", self.upload, methods=["POST"], status_code=202)
        self.router.add_api_route("/uploads", self.list_uploads, methods=["GET"])
        self.router.add_api_route("/uploads/{job_id}", self.get_upload, methods=["GET"])

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
            "blind_sample": is_blind_sample(case.case_id, self.blind_rate),
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

    # ------------------------------------------------------------ explanations

    def explain(self, case_id: str, sentence_id: str, label: str | None = None) -> dict[str, Any]:
        """Why the classifier suggests a role for this sentence: word attributions (Integrated
        Gradients), reliance on the context, and similar human-labelled sentences. Uses the newest
        checkpoint, which may differ from the one that made the queued suggestion."""
        from app.labeling_assistant.dataset import RRLExample
        from app.labeling_assistant.explain import ClassifierExplainer

        case = self._load(case_id)
        examples = [RRLExample(it.prediction.sentence_id, case_id, it.prediction.idx, it.prediction.text)
                    for it in case.items]
        target = next((e for e in examples if e.sentence_id == sentence_id), None)
        if target is None:
            raise HTTPException(404, f"sentence {sentence_id!r} is not in this case")
        clf, model = self._intake()._classifier()
        if getattr(self, "_explainer_model", None) != model:
            from app.labeling_assistant.intake import latest_checkpoint

            ckpt = latest_checkpoint(self._intake().model_dir)
            self._explainer = ClassifierExplainer(clf, ckpt / "neighbours.npz" if ckpt else None)
            self._explainer_model = model
        if label is not None and label not in clf.labels:
            raise HTTPException(422, f"unknown label {label!r}")
        item = next(it for it in case.items if it.prediction.sentence_id == sentence_id)
        out = self._explainer.explain(target, examples, label)
        return {**out, "model": model, "queued_suggestion": item.prediction.label, "queued_model": case.model}

    # ------------------------------------------------------------ uploads (JudgmentIntake)

    def _intake(self) -> JudgmentIntake:
        if self.intake is None:
            raise HTTPException(503, "uploads are not available on this server")
        return self.intake

    async def upload(self, request: Request, filename: str, uploader: str = "") -> dict[str, Any]:
        """The request body is the file itself (no multipart): the browser sends the File object."""
        intake = self._intake()
        if int(request.headers.get("content-length") or 0) > MAX_BYTES:
            raise HTTPException(413, f"{filename}: larger than {MAX_BYTES // (1024 * 1024)} MB")
        data = await request.body()
        try:
            return intake.submit(filename, data, uploader).to_dict()
        except UploadRejected as e:
            raise HTTPException(422, str(e)) from None

    def list_uploads(self) -> list[dict[str, Any]]:
        return [j.to_dict() for j in self._intake().jobs()]

    def get_upload(self, job_id: str) -> dict[str, Any]:
        try:
            return self._intake().job(job_id).to_dict()
        except KeyError:
            raise HTTPException(404, f"no upload {job_id!r}") from None
