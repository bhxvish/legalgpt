"""AnnotationRouter: JSON API behind the annotation UI (/api/annotation/...)."""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.annotation.collector import JudgmentCollector
from app.annotation.label_scheme import LabelScheme
from app.annotation.models import LabeledSentence, sentence_id
from app.annotation.segmenter import SentenceSegmenter
from app.annotation.store import AnnotationStore

SEGMENTER_VERSION = "judgment-v1"


class LabelItem(BaseModel):
    idx: int = Field(ge=0)
    label: str

    @field_validator("label")
    @classmethod
    def _known(cls, v: str) -> str:
        if not LabelScheme.validate(v):
            raise ValueError(f"unknown label {v!r}")
        return v


class SaveLabelsRequest(BaseModel):
    annotator: str = Field(min_length=1, max_length=40)
    labels: list[LabelItem] = Field(min_length=1, max_length=5000)

    @field_validator("annotator")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("annotator is required")
        return v


class SentenceOut(BaseModel):
    idx: int
    sentence_id: str
    text: str
    label: str | None = None


class Progress(BaseModel):
    sentences: int
    labelled: int


class CaseSummary(BaseModel):
    case_id: str
    title: str = ""
    court: str
    decision_year: int | None
    sections_cited: list[str]
    progress: Progress | None = None  # for the requesting annotator, once the case was opened


class CaseDetail(CaseSummary):
    sentences: list[SentenceOut]


class AnnotationRouter:
    def __init__(self, collector: JudgmentCollector, store: AnnotationStore, segmenter: SentenceSegmenter | None = None) -> None:
        self.collector = collector
        self.store = store
        self.segmenter = segmenter or SentenceSegmenter()
        self.router = APIRouter(prefix="/api/annotation", tags=["annotation"])
        self.router.add_api_route("/scheme", self.scheme, methods=["GET"])
        self.router.add_api_route("/cases", self.list_cases, methods=["GET"], response_model=list[CaseSummary])
        self.router.add_api_route("/cases/{case_id}", self.get_case, methods=["GET"], response_model=CaseDetail)
        self.router.add_api_route("/cases/{case_id}/labels", self.save_labels, methods=["PUT"], response_model=Progress)

    def scheme(self) -> list[dict[str, Any]]:
        return LabelScheme.as_dict()

    def list_cases(self, annotator: str | None = None) -> list[CaseSummary]:
        out = []
        for e in self.collector.index():
            segments = self.store.load_segments(e["case_id"])
            progress = None
            if segments is not None:
                labelled = len(self.store.load_case(e["case_id"], annotator=annotator)) if annotator else 0
                progress = Progress(sentences=len(segments), labelled=labelled)
            out.append(CaseSummary(case_id=e["case_id"], title=e.get("title", ""), court=e["court"], decision_year=e["decision_year"],
                                   sections_cited=e["sections_cited"], progress=progress))
        return out

    def _segments(self, case_id: str) -> list[str]:
        """Segment on first open and pin the result, so sentence ids never shift under labels."""
        segments = self.store.load_segments(case_id)
        if segments is None:
            try:
                record = self.collector.load(case_id)
            except KeyError:
                raise HTTPException(404, f"no collected judgment {case_id!r}") from None
            segments = self.segmenter.segment(record.raw_text, case_id)
            self.store.save_segments(case_id, segments, SEGMENTER_VERSION)
        return segments

    def get_case(self, case_id: str, annotator: str | None = None) -> CaseDetail:
        """Sentences with `annotator`'s own labels only — never another annotator's, so double
        annotation stays independent (otherwise Cohen's kappa would be inflated)."""
        try:
            record = self.collector.load(case_id)
        except KeyError:
            raise HTTPException(404, f"no collected judgment {case_id!r}") from None
        segments = self._segments(case_id)
        mine = {s.idx: s.label for s in self.store.load_case(case_id, annotator=annotator)} if annotator else {}
        return CaseDetail(
            case_id=case_id,
            title=record.title,
            court=record.court,
            decision_year=record.decision_year,
            sections_cited=record.sections_cited,
            progress=Progress(sentences=len(segments), labelled=len(mine)),
            sentences=[
                SentenceOut(idx=i, sentence_id=sentence_id(case_id, i), text=t, label=mine.get(i))
                for i, t in enumerate(segments)
            ],
        )

    def save_labels(self, case_id: str, req: SaveLabelsRequest) -> Progress:
        segments = self._segments(case_id)
        bad = [item.idx for item in req.labels if item.idx >= len(segments)]
        if bad:
            raise HTTPException(422, f"sentence index out of range for {case_id}: {bad}")
        self.store.append_many([
            LabeledSentence(
                sentence_id=sentence_id(case_id, item.idx),
                case_id=case_id,
                idx=item.idx,
                text=segments[item.idx],
                label=item.label,
                source="manual",
                annotator=req.annotator,
            )
            for item in req.labels
        ])
        return Progress(sentences=len(segments), labelled=len(self.store.load_case(case_id, annotator=req.annotator)))
