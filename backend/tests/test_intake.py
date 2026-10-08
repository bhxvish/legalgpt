"""Review-tab upload: an uploaded judgment is screened, classified and queued for review."""

import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.annotation.collector import JudgmentCollector
from app.annotation.label_scheme import LabelScheme
from app.annotation.store import AnnotationStore
from app.labeling_assistant.intake import JudgmentIntake, UploadRejected
from app.labeling_assistant.models import Prediction
from app.labeling_assistant.review_queue import HumanReviewQueue
from app.labeling_assistant.router import ReviewRouter

LABELS = list(LabelScheme.LABELS)
JUDGMENT = (Path(__file__).parent / "fixtures" / "judgment_synthetic.txt").read_text(encoding="utf-8")


class FakeClassifier:
    """Alternates confident and unsure predictions so both review paths are exercised."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, examples):
        self.calls += 1
        out = []
        for e in examples:
            conf = 0.95 if e.idx % 2 else 0.5
            label = LABELS[e.idx % len(LABELS)]
            rest = (1 - conf) / (len(LABELS) - 1)
            probs = {l: (conf if l == label else rest) for l in LABELS}
            out.append(Prediction(e.sentence_id, label, probs, conf, conf - rest, e.case_id, e.idx, e.text))
        return out


@pytest.fixture
def env(tmp_path: Path):
    store = AnnotationStore(tmp_path / "store")
    queue = HumanReviewQueue(store)
    collector = JudgmentCollector(tmp_path / "judgments")
    clf = FakeClassifier()
    intake = JudgmentIntake(collector, store, queue, tmp_path / "models", classifier_loader=lambda: (clf, "fake-v1"))
    return intake, collector, store, queue, clf


def test_text_upload_is_classified_and_queued(env) -> None:
    intake, collector, store, queue, clf = env
    job = intake.submit("Ram vs State.txt", JUDGMENT.encode("utf-8"), "priya", background=False)
    assert job.status == "done", job.message
    assert job.case_id == "upload-ram-vs-state" and job.case_id in queue.case_ids()
    case = queue.load(job.case_id)
    assert job.sentences == len(case.items) > 5 and case.model == "fake-v1"
    assert job.flagged == sum(it.needs_review for it in case.items) > 0
    assert sum(job.roles.values()) == job.sentences
    entry = next(e for e in collector.index() if e["case_id"] == job.case_id)
    assert entry["source_path"] == "upload:Ram vs State.txt (priya)" and entry["sections_cited"] == ["279", "304A"]
    assert job.title and entry["title"] == job.title  # header cause title, or the file name
    assert store.load_segments(job.case_id) is not None  # pinned for review / escalation
    assert store.load_case(job.case_id) == []  # nothing promoted before a human signs off


def _pdf(lines: list[str]) -> bytes:
    """A minimal valid single-page PDF with one text line per entry."""
    esc = lambda s: s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = "BT /F1 9 Tf 12 TL 40 800 Td " + " ".join(f"({esc(l)}) Tj T*" for l in lines) + " ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = "%PDF-1.4\n", []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out.encode("latin-1")))
        out += f"{i} 0 obj\n{body}\nendobj\n"
    xref = len(out.encode("latin-1"))
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return out.encode("latin-1")


def test_pdf_upload_is_read_and_queued(env) -> None:
    intake, *_ = env
    lines = [l.strip() for l in JUDGMENT.splitlines() if l.strip() and l.isascii()]
    job = intake.submit("bus accident.pdf", _pdf(lines), "priya", background=False)
    assert job.status == "done", job.message
    assert job.case_id == "upload-bus-accident" and job.sentences > 5


def test_rejections_are_explained(env) -> None:
    intake, *_ = env
    with pytest.raises(UploadRejected, match="only .txt and .pdf"):
        intake.submit("judgment.docx", b"x", "priya")
    with pytest.raises(UploadRejected, match="empty"):
        intake.submit("judgment.txt", b"", "priya")
    recipe = ("Mix the flour and sugar, then bake the cake for forty minutes until golden. " * 12).encode()
    job = intake.submit("recipe.txt", recipe, "priya", background=False)
    assert job.status == "rejected" and "not a criminal case" in job.message
    pdf = intake.submit("scan.pdf", _pdf(["xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"] * 3), "priya", background=False)
    assert pdf.status == "rejected" and "no usable text layer" in pdf.message


def test_same_judgment_twice(env) -> None:
    intake, collector, _, queue, clf = env
    first = intake.submit("a.txt", JUDGMENT.encode(), "priya", background=False)
    again = intake.submit("a copy.txt", JUDGMENT.encode(), "ravi", background=False)
    assert again.status == "rejected" and f"already in the review queue as {first.case_id}" in again.message
    # collected earlier (e.g. by the CLI) but never queued: the upload queues that case instead
    other = JUDGMENT + "\nThe revision petition is dismissed and the sentence is confirmed.\n"
    rec = collector.ingest_text(other, "case99")
    collector.collect(rec)
    reused = intake.submit("b.txt", other.encode(), "priya", background=False)
    assert reused.status == "done" and reused.case_id == "case99" and "case99" in queue.case_ids()


def test_upload_endpoint_runs_in_the_background(env) -> None:
    intake, collector, store, queue, _ = env
    app = FastAPI()
    app.include_router(ReviewRouter(queue, collector, intake).router)
    client = TestClient(app)
    r = client.post("/api/review/uploads", params={"filename": "x.txt", "uploader": "priya"}, content=JUDGMENT.encode())
    assert r.status_code == 202 and r.json()["status"] in ("queued", "reading", "classifying", "done")
    job_id = r.json()["job_id"]
    for _ in range(600):  # up to 30 s: the worker thread is slower when the whole suite runs
        job = client.get(f"/api/review/uploads/{job_id}").json()
        if job["status"] not in ("queued", "reading", "classifying"):
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    assert [j["job_id"] for j in client.get("/api/review/uploads").json()] == [job_id]
    assert job["case_id"] in [c["case_id"] for c in client.get("/api/review/cases").json()]
    assert client.post("/api/review/uploads", params={"filename": "x.exe"}, content=b"x").status_code == 422
    assert client.get("/api/review/uploads/nope").status_code == 404
