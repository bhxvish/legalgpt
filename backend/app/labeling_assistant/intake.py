"""JudgmentIntake: an uploaded judgment goes straight to InLegalBERT and into the review queue.

The Review tab's upload runs the same pipeline as scripts/run_assisted_labeling.py, one file at a
time on a background worker (the classifier is CPU-heavy and not thread-safe):

    read (.txt / .pdf) -> screen (JudgmentCollector) -> segment (AssistedSegmenter)
    -> predict (RRLClassifier) -> select (ReviewSelector) -> enqueue (HumanReviewQueue)

Every sentence gets a suggested role automatically. Nothing reaches the corpus until a reviewer
checks the flagged sentences and signs the case off, exactly as for script-queued cases.
"""

import queue as queue_mod
import re
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from app.annotation.collector import JudgmentCollector, slugify
from app.annotation.models import sentence_id
from app.annotation.store import AnnotationStore
from app.labeling_assistant.dataset import RRLExample
from app.labeling_assistant.pdf_text import looks_garbled, pdf_to_judgment_text
from app.labeling_assistant.review_queue import HumanReviewQueue
from app.labeling_assistant.review_selector import ReviewSelector
from app.labeling_assistant.segmenter import AssistedSegmenter
from app.labeling_assistant.titles import detect_title

ALLOWED = (".txt", ".pdf")
MAX_BYTES = 15 * 1024 * 1024
JobStatus = Literal["queued", "reading", "classifying", "done", "rejected", "failed"]


class UploadRejected(ValueError):
    """The file cannot be accepted (type, size, unreadable, not a judgment, already handled)."""


def latest_checkpoint(model_dir: Path) -> Path | None:
    runs = sorted((p for p in Path(model_dir).glob("*") if (p / "rrl_meta.json").exists()), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def decode_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


@dataclass
class UploadJob:
    job_id: str
    filename: str
    uploader: str
    status: JobStatus = "queued"
    message: str = "Waiting for the classifier"
    case_id: str = ""
    title: str = ""
    sentences: int = 0
    flagged: int = 0
    roles: dict[str, int] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JudgmentIntake:
    def __init__(
        self,
        collector: JudgmentCollector,
        store: AnnotationStore,
        queue: HumanReviewQueue,
        model_dir: Path,
        tau_conf: float = 0.7,
        audit_rate: float = 0.1,
        classifier_loader: Callable[[], tuple[Any, str]] | None = None,
    ) -> None:
        """classifier_loader returns (classifier, model name); by default the newest checkpoint in
        model_dir is loaded once, on the first upload."""
        self.collector, self.store, self.queue = collector, store, queue
        self.model_dir = Path(model_dir)
        self.selector = ReviewSelector(tau_conf=tau_conf, audit_rate=audit_rate)
        self.segmenter = AssistedSegmenter()
        self._loader = classifier_loader or self._load_latest
        self._clf: tuple[Any, str] | None = None
        self._jobs: dict[str, UploadJob] = {}
        self._work: queue_mod.Queue[tuple[UploadJob, bytes]] = queue_mod.Queue()
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ API used by the router

    def submit(self, filename: str, data: bytes, uploader: str, background: bool = True) -> UploadJob:
        """Validate cheaply now (type, size, empty); the heavy work runs on the worker thread."""
        name = Path(filename or "").name
        if not name.lower().endswith(ALLOWED):
            raise UploadRejected(f"{name or 'file'}: only .txt and .pdf judgments can be uploaded")
        if not data:
            raise UploadRejected(f"{name}: the file is empty")
        if len(data) > MAX_BYTES:
            raise UploadRejected(f"{name}: larger than {MAX_BYTES // (1024 * 1024)} MB")
        job = UploadJob(uuid.uuid4().hex[:12], name, uploader.strip() or "unknown")
        with self._lock:
            self._jobs[job.job_id] = job
        if background:
            self._ensure_worker()
            self._work.put((job, data))
        else:
            self.process(job, data)
        return job

    def job(self, job_id: str) -> UploadJob:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def jobs(self) -> list[UploadJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)[:50]

    # ------------------------------------------------------------------ the pipeline

    def process(self, job: UploadJob, data: bytes) -> UploadJob:
        try:
            job.status, job.message = "reading", "Reading the judgment"
            text = self._read(job.filename, data)
            case_id, text_title = self._collect(job, text)
            job.case_id, job.title = case_id, text_title
            job.status, job.message = "classifying", "InLegalBERT is suggesting a role for every sentence"
            clf, model = self._classifier()
            raw = self.collector.load(case_id).raw_text
            sentences = self.store.load_segments(case_id) or self.segmenter.segment(raw, case_id)
            if not sentences:
                raise UploadRejected("no sentences could be extracted from the text")
            examples = [RRLExample(sentence_id(case_id, i), case_id, i, t) for i, t in enumerate(sentences)]
            predictions = clf.predict(examples)
            batch = self.selector.select(predictions, case_id)
            self.queue.enqueue(batch, model=model)
            job.sentences, job.flagged = len(predictions), len(batch.mandatory) + len(batch.audit)
            job.roles = dict(Counter(p.label for p in predictions).most_common())
            job.status = "done"
            n = lambda k, word: f"{k} {word}{'' if k == 1 else 's'}"
            job.message = (f"Annotated {n(job.sentences, 'sentence')}; {job.flagged} highlighted for you to check "
                           f"({len(batch.mandatory)} uncertain, {n(len(batch.audit), 'spot check')})")
        except UploadRejected as e:
            job.status, job.message = "rejected", str(e)
        except Exception as e:  # keep the worker alive; report the failure on the job
            job.status, job.message = "failed", f"{type(e).__name__}: {e}"
        job.finished_at = time.time()
        return job

    def _read(self, filename: str, data: bytes) -> str:
        if filename.lower().endswith(".pdf"):
            try:
                text = pdf_to_judgment_text(data)
            except Exception as e:
                raise UploadRejected(f"could not read the PDF ({type(e).__name__})") from None
            if looks_garbled(text):
                raise UploadRejected("the PDF has no usable text layer (scanned image or broken text); "
                                     "upload a text-based PDF or a .txt file")
            return text
        return decode_text(data)

    def _collect(self, job: UploadJob, text: str) -> tuple[str, str]:
        """Screen and store the judgment; returns (case_id, title). Reuses an earlier collection of
        the same text if it was never labelled or queued."""
        known = {e["case_id"] for e in self.collector.index()}
        taken = known | set(self.store.case_ids()) | set(self.queue.case_ids())
        base = "upload-" + (slugify(Path(job.filename).stem)[:40].strip("-") or "judgment")
        case_id, n = base, 2
        while case_id in taken:
            case_id, n = f"{base}-{n}", n + 1
        record = self.collector.ingest_text(text, case_id, source_path=f"upload:{job.filename} ({job.uploader})")
        # no "A vs B" cause title in the header: the file name is usually the case name
        record.title = detect_title(record.raw_text) or Path(job.filename).stem
        ok, reason = self.collector.is_eligible(record)
        if not ok:
            dup = re.match(r"duplicate of already-collected case (\S+)", reason)
            if dup:
                existing = dup.group(1)
                if existing in set(self.queue.case_ids()):
                    raise UploadRejected(f"this judgment is already in the review queue as {existing}")
                if existing in set(self.store.case_ids()):
                    raise UploadRejected(f"this judgment is already labelled in the corpus as {existing}")
                return existing, self.collector.load(existing).title or detect_title(text)
            raise UploadRejected(f"not accepted: {reason}")
        self.collector.collect(record)
        return case_id, record.title

    def _classifier(self) -> tuple[Any, str]:
        if self._clf is None:
            self._clf = self._loader()
        return self._clf

    def _load_latest(self) -> tuple[Any, str]:
        from app.labeling_assistant.classifier import RRLClassifier

        checkpoint = latest_checkpoint(self.model_dir)
        if checkpoint is None:
            raise RuntimeError(f"no trained classifier in {self.model_dir}; run "
                               "backend/scripts/train_rrl_classifier.py first")
        return RRLClassifier.load(checkpoint), checkpoint.name

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, name="judgment-intake", daemon=True)
                self._worker.start()

    def _run(self) -> None:
        while True:
            job, data = self._work.get()
            self.process(job, data)
