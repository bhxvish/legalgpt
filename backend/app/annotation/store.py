"""AnnotationStore: the single source of truth for labelled sentences.

Layout under `root` (data/annotation_store/ by default):

    labels.jsonl                 append-only log of every LabeledSentence decision
    segments/<case_id>.json      the sentence segmentation each case was annotated against
    versions/<version>/          frozen snapshots, written once by freeze() and never modified
        labels.jsonl             the resolved (gold) labels of that snapshot
        manifest.json            case ids, per-case SHA-256, overall SHA-256, splits

The latest decision for a sentence wins, so re-labelling or adjudicating is just another append.
"""

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.annotation.label_scheme import LabelScheme
from app.annotation.models import LabeledSentence, utc_now

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class CaseEntry:
    sha256: str
    sentences: int  # in the case's segmentation
    labelled: int
    complete: bool
    label_counts: dict[str, int]


@dataclass
class Manifest:
    version: str
    created_at: str
    labels: list[str]
    case_ids: list[str]
    cases: dict[str, CaseEntry]
    total_sentences: int
    label_counts: dict[str, int]
    split_seed: int
    split_ratios: list[float]
    splits: dict[str, list[str]]
    sha256: str = ""  # over every case's digest; identifies the snapshot
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Manifest":
        d = dict(d)
        d["cases"] = {k: CaseEntry(**v) for k, v in d["cases"].items()}
        return cls(**d)


class AnnotationStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.labels_path = self.root / "labels.jsonl"

    # ---------------------------------------------------------------- labels

    def append(self, sentence: LabeledSentence) -> None:
        self.append_many([sentence])

    def append_many(self, sentences: list[LabeledSentence]) -> None:
        for s in sentences:
            if not LabelScheme.validate(s.label):
                raise ValueError(f"invalid label {s.label!r} for {s.sentence_id}")
            if s.source not in ("manual", "bert_assisted"):
                raise ValueError(f"invalid source {s.source!r} for {s.sentence_id}")
        self.root.mkdir(parents=True, exist_ok=True)
        with self.labels_path.open("a", encoding="utf-8") as f:
            for s in sentences:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

    def _log(self) -> list[LabeledSentence]:
        if not self.labels_path.exists():
            return []
        lines = self.labels_path.read_text(encoding="utf-8").splitlines()
        return [LabeledSentence.from_dict(json.loads(line)) for line in lines if line.strip()]

    def load_case(self, case_id: str, annotator: str | None = None) -> list[LabeledSentence]:
        """Latest decision per sentence (optionally only `annotator`'s), ordered by idx."""
        latest: dict[str, LabeledSentence] = {}
        for s in self._log():
            if s.case_id == case_id and (annotator is None or s.annotator == annotator):
                latest[s.sentence_id] = s  # log order = time order, so later lines win
        return sorted(latest.values(), key=lambda s: s.idx)

    def case_ids(self) -> list[str]:
        return sorted({s.case_id for s in self._log()})

    def annotators(self, case_id: str) -> list[str]:
        return sorted({s.annotator for s in self._log() if s.case_id == case_id and s.annotator})

    # ------------------------------------------------------------ segments

    def save_segments(self, case_id: str, sentences: list[str], segmenter: str = "") -> None:
        path = self.root / "segments" / f"{case_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"case_id": case_id, "segmenter": segmenter, "created_at": utc_now(), "sentences": sentences}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    def load_segments(self, case_id: str) -> list[str] | None:
        """The segmentation labels were made against, so later segmenter changes cannot shift
        sentence ids under existing labels. None if the case was never opened."""
        path = self.root / "segments" / f"{case_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["sentences"]

    # --------------------------------------------------------------- splits

    def split_by_case(
        self, ratios: tuple[float, ...] = (0.8, 0.1, 0.1), seed: int = 13, case_ids: list[str] | None = None
    ) -> dict[str, list[str]]:
        """Assign whole cases to train/val/test. Splitting by case (never by sentence) keeps
        sentences of one judgment out of more than one split. Deterministic for a given seed."""
        if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6 or min(ratios) < 0:
            raise ValueError(f"ratios must be three non-negative numbers summing to 1, got {ratios}")
        ids = sorted(case_ids if case_ids is not None else self.case_ids())
        random.Random(seed).shuffle(ids)
        n = len(ids)
        n_val, n_test = round(n * ratios[1]), round(n * ratios[2])
        # with few cases, still give each requested split at least one case when possible
        if n >= 3:
            n_val = max(n_val, 1 if ratios[1] > 0 else 0)
            n_test = max(n_test, 1 if ratios[2] > 0 else 0)
        n_train = max(n - n_val - n_test, 0)
        return {
            "train": sorted(ids[:n_train]),
            "val": sorted(ids[n_train : n_train + n_val]),
            "test": sorted(ids[n_train + n_val :]),
        }

    # --------------------------------------------------------------- freeze

    @staticmethod
    def _canonical(sentences: list[LabeledSentence]) -> list[dict[str, Any]]:
        """Fields that define the data (not who/when), in a stable order — the hashed form."""
        return [
            {"sentence_id": s.sentence_id, "case_id": s.case_id, "idx": s.idx, "text": s.text,
             "label": s.label, "source": s.source, "reviewed": s.reviewed}
            for s in sorted(sentences, key=lambda s: s.idx)
        ]

    @classmethod
    def case_digest(cls, sentences: list[LabeledSentence]) -> str:
        blob = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in cls._canonical(sentences))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def freeze(
        self,
        version: str,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 13,
        notes: str = "",
    ) -> Manifest:
        """Snapshot the current gold labels as `versions/<version>/` and return its manifest.
        Versions are immutable: freezing an existing version raises FileExistsError."""
        if not version or any(c in version for c in r'\/:*?"<>|'):
            raise ValueError(f"invalid version name {version!r}")
        out = self.root / "versions" / version
        if out.exists():
            raise FileExistsError(f"version {version!r} already frozen at {out}")

        cases: dict[str, CaseEntry] = {}
        rows: list[dict[str, Any]] = []
        totals: Counter[str] = Counter()
        for case_id in self.case_ids():
            gold = self.load_case(case_id)
            if not gold:
                continue
            segments = self.load_segments(case_id)
            n_segments = len(segments) if segments is not None else len(gold)
            counts = Counter(s.label for s in gold)
            totals.update(counts)
            cases[case_id] = CaseEntry(
                sha256=self.case_digest(gold),
                sentences=n_segments,
                labelled=len(gold),
                complete=len(gold) >= n_segments,
                label_counts=dict(sorted(counts.items())),
            )
            rows.extend(self._canonical(gold))
        if not cases:
            raise ValueError("nothing to freeze: no labelled sentences in the store")

        case_ids = sorted(cases)
        overall = hashlib.sha256("\n".join(f"{c}:{cases[c].sha256}" for c in case_ids).encode()).hexdigest()
        manifest = Manifest(
            version=version,
            created_at=utc_now(),
            labels=list(LabelScheme.LABELS),
            case_ids=case_ids,
            cases=cases,
            total_sentences=len(rows),
            label_counts=dict(sorted(totals.items())),
            split_seed=split_seed,
            split_ratios=list(split_ratios),
            splits=self.split_by_case(split_ratios, split_seed, case_ids),
            sha256=overall,
            notes=notes,
        )
        out.mkdir(parents=True)
        with (out / "labels.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        (out / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest

    def load_version(self, version: str, verify: bool = True) -> tuple[Manifest, list[LabeledSentence]]:
        """Read a frozen snapshot; with `verify`, recompute every digest against the manifest."""
        out = self.root / "versions" / version
        manifest = Manifest.from_dict(json.loads((out / "manifest.json").read_text(encoding="utf-8")))
        lines = (out / "labels.jsonl").read_text(encoding="utf-8").splitlines()
        sentences = [LabeledSentence.from_dict(json.loads(line)) for line in lines if line.strip()]
        if verify:
            by_case: dict[str, list[LabeledSentence]] = {}
            for s in sentences:
                by_case.setdefault(s.case_id, []).append(s)
            for case_id, entry in manifest.cases.items():
                if self.case_digest(by_case.get(case_id, [])) != entry.sha256:
                    raise ValueError(f"snapshot {version!r}: case {case_id} does not match its manifest digest")
        return manifest, sentences
