"""Module 1: collector, segmenter, label scheme, store, agreement."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from app.annotation.agreement import AgreementCalculator
from app.annotation.collector import JudgmentCollector, normalize_text
from app.annotation.label_scheme import LabelScheme
from app.annotation.models import LabeledSentence, sentence_id
from app.annotation.segmenter import SentenceSegmenter
from app.annotation.store import AnnotationStore
from app.config import REPO_ROOT

FIXTURES = Path(__file__).parent / "fixtures"
VALID = FIXTURES / "judgment_synthetic.txt"
HINDI = FIXTURES / "judgment_hindi.txt"


# ------------------------------------------------------------- SentenceSegmenter


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The court followed State v. Kumar on this point. The appeal failed.",
         ["The court followed State v. Kumar on this point.", "The appeal failed."]),
        ("He was convicted under S. 304A of the Code. Sentence was two years.",
         ["He was convicted under S. 304A of the Code.", "Sentence was two years."]),
    ],
)
def test_segmenter_keeps_legal_abbreviations_together(text: str, expected: list[str]) -> None:
    sentences = SentenceSegmenter().segment(text, "t")
    assert sentences == expected
    assert not any(s.endswith(("State v.", "S.")) for s in sentences)


def test_segmenter_on_judgment_layout() -> None:
    sentences = SentenceSegmenter().segment(VALID.read_text(encoding="utf-8"), "t")
    assert sentences[:2] == ["IN THE HIGH COURT OF JUDICATURE AT BOMBAY", "CRIMINAL APPELLATE JURISDICTION"]
    # paragraph number dropped; "S. 304A of the I.P.C." kept intact; split after the acronym-ending sentence
    assert ("This appeal is directed against the judgment of the Sessions Court convicting the appellant "
            "under S. 304A of the I.P.C. and sentencing him to rigorous imprisonment for two years.") in sentences
    # a sentence wrapped across two lines is re-joined
    assert any(s.startswith("The prosecution case is that") and s.endswith("riding a motorcycle.") for s in sentences)
    # "Smt. R. Rao" and "PW.2" do not split
    assert any("Smt. R. Rao, contended" in s for s in sentences)
    assert "It was further submitted that PW.2 was not a reliable witness." in sentences


# --------------------------------------------------------------- JudgmentCollector


@pytest.fixture
def collector(tmp_path: Path) -> JudgmentCollector:
    return JudgmentCollector(tmp_path / "raw_judgments")


def test_collector_accepts_valid_criminal_judgment(collector: JudgmentCollector) -> None:
    record = collector.collect(collector.ingest_file(VALID))
    assert record.selection_status == "eligible", record.rejection_reason
    assert record.case_id == "judgment-synthetic"
    assert record.sections_cited == ["279", "304A"]
    assert record.court == "High Court of Judicature at Bombay"
    assert record.decision_year == 2019
    assert (collector.store_dir / "judgment-synthetic.txt").exists()
    assert collector.load("judgment-synthetic").raw_text == record.raw_text


def test_collector_rejects_non_english(collector: JudgmentCollector) -> None:
    record = collector.collect(collector.ingest_file(HINDI))
    assert record.selection_status == "rejected"
    assert "not English" in record.rejection_reason
    assert collector.index() == []


def test_collector_rejects_duplicate_even_if_reformatted(collector: JudgmentCollector, tmp_path: Path) -> None:
    assert collector.collect(collector.ingest_file(VALID)).selection_status == "eligible"
    copy = tmp_path / "same_case_reexported.txt"
    copy.write_text(VALID.read_text(encoding="utf-8").replace("\n", "\r\n   ").upper(), encoding="utf-8")
    record = collector.collect(collector.ingest_file(copy))
    assert record.selection_status == "rejected"
    assert "duplicate of already-collected case judgment-synthetic" in record.rejection_reason
    assert len(collector.index()) == 1


def test_collector_rejects_non_criminal(collector: JudgmentCollector, tmp_path: Path) -> None:
    civil = tmp_path / "civil.txt"
    civil.write_text(
        "The plaintiff instituted a suit for specific performance of the agreement to sell. " * 10
        + "The decree of the trial court in favour of the plaintiff is upheld and the second appeal is dismissed.",
        encoding="utf-8",
    )
    record = collector.collect(collector.ingest_file(civil))
    assert record.selection_status == "rejected" and "not a criminal case" in record.rejection_reason


def test_section_detection_variants() -> None:
    text = "convicted under Sections 302, 307 and 34 of the IPC; also u/s 498A IPC and 304B/34 I.P.C. and S. 120B of the Indian Penal Code"
    assert JudgmentCollector.detect_sections(text) == ["34", "120B", "302", "304B", "307", "498A"]


def test_normalize_text_unifies_whitespace() -> None:
    assert normalize_text("a  b\r\n\r\n\r\n\r\nc  ") == "a b\n\nc"


# ---------------------------------------------------------------- LabelScheme


def test_label_scheme() -> None:
    assert LabelScheme.LABELS == ("Facts", "Law Applied", "Precedent", "Argument", "Ruling")
    assert not LabelScheme.validate("None")  # dropped in Phase 3
    assert all(LabelScheme.validate(label) for label in LabelScheme.LABELS)
    assert not LabelScheme.validate("Ratio")
    assert all(LabelScheme.describe(label).endswith(".") for label in LabelScheme.LABELS)
    with pytest.raises(ValueError):
        LabelScheme.describe("Ratio")


def test_guideline_matches_label_scheme() -> None:
    sys.path.insert(0, str(REPO_ROOT / "backend" / "scripts"))
    import build_guideline

    committed = (REPO_ROOT / "docs" / "annotation_guideline.md").read_text(encoding="utf-8")
    assert committed == build_guideline.render(), "guideline is stale: run backend/scripts/build_guideline.py"
    for label in LabelScheme.LABELS:
        assert f"· {label}\n" in committed and LabelScheme.describe(label) in committed


# -------------------------------------------------------------- AnnotationStore


def _label(case: str, idx: int, label: str, annotator: str = "alice", text: str = "") -> LabeledSentence:
    return LabeledSentence(sentence_id(case, idx), case, idx, text or f"sentence {idx} of {case}", label, annotator=annotator)


@pytest.fixture
def store(tmp_path: Path) -> AnnotationStore:
    return AnnotationStore(tmp_path / "annotation_store")


def test_store_latest_decision_wins_and_annotator_filter(store: AnnotationStore) -> None:
    store.append(_label("c1", 0, "Facts"))
    store.append(_label("c1", 1, "Ruling"))
    store.append(_label("c1", 0, "Argument", annotator="bob"))
    store.append(_label("c1", 0, "Law Applied"))  # alice relabels
    assert [(s.idx, s.label) for s in store.load_case("c1")] == [(0, "Law Applied"), (1, "Ruling")]
    assert [(s.idx, s.label) for s in store.load_case("c1", annotator="bob")] == [(0, "Argument")]
    assert store.annotators("c1") == ["alice", "bob"]


def test_store_rejects_invalid_label(store: AnnotationStore) -> None:
    with pytest.raises(ValueError):
        store.append(_label("c1", 0, "Ratio"))
    assert not store.labels_path.exists()


def test_split_by_case_has_no_leakage_and_is_deterministic(store: AnnotationStore) -> None:
    for c in range(20):
        for i in range(5):
            store.append(_label(f"case{c:02d}", i, "Facts"))
    splits = store.split_by_case((0.8, 0.1, 0.1), seed=7)
    assert [len(splits[k]) for k in ("train", "val", "test")] == [16, 2, 2]
    assert set(splits["train"]).isdisjoint(splits["val"]) and set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])
    assert sorted(sum(splits.values(), [])) == store.case_ids()
    assert store.split_by_case((0.8, 0.1, 0.1), seed=7) == splits
    assert store.split_by_case((0.8, 0.1, 0.1), seed=8) != splits
    with pytest.raises(ValueError):
        store.split_by_case((0.5, 0.5, 0.5))


def test_freeze_writes_manifest_with_case_ids_and_sha256(store: AnnotationStore) -> None:
    store.save_segments("c1", ["s0", "s1"])
    store.append_many([_label("c1", 0, "Facts"), _label("c1", 1, "Ruling"), _label("c2", 0, "Argument")])
    manifest = store.freeze("v0.1", notes="test")
    path = store.root / "versions" / "v0.1" / "manifest.json"
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["case_ids"] == ["c1", "c2"]
    assert len(on_disk["sha256"]) == 64 and all(len(e["sha256"]) == 64 for e in on_disk["cases"].values())
    assert on_disk["cases"]["c1"]["complete"] is True and on_disk["cases"]["c1"]["sentences"] == 2
    assert on_disk["label_counts"] == {"Argument": 1, "Facts": 1, "Ruling": 1}
    assert manifest.sha256 == on_disk["sha256"]
    # immutable, reproducible, verifiable
    with pytest.raises(FileExistsError):
        store.freeze("v0.1")
    assert store.freeze("v0.2").sha256 == manifest.sha256  # same data -> same digest
    store.append(_label("c2", 0, "Ruling"))
    assert store.freeze("v0.3").sha256 != manifest.sha256
    loaded, sentences = store.load_version("v0.1")
    assert loaded.sha256 == manifest.sha256 and len(sentences) == 3


def test_load_version_detects_tampering(store: AnnotationStore) -> None:
    store.append(_label("c1", 0, "Facts"))
    store.freeze("v1")
    labels = store.root / "versions" / "v1" / "labels.jsonl"
    labels.write_text(labels.read_text(encoding="utf-8").replace('"Facts"', '"Ruling"'), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        store.load_version("v1")


def test_freeze_empty_store_raises(store: AnnotationStore) -> None:
    with pytest.raises(ValueError):
        store.freeze("v1")


# ---------------------------------------------------------- AgreementCalculator


def test_cohen_kappa_on_double_annotated_fixture() -> None:
    a = ["Facts", "Facts", "Argument", "Argument", "Ruling", "Ruling", "Law Applied", "Precedent", "Ruling", "Facts"]
    b = ["Facts", "Argument", "Argument", "Argument", "Ruling", "Facts", "Law Applied", "Precedent", "Ruling", "Facts"]
    calc = AgreementCalculator()
    kappa = calc.cohen_kappa(a, b)
    assert -1.0 <= kappa <= 1.0
    # hand computation: p_o = 8/10; p_e = (3*3 + 2*3 + 3*2 + 1*1 + 1*1)/100 = 0.23
    assert kappa == pytest.approx((0.8 - 0.23) / (1 - 0.23))
    cm = calc.confusion_matrix(a, b)
    assert cm.shape == (5, 5) and cm.sum() == 10 and np.trace(cm) == 8
    assert cm[0, 3] == 1  # A said Facts, B said Argument


def test_cohen_kappa_extremes() -> None:
    calc = AgreementCalculator()
    assert calc.cohen_kappa(["Facts", "Ruling"] * 5, ["Facts", "Ruling"] * 5) == pytest.approx(1.0)
    assert calc.cohen_kappa(["Facts"] * 4, ["Facts"] * 4) == 1.0
    assert calc.cohen_kappa(["Facts", "Ruling"] * 5, ["Ruling", "Facts"] * 5) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        calc.cohen_kappa(["Facts"], ["Facts", "Ruling"])


def test_paired_labels_aligns_shared_sentences(store: AnnotationStore) -> None:
    store.append_many([_label("c1", 0, "Facts"), _label("c1", 1, "Ruling"), _label("c1", 2, "Argument")])
    store.append_many([_label("c1", 0, "Facts", "bob"), _label("c1", 2, "Ruling", "bob")])
    a, b = AgreementCalculator.paired_labels(store, ["c1"], "alice", "bob")
    assert (a, b) == (["Facts", "Argument"], ["Facts", "Ruling"])


@pytest.mark.parametrize(
    "header, court",
    [
        ("Supreme Court of India\nState Of Punjab vs Ramesh on 12 March, 2019", "Supreme Court of India"),
        ("IN THE HIGH COURT OF DELHI AT NEW DELHI\nCRL.A. 12/2018", "High Court of Delhi"),
        ("Allahabad High Court\nRam vs State Of U.P. on 3 May, 2017", "Allahabad High Court"),
        ("JUDGMENT\nThe High Court had acquitted the accused.", ""),
    ],
)
def test_court_detection(header: str, court: str) -> None:
    assert JudgmentCollector.detect_court(header) == court


def _boundaries(stream: str, sentences: list[str]) -> set[int]:
    """Sentence-end offsets in an alphanumeric-only stream, found by matching each sentence's
    text in order (so dropped paragraph numbers or punctuation cannot shift later offsets)."""
    import re as _re

    ends, ptr = set(), 0
    for s in sentences:
        key = _re.sub(r"[^0-9a-z]", "", s.lower())
        i = stream.find(key, ptr) if key else -1
        if i != -1:
            ptr = i + len(key)
            ends.add(ptr)
    return ends


def test_segmenter_agrees_with_team_seed_segmentation() -> None:
    """On the imported seed corpus (git-ignored, so skipped without it), re-segmenting each case's
    text must reproduce the team's sentence boundaries closely. Measured 0.956 F1 at import."""
    import re as _re

    from app.config import get_settings

    store = AnnotationStore(get_settings().annotation_store_dir)
    def imported_from_sheet(case_id: str) -> bool:
        path = store.root / "segments" / f"{case_id}.json"
        return path.exists() and json.loads(path.read_text(encoding="utf-8"))["segmenter"].startswith("xlsx:")

    cases = [c for c in store.case_ids() if imported_from_sheet(c)]
    if len(cases) < 5:
        pytest.skip("team seed corpus not imported (backend/scripts/import_annotations.py)")
    seg, tp, fp, fn = SentenceSegmenter(), 0, 0, 0
    for c in cases:
        gold = store.load_segments(c)
        stream = _re.sub(r"[^0-9a-z]", "", " ".join(gold).lower())
        g, o = _boundaries(stream, gold), _boundaries(stream, seg.segment(" ".join(gold), c))
        tp, fp, fn = tp + len(g & o), fp + len(o - g), fn + len(g - o)
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    assert 2 * precision * recall / (precision + recall) >= 0.94, (precision, recall)
