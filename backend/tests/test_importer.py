"""Spreadsheet importer, on a small workbook with the quirks seen in the team's real sheet."""

from pathlib import Path

import pytest
from openpyxl import Workbook

from app.annotation.collector import JudgmentCollector
from app.annotation.importer import import_annotations, normalize_doc_id
from app.annotation.store import AnnotationStore

SENTENCES = [
    ("FACT", "The appellant was charged under Section 304A of the Indian Penal Code for causing the death of a pedestrian."),
    ("ARGUMENT", "Learned counsel for the appellant submitted that the prosecution witnesses were not reliable."),
    ("LAW", "Section 304A punishes whoever causes the death of any person by a rash or negligent act."),
    ("PRECEDENT", "In State v. Kumar, the Court held that speed alone does not establish rashness."),
    ("RULING", "We find that the prosecution has proved its case beyond reasonable doubt, and the appeal is dismissed."),
]


def _workbook(path: Path) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "Annotations"
    ws.append(["doc_id", "case_name", "sentence_id", "sentence_text", "label", "annotator", "notes"])
    ws.append(["case01", "A vs State (1 Jan 2020)", "sentence_id", "sentence_text", "label", "example", "header echo"])
    # case 7 is written three ways, out of order, with a gap (no sentence 4) and one stray case_name
    for pos, (label, text) in [(3, SENTENCES[2]), (1, SENTENCES[0]), (2, SENTENCES[1]), (5, SENTENCES[3]), (6, SENTENCES[4])]:
        doc = {1: "case 7", 2: "Case7", 3: " case7"}.get(pos, "case7")
        name = "state" if pos == 2 else "Ram vs State of Delhi (12 Oct, 2021)"
        ws.append([doc, name, float(pos), text * 3, label, None if pos != 1 else "priya", None])
    ws.append(["case8", "B vs State (2019)", 1.0, "Too short to be a judgment.", "FACT", None, None])
    ws.append(["case8", "B vs State (2019)", 2.0, "Another line.", "RATIO", None, None])
    path = path / "sheet.xlsx"
    wb.save(path)
    return path


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Path, JudgmentCollector, AnnotationStore]:
    return _workbook(tmp_path), JudgmentCollector(tmp_path / "raw"), AnnotationStore(tmp_path / "store")


def test_normalize_doc_id() -> None:
    assert {normalize_doc_id(x) for x in ("case 11", " case11 ", "Case11")} == {"case11"}


def test_import_cases_labels_and_segments(setup: tuple[Path, JudgmentCollector, AnnotationStore]) -> None:
    xlsx, collector, store = setup
    report = import_annotations(xlsx, collector, store, default_annotator="team-sheet")
    by_case = {c.case_id: c for c in report.cases}

    case = by_case["case7"]
    assert case.status == "imported" and case.sentences == 5
    assert case.title == "Ram vs State of Delhi (12 Oct, 2021)"  # stray "state" ignored
    assert case.detail == "1 sentence_id gap(s) in the sheet"
    assert by_case["case8"].status == "rejected" and "too short" in by_case["case8"].detail
    assert any("repeated header" in p for p in report.problems)
    assert any("unknown label 'RATIO'" in p for p in report.problems)

    gold = store.load_case("case7")
    assert [s.label for s in gold] == ["Facts", "Argument", "Law Applied", "Precedent", "Ruling"]  # sheet order by sentence_id
    assert [s.idx for s in gold] == [0, 1, 2, 3, 4]
    assert gold[0].annotator == "priya" and gold[1].annotator == "team-sheet"
    assert store.load_segments("case7") == [s.text for s in gold]

    record = collector.load("case7")
    assert record.title == "Ram vs State of Delhi (12 Oct, 2021)" and record.decision_year == 2021
    assert record.sections_cited == ["304A"]
    assert "case8" not in store.case_ids()


def test_reimport_is_skipped(setup: tuple[Path, JudgmentCollector, AnnotationStore]) -> None:
    xlsx, collector, store = setup
    import_annotations(xlsx, collector, store, default_annotator="x")
    lines = store.labels_path.read_text(encoding="utf-8")
    again = import_annotations(xlsx, collector, store, default_annotator="x")
    assert {c.case_id: c.status for c in again.cases}["case7"] == "skipped"
    assert store.labels_path.read_text(encoding="utf-8") == lines


def test_dry_run_writes_nothing(setup: tuple[Path, JudgmentCollector, AnnotationStore]) -> None:
    xlsx, collector, store = setup
    report = import_annotations(xlsx, collector, store, default_annotator="x", dry_run=True)
    assert [c.case_id for c in report.imported] == ["case7"]
    assert store.case_ids() == [] and collector.index() == []
