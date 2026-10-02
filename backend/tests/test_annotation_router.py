"""Annotation API: load a collected judgment, segment it, save labels for every sentence."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.annotation.collector import JudgmentCollector
from app.annotation.label_scheme import LabelScheme
from app.annotation.router import AnnotationRouter
from app.annotation.store import AnnotationStore

FIXTURE = Path(__file__).parent / "fixtures" / "judgment_synthetic.txt"
CASE = "judgment-synthetic"


@pytest.fixture
def env(tmp_path: Path) -> tuple[TestClient, AnnotationStore]:
    collector = JudgmentCollector(tmp_path / "raw_judgments")
    assert collector.collect(collector.ingest_file(FIXTURE)).selection_status == "eligible"
    store = AnnotationStore(tmp_path / "annotation_store")
    app = FastAPI()
    app.include_router(AnnotationRouter(collector, store).router)
    return TestClient(app), store


def test_scheme_lists_labels_with_descriptions(env: tuple[TestClient, AnnotationStore]) -> None:
    client, _ = env
    scheme = client.get("/api/annotation/scheme").json()
    assert [s["name"] for s in scheme] == list(LabelScheme.LABELS)
    assert all(s["description"] == LabelScheme.describe(s["name"]) for s in scheme)


def test_label_every_sentence_of_a_case(env: tuple[TestClient, AnnotationStore]) -> None:
    client, store = env
    cases = client.get("/api/annotation/cases", params={"annotator": "alice"}).json()
    assert [c["case_id"] for c in cases] == [CASE] and cases[0]["progress"] is None  # not opened yet

    case = client.get(f"/api/annotation/cases/{CASE}", params={"annotator": "alice"}).json()
    sentences = case["sentences"]
    assert len(sentences) > 10 and all(s["label"] is None for s in sentences)
    assert case["sections_cited"] == ["279", "304A"]

    labels = [{"idx": s["idx"], "label": LabelScheme.LABELS[s["idx"] % 5]} for s in sentences]
    progress = client.put(f"/api/annotation/cases/{CASE}/labels", json={"annotator": "alice", "labels": labels}).json()
    assert progress == {"sentences": len(sentences), "labelled": len(sentences)}

    saved = store.load_case(CASE)
    assert [(s.idx, s.label) for s in saved] == [(l["idx"], l["label"]) for l in labels]
    assert all(s.text == sentences[s.idx]["text"] and s.source == "manual" and s.annotator == "alice" for s in saved)
    reopened = client.get(f"/api/annotation/cases/{CASE}", params={"annotator": "alice"}).json()
    assert [s["label"] for s in reopened["sentences"]] == [l["label"] for l in labels]

    manifest = store.freeze("v-test")
    assert manifest.case_ids == [CASE] and manifest.cases[CASE].complete


def test_annotators_never_see_each_others_labels(env: tuple[TestClient, AnnotationStore]) -> None:
    client, _ = env
    client.put(f"/api/annotation/cases/{CASE}/labels", json={"annotator": "alice", "labels": [{"idx": 0, "label": "Facts"}]})
    bob = client.get(f"/api/annotation/cases/{CASE}", params={"annotator": "bob"}).json()
    assert bob["sentences"][0]["label"] is None and bob["progress"]["labelled"] == 0


def test_segmentation_is_pinned_on_first_open(env: tuple[TestClient, AnnotationStore]) -> None:
    client, store = env
    first = client.get(f"/api/annotation/cases/{CASE}").json()["sentences"]
    assert store.load_segments(CASE) == [s["text"] for s in first]


@pytest.mark.parametrize(
    "body, status",
    [
        ({"annotator": "alice", "labels": [{"idx": 0, "label": "Ratio"}]}, 422),
        ({"annotator": "  ", "labels": [{"idx": 0, "label": "Facts"}]}, 422),
        ({"annotator": "alice", "labels": [{"idx": 9999, "label": "Facts"}]}, 422),
        ({"annotator": "alice", "labels": []}, 422),
    ],
)
def test_invalid_label_requests_are_rejected(env: tuple[TestClient, AnnotationStore], body: dict, status: int) -> None:
    client, store = env
    assert client.put(f"/api/annotation/cases/{CASE}/labels", json=body).status_code == status
    assert store.load_case(CASE) == []


def test_unknown_case_is_404(env: tuple[TestClient, AnnotationStore]) -> None:
    client, _ = env
    assert client.get("/api/annotation/cases/nope").status_code == 404
