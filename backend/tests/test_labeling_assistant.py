"""Module 2: evaluator, review selector, review queue, classifier (tiny model), review API."""

from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.annotation.collector import JudgmentCollector
from app.annotation.label_scheme import LabelScheme
from app.annotation.store import AnnotationStore
from app.labeling_assistant.evaluator import ClassifierEvaluator
from app.labeling_assistant.models import Prediction
from app.labeling_assistant.review_queue import HumanReviewQueue, ReviewIncomplete
from app.labeling_assistant.review_selector import ReviewSelector
from app.labeling_assistant.router import ReviewRouter

LABELS = list(LabelScheme.LABELS)


def pred(case: str, idx: int, label: str, confidence: float) -> Prediction:
    rest = (1 - confidence) / (len(LABELS) - 1)
    probs = {l: (confidence if l == label else rest) for l in LABELS}
    return Prediction(f"{case}:{idx:04d}", label, probs, confidence, confidence - rest, case, idx, f"Sentence {idx} of {case}.")


# ---------------------------------------------------------------- evaluator


def test_evaluate_metrics_by_hand() -> None:
    y_true = ["Facts", "Facts", "Ruling", "Ruling", "Argument"]
    y_pred = ["Facts", "Ruling", "Ruling", "Ruling", "Facts"]
    m = ClassifierEvaluator().evaluate(y_true, y_pred, LABELS)
    assert m.accuracy == pytest.approx(3 / 5)
    assert m.per_class["Facts"].precision == pytest.approx(1 / 2) and m.per_class["Facts"].recall == pytest.approx(1 / 2)
    assert m.per_class["Ruling"].precision == pytest.approx(2 / 3) and m.per_class["Ruling"].recall == 1.0
    assert m.per_class["Argument"].f1 == 0.0
    # macro over the 3 classes present in the gold labels: (0.5 + 0.8 + 0) / 3
    assert m.macro_f1 == pytest.approx((0.5 + 0.8 + 0.0) / 3)
    assert m.confusion[0, 4] == 1 and m.confusion.sum() == 5
    assert "macro-F1 = 0.433" in m.format()


def test_cross_validation_is_grouped_by_case() -> None:
    from types import SimpleNamespace

    ds = [SimpleNamespace(case_id=f"c{c}", label=LABELS[i % 5]) for c in range(6) for i in range(4)]
    seen: list[tuple[set[str], set[str]]] = []

    def train_and_predict(train, test):
        seen.append(({e.case_id for e in train}, {e.case_id for e in test}))
        return [e.label for e in test]  # oracle

    report = ClassifierEvaluator().cross_validate(ds, k=3, seed=1, train_and_predict=train_and_predict, labels=LABELS)
    assert len(report.folds) == 3 and all(m.accuracy == 1.0 for m in report.folds)
    assert all(not (train & test) for train, test in seen)  # no case in both train and test of a fold
    assert set().union(*(test for _, test in seen)) == {f"c{c}" for c in range(6)}
    assert report.macro_f1 == (1.0, 0.0)


# ----------------------------------------------------------- ReviewSelector


def test_selector_uses_confidence() -> None:
    preds = [pred("c", i, "Facts", conf) for i, conf in enumerate([0.95, 0.4, 0.69, 0.71, 0.99, 0.9, 0.85, 0.8, 0.75, 0.97])]
    batch = ReviewSelector(tau_conf=0.7, audit_rate=0.1, seed=3).select(preds, "c")
    assert sorted(p.idx for p in batch.mandatory) == [1, 2]  # exactly the ones below 0.7
    assert len(batch.audit) == 1 and batch.audit[0].confidence >= 0.7
    assert len(batch.all) == 10 and not ({p.idx for p in batch.audit} & {p.idx for p in batch.auto})
    stricter = ReviewSelector(tau_conf=0.9, audit_rate=0.1, seed=3).select(preds, "c")
    assert len(stricter.mandatory) == 6  # raising the threshold flags more sentences


def test_selector_always_audits_some_confident_predictions_and_is_reproducible() -> None:
    preds = [pred("c", i, "Facts", 0.99) for i in range(5)]
    a = ReviewSelector(audit_rate=0.0, min_audit=1, seed=1).select(preds, "c")
    assert len(a.audit) == 1 and not a.mandatory  # never trust high confidence blindly
    b = ReviewSelector(audit_rate=0.0, min_audit=1, seed=1).select(preds, "c")
    assert [p.idx for p in a.audit] == [p.idx for p in b.audit]
    assert len(ReviewSelector(audit_rate=0.5).select([pred("c", i, "Facts", 0.9) for i in range(10)], "c").audit) == 5


# ---------------------------------------------------------- HumanReviewQueue


@pytest.fixture
def store(tmp_path: Path) -> AnnotationStore:
    return AnnotationStore(tmp_path / "store")


def _queue_case(queue: HumanReviewQueue, case: str, n: int = 20) -> None:
    # 2 low-confidence + 18 confident; audit 50% of confident -> 9 audited, 9 auto
    preds = [pred(case, i, "Facts", 0.5 if i < 2 else 0.95) for i in range(n)]
    queue.enqueue(ReviewSelector(tau_conf=0.7, audit_rate=0.5, seed=0).select(preds, case), model="test-model")


def test_concurrent_corrections_are_all_kept(store: AnnotationStore) -> None:
    """A fast reviewer's saves arrive in parallel (FastAPI thread pool): none may be lost and the
    case file must stay valid JSON. Two queue objects, as the API and a script would have."""
    from concurrent.futures import ThreadPoolExecutor

    _queue_case(HumanReviewQueue(store), "fast", n=60)
    queues = [HumanReviewQueue(store), HumanReviewQueue(store)]
    flagged = [it.prediction.sentence_id for it in queues[0].load("fast").items if it.needs_review]
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i_sid: queues[i_sid[0] % 2].apply_correction(i_sid[1], "Ruling", "priya"),
                      enumerate(flagged)))
    case = queues[1].load("fast")
    assert sum(it.status == "corrected" for it in case.items) == len(flagged) > 20
    assert not list(store.root.glob("review_queue/*.tmp"))


def test_high_audit_error_case_is_escalated_not_promoted(store: AnnotationStore) -> None:
    queue = HumanReviewQueue(store, max_audit_error=0.2)
    _queue_case(queue, "bad")
    case = queue.load("bad")
    audits = [it for it in case.items if it.kind == "audit"]
    for it in case.items:
        if it.needs_review:
            wrong = it.kind == "audit" and audits.index(it) < 5  # 5 of 9 audited predictions were wrong
            queue.apply_correction(it.prediction.sentence_id, "Ruling" if wrong else "Facts", "priya")
    assert queue.audit_error_rate("bad") == pytest.approx(5 / 9)
    assert queue.sign_off("bad", "priya") is False
    assert queue.load("bad").status == "escalated"
    assert store.load_case("bad") == []  # nothing promoted
    assert store.load_segments("bad") is not None  # ready for full manual annotation


def test_low_error_case_is_promoted_as_bert_assisted(store: AnnotationStore) -> None:
    queue = HumanReviewQueue(store, max_audit_error=0.2)
    _queue_case(queue, "good")
    flagged = [it for it in queue.load("good").items if it.needs_review]
    queue.apply_correction(flagged[0].prediction.sentence_id, "Argument", "priya")  # a low-confidence fix
    for it in flagged[1:]:
        queue.apply_correction(it.prediction.sentence_id, it.prediction.label, "priya")  # accept
    assert queue.audit_error_rate("good") == 0.0
    assert queue.sign_off("good", "priya") is True
    gold = store.load_case("good")
    assert len(gold) == 20 and {s.source for s in gold} == {"bert_assisted"}
    assert gold[0].label == "Argument" and gold[0].reviewed and gold[0].annotator == "priya"
    auto = [s for s in gold if not s.reviewed]
    assert len(auto) == 9 and {s.annotator for s in auto} == {"model:test-model"}
    with pytest.raises(ValueError):
        queue.sign_off("good", "priya")  # already promoted


def test_sign_off_requires_every_flagged_item_reviewed(store: AnnotationStore) -> None:
    queue = HumanReviewQueue(store)
    _queue_case(queue, "c")
    with pytest.raises(ReviewIncomplete):
        queue.sign_off("c", "priya")
    assert queue.load("c").status == "open"


def test_apply_correction_validation(store: AnnotationStore) -> None:
    queue = HumanReviewQueue(store)
    _queue_case(queue, "c")
    with pytest.raises(ValueError):
        queue.apply_correction("c:0000", "Ratio", "priya")
    with pytest.raises(ValueError):
        queue.apply_correction("c:0000", "Facts", "  ")
    with pytest.raises(KeyError):
        queue.apply_correction("c:0999", "Facts", "priya")
    assert queue.apply_correction("c:0000", "Facts", "priya").status == "accepted"
    assert queue.apply_correction("c:0000", "Ruling", "priya").status == "corrected"
    with pytest.raises(FileExistsError):
        _queue_case(queue, "c")


# ---------------------------------------------------------------- review API


def test_review_api_flow(store: AnnotationStore, tmp_path: Path) -> None:
    queue = HumanReviewQueue(store, max_audit_error=0.2)
    _queue_case(queue, "c")
    app = FastAPI()
    app.include_router(ReviewRouter(queue, JudgmentCollector(tmp_path / "raw")).router)
    client = TestClient(app)
    summary = client.get("/api/review/cases").json()[0]
    assert summary["flagged"] == 11 and summary["reviewed"] == 0 and summary["status"] == "open"
    detail = client.get("/api/review/cases/c").json()
    assert detail["items"][0]["confidence"] == 0.5 and detail["items"][0]["kind"] == "mandatory"
    assert client.post("/api/review/cases/c/sign-off", json={"reviewer": "priya"}).status_code == 409
    for it in detail["items"]:
        if it["kind"] != "auto":
            r = client.put(f"/api/review/cases/c/items/{it['sentence_id']}", json={"reviewer": "priya", "label": it["predicted"]})
            assert r.status_code == 200
    assert client.put("/api/review/cases/c/items/c:0000", json={"reviewer": "priya", "label": "Ratio"}).status_code == 422
    result = client.post("/api/review/cases/c/sign-off", json={"reviewer": "priya"}).json()
    assert result["promoted"] is True and result["status"] == "promoted"
    assert client.get("/api/review/cases/nope").status_code == 404


# ------------------------------------------------- RRLClassifier (tiny model)


@pytest.fixture(scope="module")
def tiny_classifier():
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    from app.labeling_assistant.classifier import RRLClassifier

    words = "the court held accused submitted counsel section punishes whoever facts appeal dismissed in v state".split()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]
    tok = BertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)})  # (vocab_file= is ignored in transformers 5)
    assert tok("the court")["input_ids"] == [2, 5, 6, 3]
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, num_labels=len(LABELS), hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    return RRLClassifier(LABELS, model_name="tiny-bert", model=BertForSequenceClassification(cfg), tokenizer=tok, device="cpu")


def _examples(n: int = 40):
    from app.labeling_assistant.dataset import RRLExample

    texts = {"Ruling": "the court held appeal dismissed", "Argument": "counsel submitted accused",
             "Law Applied": "section punishes whoever", "Precedent": "in state v accused held",
             "Facts": "facts accused state"}
    return [RRLExample(f"c{i % 4}:{i:04d}", f"c{i % 4}", i, texts[LABELS[i % 5]], LABELS[i % 5]) for i in range(n)]


def test_classifier_trains_predicts_and_round_trips(tiny_classifier, tmp_path: Path) -> None:
    from app.labeling_assistant.classifier import RRLClassifier, TrainConfig

    clf = tiny_classifier
    ds = clf.dataset(_examples())
    report = clf.train(ds, ds, TrainConfig(epochs=40, batch_size=8, lr=5e-3, head_lr=5e-3, freeze_layers=0,
                                           warmup_ratio=0.0, bucket_pool_batches=0), log=lambda _: None)
    assert report.history[-1].train_loss < report.history[0].train_loss  # it learns
    assert report.best_val_macro_f1 is not None and report.best_val_macro_f1 > 0.5

    preds = clf.predict([("x:0001", "the court held appeal dismissed"), ("x:0002", "counsel submitted accused")])
    for p in preds:
        assert sum(p.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
        assert p.confidence == pytest.approx(max(p.probabilities.values()), abs=1e-5)
        ranked = sorted(p.probabilities.values(), reverse=True)
        assert p.margin == pytest.approx(ranked[0] - ranked[1], abs=1e-5)
        assert p.label == max(p.probabilities, key=p.probabilities.get)

    path = clf.save(tmp_path / "ckpt")
    again = RRLClassifier.load(path, device="cpu")
    assert again.labels == LABELS
    reloaded = again.predict([("x:0001", "the court held appeal dismissed")])[0]
    assert np.allclose(list(reloaded.probabilities.values()), list(preds[0].probabilities.values()), atol=1e-5)


def test_freeze_layers_freezes_embeddings_and_lower_layers(tiny_classifier) -> None:
    clf = tiny_classifier
    clf._freeze(1)
    bert = clf.model.bert
    assert not any(p.requires_grad for p in bert.embeddings.parameters())
    assert not any(p.requires_grad for p in bert.encoder.layer[0].parameters())
    assert all(p.requires_grad for p in bert.encoder.layer[1].parameters())
    assert all(p.requires_grad for p in clf.model.classifier.parameters())
    clf._freeze(0)
    assert all(p.requires_grad for p in clf.model.parameters())


# ------------------------------------------------------- assisted-path segmentation and titles


@pytest.mark.parametrize(
    "header, title",
    [
        ("Supreme Court of India\nRAJO @ RAJWA versus THE STATE OF BIHAR & ORS.\nDecided on 25-08-2023", "RAJO @ RAJWA versus THE STATE OF BIHAR & ORS."),
        ("Delhi High Court\nNarender vs State Of Delhi on 12 October, 2021\n", "Narender vs State Of Delhi on 12 October, 2021"),
        ("JUDGMENT\nThe appeal is dismissed.", ""),
    ],
)
def test_title_detection(header: str, title: str) -> None:
    from app.labeling_assistant.titles import detect_title

    assert detect_title(header) == title


def test_assisted_segmenter_joins_page_breaks_and_splits_coram_manual_one_unchanged() -> None:
    from app.annotation.segmenter import SentenceSegmenter
    from app.labeling_assistant.segmenter import AssistedSegmenter

    page_break = ("The accused was driving the truck at a high speed on the\n\nhighway near the village when "
                  "the accident occurred. He was arrested.")
    assert AssistedSegmenter().segment(page_break, "t") == [
        "The accused was driving the truck at a high speed on the highway near the village when the accident occurred.",
        "He was arrested.",
    ]
    assert len(SentenceSegmenter().segment(page_break, "t")) == 3  # manual flow keeps its Phase 2 behaviour

    coram = "[ABHAY S. OKA, J.]\nThe appeal arises out of a judgment of the High Court."
    assert AssistedSegmenter().segment(coram, "t") == [
        "[ABHAY S. OKA, J.]", "The appeal arises out of a judgment of the High Court."]
    assert len(SentenceSegmenter().segment(coram, "t")) == 1


# ----------------------------------------------- context, human-checked data, calibration, review load


def test_context_gives_position_and_neighbours_within_the_same_case() -> None:
    from app.labeling_assistant.dataset import RRLExample, context_texts

    ex = [RRLExample(f"a:{i:04d}", "a", i, f"A{i}.") for i in (2, 0, 1)] + [RRLExample("b:0000", "b", 0, "B0.")]
    ctx = context_texts(ex + [RRLExample("x", "", -1, "no case")])
    assert ctx["a:0000"] == "start of judgment. before: (none) after: A1."
    assert ctx["a:0001"] == "early of judgment. before: A0. after: A2."  # position j * 5 // n: 1 * 5 // 3 = 1
    assert ctx["a:0002"] == "late of judgment. before: A1. after: (none)"
    assert ctx["b:0000"] == "start of judgment. before: (none) after: (none)"  # never crosses into case a
    assert "x" not in ctx


def test_context_is_the_second_segment(tiny_classifier) -> None:
    from app.labeling_assistant.dataset import RRLDataset

    ex = _examples(10)
    plain = RRLDataset(ex, tiny_classifier.tokenizer, LABELS, context=False)
    paired = RRLDataset(ex, tiny_classifier.tokenizer, LABELS, max_length=64, context=True)
    assert "token_type_ids" not in plain[0] and set(paired[0]["token_type_ids"]) == {0, 1}
    assert len(paired[0]["input_ids"]) > len(plain[0]["input_ids"])
    short = RRLDataset(ex, tiny_classifier.tokenizer, LABELS, max_length=12, context=True)
    assert all(len(ids) <= 12 for ids in short.input_ids)


def test_human_only_trains_on_checked_labels_but_keeps_unchecked_as_context(tmp_path: Path, tiny_classifier) -> None:
    from app.annotation.models import LabeledSentence
    from app.labeling_assistant.dataset import RRLDataset, is_human_checked

    store = AnnotationStore(tmp_path / "store")
    rows = [LabeledSentence(f"c{c}:{i:04d}", f"c{c}", i, f"sentence {i}", LABELS[i % 5],
                            source="manual" if c < 4 else "bert_assisted", reviewed=c < 4 or i % 2 == 0,
                            annotator="a" if c < 4 or i % 2 == 0 else "model:m")
            for c in range(6) for i in range(6)]
    store.append_many(rows)
    assert sum(is_human_checked(r) for r in rows) == 4 * 6 + 2 * 3
    store.freeze("vt", split_ratios=(0.5, 0.17, 0.33), split_seed=1)
    splits = [RRLDataset.from_version(store, "vt", name, tiny_classifier.tokenizer, LABELS, 64, context=True, human_only=True)
              for name in ("train", "val", "test")]
    trained_on = [e for ds in splits for e in ds.examples]
    assert len(trained_on) == 30 and all(e.label is not None for e in trained_on)
    assert not {"c4:0001", "c5:0003"} & {e.sentence_id for e in trained_on}  # unchecked: not a training label
    everything = [e for name in ("train", "val", "test")
                  for e in RRLDataset.from_version(store, "vt", name, tiny_classifier.tokenizer, LABELS).examples]
    assert len(everything) == 36  # human_only=False keeps all labels (the old behaviour)


def test_review_load_table() -> None:
    from app.labeling_assistant.evaluator import review_load

    rows = {r.bar: r for r in review_load([0.95, 0.9, 0.75, 0.6, 0.4], [True, True, False, True, False], bars=(0.5, 0.9))}
    assert rows[0.5].skip_share == pytest.approx(0.8) and rows[0.5].skip_accuracy == pytest.approx(0.75)
    assert rows[0.9].skip_share == pytest.approx(0.4) and rows[0.9].skip_accuracy == 1.0
    assert rows[0.9].review_share == pytest.approx(0.6)


def test_calibration_changes_confidence_not_labels_and_round_trips(tiny_classifier, tmp_path: Path) -> None:
    from app.labeling_assistant.classifier import RRLClassifier

    clf = tiny_classifier  # trained in test_classifier_trains_predicts_and_round_trips (same module)
    ds = clf.dataset(_examples())
    before = clf.predict(_examples())
    clf.temperature = 1.0
    t = clf.calibrate(ds)
    after = clf.predict(_examples())
    assert t > 0 and [p.label for p in before] == [p.label for p in after]
    again = RRLClassifier.load(clf.save(tmp_path / "cal"), device="cpu")
    assert again.temperature == pytest.approx(t) and again.context is False
    clf.temperature = 1.0
