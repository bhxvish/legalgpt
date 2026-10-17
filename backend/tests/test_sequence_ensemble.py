import json

import numpy as np
import torch

from app.labeling_assistant import sequence
from app.labeling_assistant.dataset import RRLExample
from app.labeling_assistant.sequence import LABELS, EnsembleClassifier, SeqTagger, doc_tensors, is_ensemble, position_features


class FakeClassifier:
    """Stands in for RRLClassifier: fixed probabilities per sentence text."""

    labels = LABELS
    tokenizer = model = None
    context, max_length, temperature, device = True, 256, 1.3, "cpu"

    def __init__(self, table):
        self.table = table

    def dataset(self, examples):
        return examples

    def _probabilities(self, examples):
        return np.array([self.table[e.text] for e in examples], dtype=np.float32)


class FakeEncoder(torch.nn.Module):
    def to(self, device):
        return self


def one_hot(i, p=0.9):
    v = np.full(len(LABELS), (1 - p) / (len(LABELS) - 1), dtype=np.float32)
    v[i] = p
    return v


def test_position_features_span_the_judgment():
    f = position_features(5)
    assert f.shape == (5, 3)
    assert f[0, 0] == 0 and f[-1, 0] == 1


def test_doc_tensors_only_train_on_checked_or_mapped_labels():
    emb = {f"s{i}": np.ones(4, dtype=np.float32) for i in range(3)}
    case = [RRLExample("s0", "c", 0, "a", "Facts"), RRLExample("s1", "c", 1, "b", None), RRLExample("s2", "c", 2, "c", "Ruling")]
    x, y = doc_tensors(case, emb)
    assert x.shape == (1, 3, 7)
    assert y.tolist() == [0, -100, 4]


def test_ensemble_averages_restores_order_and_sums_to_one(monkeypatch):
    texts = {"t0": one_hot(0), "t1": one_hot(1), "t2": one_hot(4)}
    monkeypatch.setattr(sequence, "embed_texts", lambda texts, *a, **k: np.zeros((len(texts), 4), dtype=np.float32))
    tagger = SeqTagger(dim_in=7)
    ens = EnsembleClassifier(FakeClassifier(texts), FakeEncoder(), None, [tagger], weight=0.5)
    # two judgments, interleaved and out of order: rows must come back in the order given
    ex = [RRLExample("b1", "B", 1, "t2"), RRLExample("a0", "A", 0, "t0"), RRLExample("b0", "B", 0, "t1")]
    preds = ens.predict(ex)
    assert [p.sentence_id for p in preds] == ["b1", "a0", "b0"]
    for p in preds:
        assert abs(sum(p.probabilities.values()) - 1) < 1e-4
        assert 0 <= p.margin <= p.confidence <= 1
    # with a weight of 1 the ensemble is the fine-tuned classifier alone
    ens.weight = 1.0
    assert [p.label for p in ens.predict(ex)] == ["Ruling", "Facts", "Law Applied"]


def test_temperature_softens_without_changing_the_label(monkeypatch):
    monkeypatch.setattr(sequence, "embed_texts", lambda texts, *a, **k: np.zeros((len(texts), 4), dtype=np.float32))
    ens = EnsembleClassifier(FakeClassifier({"t": one_hot(2, 0.8)}), FakeEncoder(), None, [SeqTagger(7)], weight=1.0)
    ex = [RRLExample("x", "C", 0, "t")]
    sharp = ens.probabilities(ex, temperature=1.0)[0]
    soft = ens.probabilities(ex, temperature=3.0)[0]
    assert sharp.argmax() == soft.argmax() == 2
    assert soft.max() < sharp.max()


def test_is_ensemble_reads_the_checkpoint_kind(tmp_path):
    (tmp_path / "rrl_meta.json").write_text(json.dumps({"kind": "ensemble"}), encoding="utf-8")
    assert is_ensemble(tmp_path)
    (tmp_path / "rrl_meta.json").write_text(json.dumps({"labels": LABELS}), encoding="utf-8")
    assert not is_ensemble(tmp_path)
