"""Explanations for the classifier (Integrated Gradients, neighbours), fact-check what-ifs, and
the chat's retrieval breakdown."""

import json

import numpy as np
import pytest

from app.annotation.label_scheme import LabelScheme
from app.labeling_assistant.dataset import RRLExample
from app.labeling_assistant.explain import ClassifierExplainer
from app.verification.what_if import verdict_of, what_if

LABELS = list(LabelScheme.LABELS)


@pytest.fixture(scope="module")
def tiny():
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    from app.labeling_assistant.classifier import RRLClassifier

    words = "the court held accused submitted counsel section punishes whoever facts appeal dismissed in v state".split()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]
    tok = BertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)})
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, num_labels=len(LABELS), hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    import torch

    torch.manual_seed(0)
    return RRLClassifier(LABELS, model_name="tiny-bert", model=BertForSequenceClassification(cfg), tokenizer=tok,
                         device="cpu", context=True, max_length=64)


CASE = [RRLExample("c:0", "c", 0, "the court held"), RRLExample("c:1", "c", 1, "counsel submitted the appeal"),
        RRLExample("c:2", "c", 2, "the appeal dismissed")]


def test_integrated_gradients_are_complete(tiny) -> None:
    """Attributions add up to logit(input) - logit(baseline) once the path is fine enough."""
    ex = ClassifierExplainer(tiny)
    ctx = ex._context(CASE[1], CASE)
    gaps = [ex.integrated_gradients(CASE[1].text, ctx, 0, steps=s)["completeness_gap"] for s in (4, 128)]
    assert gaps[1] < 0.01 and gaps[1] <= gaps[0] + 1e-6


def test_explanation_covers_every_word_and_the_context_share(tiny) -> None:
    out = ClassifierExplainer(tiny).explain(CASE[1], CASE, steps=16)
    assert [w["text"] for w in out["words"]] == ["counsel", "submitted", "the", "appeal"]
    assert all(CASE[1].text[w["start"]:w["end"]] == w["text"] for w in out["words"])
    assert 0.0 <= out["context_share"] <= 1.0 and max(abs(w["weight"]) for w in out["words"]) == pytest.approx(1.0)
    assert out["label"] in LABELS and abs(sum(out["probabilities"].values()) - 1) < 1e-3
    other = ClassifierExplainer(tiny).explain(CASE[1], CASE, label="Ruling", steps=16)
    assert other["label"] == "Ruling"  # "why not Ruling?"


def test_neighbours_exclude_the_same_case(tiny, tmp_path) -> None:
    ex = ClassifierExplainer(tiny, tmp_path / "neighbours.npz")
    pool = [RRLExample("x:0", "x", 0, "counsel submitted", "Argument"), RRLExample("c:9", "c", 9, "counsel submitted the appeal", "Argument"),
            RRLExample("y:0", "y", 0, "section punishes whoever", "Law Applied")]
    vecs = ex.pooled([p.text for p in pool], ["" for _ in pool])
    np.savez(tmp_path / "neighbours.npz", vectors=vecs)
    (tmp_path / "neighbours.json").write_text(json.dumps([{"sentence_id": p.sentence_id, "case_id": p.case_id,
                                                           "label": p.label, "text": p.text} for p in pool]))
    got = ex.neighbours(CASE[1].text, "", k=2, exclude_case="c")
    assert len(got) == 2 and all(n["case_id"] != "c" for n in got)


# ------------------------------------------------------------------ what-if
def _el(pred: str, status: str) -> dict:
    return {"predicate": pred, "description": pred, "required": ["true"], "found": "x", "status": status}


def test_what_if_matches_the_verdict_rules() -> None:
    assert verdict_of(["satisfied", "missing", "violated"]) == "INCONSISTENT"
    ins = what_if([_el("caused_death", "satisfied"), _el("act_type", "missing")])
    assert ins["verdict"] == "INSUFFICIENT" and [m["predicate"] for m in ins["missing"]] == ["act_type"]
    assert ins["missing"][0]["verdict_if_satisfied"] == "CONSISTENT" and "rash" in ins["missing"][0]["question"]
    inc = what_if([_el("on_public_way", "violated"), _el("act_type", "missing")])
    assert inc["blockers"][0]["verdict_if_satisfied"] == "INSUFFICIENT"  # the other element is still missing
    ok = what_if([_el("caused_hurt", "satisfied"), _el("act_type", "satisfied")])
    assert ok["verdict"] == "CONSISTENT" and ok["sensitive_elements"] == ["caused_hurt", "act_type"]


# ------------------------------------------------------------------ chat retrieval breakdown
def test_chat_streams_how_the_sources_were_chosen(fake_embedder, memory_store) -> None:
    from app.core.models import Chunk
    from app.core.retriever import Retriever
    from tests.conftest import StubLLM
    from tests.test_chat_router import client_for, parse_sse

    chunk = Chunk(chunk_id="IPC:304A:x", doc_id="IPC", section="304A", title="Causing death by negligence",
                  citation_path="IPC > Chapter XVI > Section 304A",
                  text="Section 304A. Causing death by negligence\nWhoever causes the death of any person by doing any rash or negligent act.")
    memory_store.upsert([chunk], fake_embedder.embed([chunk.text]))
    client = client_for(Retriever(fake_embedder, memory_store, min_similarity=0.2), StubLLM())
    events = parse_sse(client.post("/api/chat", json={"question": "Punishment for death caused by a negligent act under s. 304A?",
                                                      "mode": "legal"}).text)
    names = [e for e, _ in events]
    assert names[:3] == ["meta", "sources", "retrieval"]
    info = dict(events)["retrieval"]
    assert info["named_sections"] == ["304A"] and info["sources"][0]["named"] is True
    assert {"floor", "base_floor", "lexicon_terms", "used_context", "scope_probability"} <= set(info)
