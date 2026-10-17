"""Sentence roles as a sequence task, and the ensemble the Review tab uses.

SeqTagger: a 2-layer BiLSTM that reads a whole judgment in order. Each sentence is a mean-pooled
vector from an InLegalBERT encoder (frozen) plus its relative position in the judgment.

EnsembleClassifier: the fine-tuned RRLClassifier and SeqTagger(s) averaged 50/50, then
temperature-scaled so the confidences can drive the review queue. It offers the same predict()
as RRLClassifier, and passes tokenizer / model / context through to the fine-tuned member, so the
explanations (explain.py) describe that member. Evaluated in docs/reports/paper_analysis.md §7b
(macro-F1 0.807 vs 0.739 for the fine-tuned model alone, 5-fold CV).

Checkpoint layout (written by backend/scripts/train_rrl_final.py):
    <dir>/rrl_meta.json   {"kind": "ensemble", "classifier": ..., "encoder": ..., "sequence": ..., ...}
    <dir>/classifier/     an RRLClassifier checkpoint
    <dir>/encoder/        the encoder whose vectors the BiLSTM was trained on
    <dir>/sequence.pt     {"dim_in", "hidden", "state_dicts": [one per seed]}
"""

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from app.annotation.label_scheme import LabelScheme
from app.labeling_assistant.dataset import RRLExample, is_human_checked
from app.labeling_assistant.models import Prediction

LABELS = list(LabelScheme.LABELS)
EMB_MAX_LENGTH = 128


class SeqTagger(nn.Module):
    def __init__(self, dim_in: int, hidden: int = 128, n_labels: int = len(LABELS), dropout: float = 0.3) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim_in, 256), nn.GELU())
        self.lstm = nn.LSTM(256, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_labels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (1, n_sentences, dim_in)
        h, _ = self.lstm(self.proj(x))
        return self.out(h)


@torch.no_grad()
def embed_texts(texts: Sequence[str], model: Any, tokenizer: Any, device: str, batch_size: int = 64) -> np.ndarray:
    """Mean-pooled last-layer vectors, in the order given (batched by length for speed)."""
    model.eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=EMB_MAX_LENGTH, padding=True,
                        return_tensors="pt").to(device)
        hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        out[idx] = ((hidden * mask).sum(1) / mask.sum(1)).float().cpu().numpy()
    return out


def position_features(n: int) -> np.ndarray:
    rel = np.arange(n, dtype=np.float32) / max(1, n - 1)
    return np.stack([rel, np.sin(np.pi * rel), np.cos(np.pi * rel)], axis=1)


def target(s: Any) -> int:
    """Label id, or -100 (ignored by the loss): our sentences count only when a person checked them;
    public-corpus sentences count when their role maps to one of ours."""
    checked = is_human_checked(s) if hasattr(s, "source") else s.label is not None
    return LABELS.index(s.label) if checked and s.label in LABELS else -100


def doc_tensors(case: Sequence[Any], emb: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """One judgment (sentences in order): features and label ids (-100 = not trained on)."""
    x = np.concatenate([np.stack([emb[s.sentence_id] for s in case]), position_features(len(case))], axis=1)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor([target(s) for s in case])


def macro_f1(gold: Sequence[int], probs: np.ndarray) -> float:
    if not len(gold):
        return 0.0
    pred, gold_a = probs.argmax(1), np.array(gold)
    f1s = []
    for c in range(len(LABELS)):
        tp = int(((pred == c) & (gold_a == c)).sum())
        fp = int(((pred == c) & (gold_a != c)).sum())
        fn = int(((pred != c) & (gold_a == c)).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s))


def predict_docs(model: SeqTagger, docs, device: str) -> tuple[list[int], np.ndarray]:
    model.eval()
    gold, probs = [], []
    with torch.no_grad():
        for x, y in docs:
            p = torch.softmax(model(x.to(device)).squeeze(0), -1).cpu().numpy()
            keep = (y >= 0).numpy()
            gold += y[keep].tolist()
            probs.append(p[keep])
    return gold, (np.concatenate(probs) if probs else np.zeros((0, len(LABELS))))


def train_tagger(docs_fit, docs_val, dim_in: int, seed: int, device: str, epochs: int = 60, patience: int = 10,
                 init: dict | None = None) -> SeqTagger:
    """Masked cross-entropy (square-root class weights), AdamW, early stopping on validation macro-F1."""
    torch.manual_seed(seed)
    random.seed(seed)
    counts = np.bincount(torch.cat([y[y >= 0] for _, y in docs_fit]).numpy(), minlength=len(LABELS))
    weights = torch.tensor((counts.sum() / (len(LABELS) * np.maximum(counts, 1))) ** 0.5, dtype=torch.float32, device=device)
    model = SeqTagger(dim_in).to(device)
    if init is not None:
        model.load_state_dict(init)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    best, best_state, bad = -1.0, None, 0
    docs_fit = list(docs_fit)
    for _ in range(epochs):
        model.train()
        random.shuffle(docs_fit)
        for x, y in docs_fit:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)).squeeze(0), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        f1 = macro_f1(*predict_docs(model, docs_val, device))
        if f1 > best:
            best, best_state, bad = f1, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def to_predictions(labels: Sequence[str], examples: Sequence[RRLExample], probs: np.ndarray) -> list[Prediction]:
    out = []
    for e, p in zip(examples, probs):
        order = np.argsort(p)[::-1]
        top, second = float(p[order[0]]), float(p[order[1]]) if len(p) > 1 else 0.0
        out.append(Prediction(sentence_id=e.sentence_id, label=labels[int(order[0])],
                              probabilities={lab: round(float(v), 6) for lab, v in zip(labels, p)},
                              confidence=round(top, 6), margin=round(top - second, 6),
                              case_id=e.case_id, idx=e.idx, text=e.text))
    return out


class EnsembleClassifier:
    """Fine-tuned classifier + BiLSTM(s), averaged and temperature-scaled. Same predict() as RRLClassifier."""

    kind = "ensemble"

    def __init__(self, classifier: Any, encoder: Any, encoder_tokenizer: Any, taggers: Sequence[SeqTagger],
                 weight: float = 0.5, temperature: float = 1.0) -> None:
        self.classifier, self.encoder, self.encoder_tokenizer = classifier, encoder, encoder_tokenizer
        self.taggers = list(taggers)
        self.weight, self.ens_temperature = weight, temperature
        self.encoder.to(self.device)
        for t in self.taggers:
            t.to(self.device).eval()

    # the explainer (explain.py) works on the fine-tuned member
    labels = property(lambda self: self.classifier.labels)
    tokenizer = property(lambda self: self.classifier.tokenizer)
    model = property(lambda self: self.classifier.model)
    context = property(lambda self: self.classifier.context)
    max_length = property(lambda self: self.classifier.max_length)
    device = property(lambda self: self.classifier.device)
    temperature = property(lambda self: self.classifier.temperature)

    def _sequence_probs(self, examples: Sequence[RRLExample]) -> np.ndarray:
        """BiLSTM probabilities, one judgment at a time, rows in the order of `examples`."""
        out = np.zeros((len(examples), len(self.labels)), dtype=np.float32)
        groups: dict[str, list[int]] = {}
        for i, e in enumerate(examples):
            groups.setdefault(e.case_id or "", []).append(i)
        for rows in groups.values():
            rows = sorted(rows, key=lambda i: (examples[i].idx, i))
            vecs = embed_texts([examples[i].text for i in rows], self.encoder, self.encoder_tokenizer, self.device)
            x = torch.tensor(np.concatenate([vecs, position_features(len(rows))], axis=1)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                p = np.mean([torch.softmax(t(x).squeeze(0), -1).cpu().numpy() for t in self.taggers], axis=0)
            out[rows] = p
        return out

    def probabilities(self, examples: Sequence[RRLExample], temperature: float | None = None) -> np.ndarray:
        unlabelled = [RRLExample(e.sentence_id, e.case_id, e.idx, e.text, None) for e in examples]
        p1 = self.classifier._probabilities(self.classifier.dataset(unlabelled))
        p = self.weight * p1 + (1 - self.weight) * self._sequence_probs(unlabelled)
        t = self.ens_temperature if temperature is None else temperature
        logp = np.log(np.clip(p, 1e-9, 1)) / t
        logp -= logp.max(1, keepdims=True)
        q = np.exp(logp)
        return q / q.sum(1, keepdims=True)

    def predict(self, sentences: Sequence[RRLExample] | Sequence[tuple[str, str]]) -> list[Prediction]:
        examples = [s if isinstance(s, RRLExample) else RRLExample(s[0], "", -1, s[1]) for s in sentences]
        if not examples:
            return []
        return to_predictions(self.labels, examples, self.probabilities(examples))

    def calibrate(self, examples: Sequence[RRLExample]) -> float:
        """One temperature on the averaged probabilities, fitted on labelled held-out sentences."""
        p = self.probabilities(examples, temperature=1.0)
        gold = torch.tensor([self.labels.index(e.label) for e in examples])
        logp = torch.log(torch.from_numpy(np.clip(p, 1e-9, 1)).float())
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(logp / log_t.exp(), gold)
            loss.backward()
            return loss

        opt.step(closure)
        self.ens_temperature = float(log_t.detach().exp())
        return self.ens_temperature

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "EnsembleClassifier":
        from transformers import AutoModel, AutoTokenizer

        from app.labeling_assistant.classifier import RRLClassifier

        path = Path(path)
        meta = json.loads((path / "rrl_meta.json").read_text(encoding="utf-8"))
        clf = RRLClassifier.load(path / meta["classifier"], device=device)
        encoder = AutoModel.from_pretrained(path / meta["encoder"])
        tok = AutoTokenizer.from_pretrained(path / meta["encoder"])
        seq = torch.load(path / meta["sequence"], map_location="cpu")
        taggers = []
        for state in seq["state_dicts"]:
            t = SeqTagger(seq["dim_in"], hidden=seq.get("hidden", 128))
            t.load_state_dict(state)
            taggers.append(t)
        ens = cls(clf, encoder, tok, taggers, weight=meta.get("weight", 0.5), temperature=meta.get("temperature", 1.0))
        ens.meta = meta  # type: ignore[attr-defined]
        return ens


def is_ensemble(checkpoint: str | Path) -> bool:
    meta = Path(checkpoint) / "rrl_meta.json"
    return meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("kind") == "ensemble"
