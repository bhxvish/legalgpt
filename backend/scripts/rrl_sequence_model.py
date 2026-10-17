"""Sentence-role labelling as a sequence task: a BiLSTM reads a whole judgment in order.

The fine-tuned classifier (train_rrl_classifier.py) labels each sentence with at most one
neighbour on each side. This model reads every sentence of the judgment in order:
  1. each sentence becomes a vector from InLegalBERT (frozen, mean-pooled; computed once and cached),
     plus its relative position in the judgment;
  2. a 2-layer BiLSTM over the judgment's sentences, then a per-sentence softmax.
Only human-checked labels contribute to the loss; unchecked sentences are still read as context.

Evaluation uses exactly the case folds of a cross-validation file from train_rrl_classifier.py
(--folds-from), so the numbers are comparable fold by fold. With --ensemble-with, the held-out
probabilities of that run's fine-tuned classifier are averaged with this model's (two
independently trained models; no stage-2 training on stage-1 outputs, so no leakage).

Options for the transfer experiment: --encoder takes the sentence vectors from another checkpoint
(e.g. the one fine-tuned on the public corpus, pretrain_rrl_public.py), and --pretrain-public first
trains the BiLSTM on the public judgments (data/rrl_public) and starts every fold from those weights.

Usage (from the repo root):
    .venv-gpu\\Scripts\\python backend/scripts/rrl_sequence_model.py --version v0.3 \\
        --folds-from data/models/rrl_cv/v0.3-cv5-abl-full-*.json [--ensemble-with <same file>]
"""

import argparse
import glob
import json
import random
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation.label_scheme import LabelScheme  # noqa: E402
from app.annotation.store import AnnotationStore  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.labeling_assistant.classifier import DEFAULT_MODEL  # noqa: E402
from app.labeling_assistant.dataset import is_human_checked  # noqa: E402
from app.labeling_assistant.evaluator import ClassifierEvaluator  # noqa: E402

LABELS = list(LabelScheme.LABELS)


def embed(sentences, model_name: str, cache: Path, device: str, max_length: int = 128) -> dict[str, np.ndarray]:
    """sentence_id -> mean-pooled last-layer InLegalBERT vector (frozen), cached to `cache`."""
    if cache.exists():
        z = np.load(cache)
        have = dict(zip(z["ids"].tolist(), z["vecs"]))
        if all(s.sentence_id in have for s in sentences):
            return have
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    ids, vecs = [], []
    order = sorted(sentences, key=lambda s: len(s.text))  # similar lengths per batch: less padding
    with torch.no_grad():
        for i in range(0, len(order), 64):
            batch = order[i : i + 64]
            enc = tok([s.text for s in batch], truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
            out = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1)
            ids += [s.sentence_id for s in batch]
            vecs.append(pooled.float().cpu().numpy())
    vecs_arr = np.concatenate(vecs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=np.array(ids), vecs=vecs_arr)
    return dict(zip(ids, vecs_arr))


class SeqTagger(nn.Module):
    def __init__(self, dim_in: int, hidden: int = 128, n_labels: int = len(LABELS), dropout: float = 0.3) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim_in, 256), nn.GELU())
        self.lstm = nn.LSTM(256, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_labels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (1, n_sentences, dim_in)
        h, _ = self.lstm(self.proj(x))
        return self.out(h)


def target(s) -> int:
    """Label id, or -100 (ignored by the loss): our sentences count only when a person checked them;
    public sentences count when their role maps to one of ours."""
    checked = is_human_checked(s) if hasattr(s, "source") else s.label is not None
    return LABELS.index(s.label) if checked and s.label in LABELS else -100


def doc_tensors(case, emb):
    """One judgment: features (sentence vector + relative position) and label ids (-100 = not trained on)."""
    n = len(case)
    rel = np.arange(n, dtype=np.float32) / max(1, n - 1)
    pos = np.stack([rel, np.sin(np.pi * rel), np.cos(np.pi * rel)], axis=1)
    x = np.concatenate([np.stack([emb[s.sentence_id] for s in case]), pos], axis=1)
    y = [target(s) for s in case]
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor(y)


def train_fold(docs_fit, docs_val, dim_in, seed, device, epochs=60, patience=10, init=None):
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


def predict_docs(model, docs, device):
    model.eval()
    gold, probs = [], []
    with torch.no_grad():
        for x, y in docs:
            p = torch.softmax(model(x.to(device)).squeeze(0), -1).cpu().numpy()
            keep = (y >= 0).numpy()
            gold += y[keep].tolist()
            probs.append(p[keep])
    return gold, (np.concatenate(probs) if probs else np.zeros((0, len(LABELS))))


def macro_f1(gold, probs):
    if not len(gold):
        return 0.0
    pred = probs.argmax(1)
    f1s = []
    for c in range(len(LABELS)):
        tp = int(((pred == c) & (np.array(gold) == c)).sum())
        fp = int(((pred == c) & (np.array(gold) != c)).sum())
        fn = int(((pred != c) & (np.array(gold) == c)).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v0.3")
    ap.add_argument("--folds-from", required=True, help="a cross-validation JSON from train_rrl_classifier.py (glob ok)")
    ap.add_argument("--ensemble-with", default=None, help="a CV JSON saved with --save-predictions (glob ok)")
    ap.add_argument("--seeds", type=int, default=3, help="independent runs per fold; probabilities are averaged")
    ap.add_argument("--val-cases", type=int, default=3)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="encoder for the sentence vectors")
    ap.add_argument("--encoder", default=None, help="alias of --model (a fine-tuned checkpoint directory works too)")
    ap.add_argument("--pretrain-public", type=Path, default=None, help="data/rrl_public: pretrain the BiLSTM on the public judgments")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    s = get_settings()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, sentences = AnnotationStore(s.annotation_store_dir).load_version(args.version, verify=True)
    cases: dict[str, list] = {}
    for x in sentences:
        cases.setdefault(x.case_id, []).append(x)
    for c in cases.values():
        c.sort(key=lambda x: x.idx)
    encoder = args.encoder or args.model
    enc_tag = "" if encoder == DEFAULT_MODEL else "-" + Path(encoder).name
    public = {}
    if args.pretrain_public:
        from app.labeling_assistant.dataset import RRLExample

        for split in ("train", "dev"):
            rows = [json.loads(x) for x in (args.pretrain_public / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
            for r in rows:
                public.setdefault(split, {}).setdefault(r["case_id"], []).append(
                    RRLExample(r["sentence_id"], r["case_id"], r["idx"], r["text"], r["label"]))
    all_sentences = list(sentences) + [e for split in public.values() for doc in split.values() for e in doc]
    emb = embed(all_sentences, encoder, s.rrl_model_dir.parent / "rrl_seq" / f"emb-{args.version}{enc_tag}{'-pub' if public else ''}.npz", device)
    dim_in = next(iter(emb.values())).shape[0] + 3

    init = None
    if public:
        pub_fit = [doc_tensors(sorted(d, key=lambda e: e.idx), emb) for d in public["train"].values()]
        pub_val = [doc_tensors(sorted(d, key=lambda e: e.idx), emb) for d in public["dev"].values()]
        pre = train_fold(pub_fit, pub_val, dim_in, 7, device, epochs=30, patience=5)
        init = {k: v.detach().clone() for k, v in pre.state_dict().items()}
        print(f"BiLSTM pretrained on {len(pub_fit)} public judgments (dev macro-F1 {macro_f1(*predict_docs(pre, pub_val, device)):.3f})")

    folds_file = sorted(glob.glob(args.folds_from))[-1]
    fold_cases = json.load(open(folds_file, encoding="utf-8"))["fold_cases"]
    ens = None
    if args.ensemble_with:
        ens_file = sorted(glob.glob(args.ensemble_with))[-1]
        ens = {p["sentence_id"]: p["probabilities"] for p in json.load(open(ens_file, encoding="utf-8"))["predictions"]}

    evaluator = ClassifierEvaluator()
    rows = {"sequence": [], "ensemble": []}
    saved = []
    for k, test_cases in enumerate(fold_cases):
        train_cases = sorted(c for c in cases if c not in test_cases)
        val = set(random.Random(13 + k).sample(train_cases, args.val_cases))
        fit_docs = [doc_tensors(cases[c], emb) for c in train_cases if c not in val]
        val_docs = [doc_tensors(cases[c], emb) for c in val]
        test_docs = [doc_tensors(cases[c], emb) for c in test_cases]
        probs_sum = None
        for seed in range(args.seeds):
            model = train_fold(list(fit_docs), val_docs, dim_in, 1000 * k + seed, device, init=init)
            gold, probs = predict_docs(model, test_docs, device)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        probs = probs_sum / args.seeds
        ids = [x.sentence_id for c in test_cases for x in cases[c] if is_human_checked(x) and x.label in LABELS]
        gold_labels = [LABELS[g] for g in gold]
        rows["sequence"].append(evaluator.evaluate(gold_labels, [LABELS[i] for i in probs.argmax(1)], LABELS))
        for sid, g, p in zip(ids, gold_labels, probs):
            rec = {"sentence_id": sid, "gold": g, "probabilities": dict(zip(LABELS, map(float, p)))}
            if ens:
                q = np.array([ens[sid][lab] for lab in LABELS])
                rec["ensemble"] = dict(zip(LABELS, map(float, (p + q) / 2)))
            saved.append(rec)
        if ens:
            fold_recs = saved[-len(ids):]
            rows["ensemble"].append(evaluator.evaluate(gold_labels, [max(r["ensemble"], key=r["ensemble"].get) for r in fold_recs], LABELS))
        print(f"fold {k + 1}: sequence macro-F1 {rows['sequence'][-1].macro_f1:.3f}"
              + (f" | ensemble {rows['ensemble'][-1].macro_f1:.3f}" if ens else ""))

    summary = {}
    for name, ms in rows.items():
        if not ms:
            continue
        f1 = [m.macro_f1 for m in ms]
        acc = [m.accuracy for m in ms]
        summary[name] = {"macro_f1_mean": st.mean(f1), "macro_f1_sd": st.stdev(f1), "accuracy_mean": st.mean(acc),
                         "folds": [m.to_dict() for m in ms]}
        print(f"{name}: macro-F1 {st.mean(f1):.3f} ± {st.stdev(f1):.3f}, accuracy {st.mean(acc):.3f}")
    tag = f"-{args.tag}" if args.tag else ""
    out = s.rrl_model_dir.parent / "rrl_cv" / f"{args.version}-cv{len(fold_cases)}-sequence{tag}-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"version": args.version, "folds_from": folds_file, "ensemble_with": args.ensemble_with and ens_file,
                               "seeds": args.seeds, "fold_cases": fold_cases, "summary": summary, "predictions": saved}, indent=1),
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
