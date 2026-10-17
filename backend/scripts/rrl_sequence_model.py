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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation.label_scheme import LabelScheme  # noqa: E402
from app.annotation.store import AnnotationStore  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.labeling_assistant.classifier import DEFAULT_MODEL  # noqa: E402
from app.labeling_assistant.dataset import is_human_checked  # noqa: E402
from app.labeling_assistant.evaluator import ClassifierEvaluator  # noqa: E402
from app.labeling_assistant.sequence import (  # noqa: E402  (shared with the app's EnsembleClassifier)
    doc_tensors, embed_texts, macro_f1, predict_docs, train_tagger as train_fold)

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
    model = AutoModel.from_pretrained(model_name).to(device)
    ids = [s.sentence_id for s in sentences]
    vecs_arr = embed_texts([s.text for s in sentences], model, tok, device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=np.array(ids), vecs=vecs_arr)
    return dict(zip(ids, vecs_arr))


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
