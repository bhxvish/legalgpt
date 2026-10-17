"""Train the deployable sentence-role ensemble (paper_analysis.md §7b) on all annotated judgments.

  1. classifier: the public-corpus checkpoint (pretrain_rrl_public.py), fine-tuned on our judgments
     (context, 256 tokens, 6 epochs, best epoch on the validation judgments, temperature-calibrated)
  2. encoder:    the public-corpus checkpoint itself, whose vectors the BiLSTM reads
  3. sequence:   BiLSTM pretrained on the public judgments, then trained on ours (3 seeds, averaged)
  4. ensemble:   50/50 average, one temperature fitted on the validation judgments
The snapshot's validation split (4 judgments) is held out of training for epoch choice, early
stopping and calibration; every other judgment (train + test splits, 32) is trained on. The
review-load table on the validation judgments is printed and stored.

Usage (Colab GPU, after build_rrl_public.py and pretrain_rrl_public.py):
    python backend/scripts/train_rrl_final.py --stage-a data/models/rrl_public/inrr-<time> --bf16
Writes data/models/rrl/<version>-ensemble-<time>/ (the app loads the newest checkpoint there).
"""

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation.label_scheme import LabelScheme  # noqa: E402
from app.annotation.store import AnnotationStore  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.labeling_assistant.classifier import RRLClassifier, TrainConfig  # noqa: E402
from app.labeling_assistant.dataset import RRLDataset, RRLExample, is_human_checked  # noqa: E402
from app.labeling_assistant.evaluator import ClassifierEvaluator, format_review_load, review_load  # noqa: E402
from app.labeling_assistant.sequence import (  # noqa: E402
    EnsembleClassifier, SeqTagger, doc_tensors, embed_texts, macro_f1, predict_docs, train_tagger)

LABELS = list(LabelScheme.LABELS)


def public_docs(folder: Path, split: str) -> dict[str, list[RRLExample]]:
    docs: dict[str, list[RRLExample]] = {}
    for line in (folder / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            docs.setdefault(r["case_id"], []).append(RRLExample(r["sentence_id"], r["case_id"], r["idx"], r["text"], r["label"]))
    return {k: sorted(v, key=lambda e: e.idx) for k, v in docs.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v0.3")
    ap.add_argument("--stage-a", type=Path, required=True, help="public-corpus checkpoint")
    ap.add_argument("--public", type=Path, default=None, help="data/rrl_public (default: next to the repo data)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="smoke test: cap the fine-tuning steps")
    ap.add_argument("--seq-epochs", type=int, default=60, help="smoke test: fewer BiLSTM epochs")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    s = get_settings()
    public = args.public or s.rrl_model_dir.parents[1] / "rrl_public"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest, sentences = AnnotationStore(s.annotation_store_dir).load_version(args.version, verify=True)
    val_cases = set(manifest.splits["val"])
    pool = RRLDataset.examples_from(sentences)
    checked = [e for e, x in zip(pool, sentences) if is_human_checked(x)]
    fit = [e for e in checked if e.case_id not in val_cases]
    val = [e for e in checked if e.case_id in val_cases]
    print(f"{args.version}: {len({e.case_id for e in fit})} judgments trained on ({len(fit)} sentences), "
          f"{len(val_cases)} held out for validation ({len(val)}): {sorted(val_cases)}")
    out = args.out or s.rrl_model_dir / f"{args.version}-ensemble-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    # 1. fine-tuned classifier, starting from the public-corpus checkpoint
    clf = RRLClassifier(LABELS, model_name=str(args.stage_a), max_length=256, context=True)
    cfg = TrainConfig(epochs=6, batch_size=16, lr=3e-5, freeze_layers=0, bf16=args.bf16, max_steps=args.max_steps)
    report = clf.train(clf.dataset(fit, pool), clf.dataset(val, pool), cfg)
    t_clf = clf.calibrate(clf.dataset(val, pool))
    clf_metrics, _ = clf.evaluate(clf.dataset(val, pool))
    clf.save(out / "classifier", extra={"stage": "fine-tuned from the public-corpus checkpoint", "config": asdict(cfg)})
    print(f"classifier: best epoch {report.best_epoch}, temperature {t_clf:.2f}, validation macro-F1 {clf_metrics.macro_f1:.3f}")

    # 2. encoder = the public-corpus checkpoint
    shutil.copytree(args.stage_a, out / "encoder", dirs_exist_ok=True)
    from transformers import AutoModel, AutoTokenizer

    encoder = AutoModel.from_pretrained(out / "encoder").to(device)
    enc_tok = AutoTokenizer.from_pretrained(out / "encoder")

    # 3. BiLSTM: pretrain on the public judgments, then train on ours
    pub = {split: public_docs(public, split) for split in ("train", "dev")}
    ours: dict[str, list[RRLExample]] = {}
    for e, x in zip(pool, sentences):
        ours.setdefault(e.case_id, []).append(x)  # LabeledSentence: target() uses its review status
    ours = {k: sorted(v, key=lambda x: x.idx) for k, v in ours.items()}
    every = [x for d in ours.values() for x in d] + [e for split in pub.values() for d in split.values() for e in d]
    vecs = embed_texts([x.text for x in every], encoder, enc_tok, device)
    emb = dict(zip([x.sentence_id for x in every], vecs))
    dim_in = vecs.shape[1] + 3
    pre = train_tagger([doc_tensors(d, emb) for d in pub["train"].values()],
                       [doc_tensors(d, emb) for d in pub["dev"].values()], dim_in, 7, device,
                       epochs=min(30, args.seq_epochs), patience=5)
    init = {k: v.detach().clone() for k, v in pre.state_dict().items()}
    fit_docs = [doc_tensors(d, emb) for c, d in ours.items() if c not in val_cases]
    val_docs = [doc_tensors(d, emb) for c, d in ours.items() if c in val_cases]
    taggers = [train_tagger(fit_docs, val_docs, dim_in, 100 + k, device, epochs=args.seq_epochs, init=init)
               for k in range(args.seeds)]
    seq_f1 = macro_f1(*predict_docs(taggers[0], val_docs, device))
    torch.save({"dim_in": dim_in, "hidden": 128, "state_dicts": [t.state_dict() for t in taggers]}, out / "sequence.pt")
    print(f"sequence: {len(taggers)} BiLSTMs, validation macro-F1 (first seed) {seq_f1:.3f}")

    # 4. ensemble: one temperature on the validation judgments, then the review-load table
    taggers_cpu = []
    for t in taggers:
        u = SeqTagger(dim_in)
        u.load_state_dict(t.state_dict())
        taggers_cpu.append(u)
    ens = EnsembleClassifier(clf, encoder, enc_tok, taggers_cpu)
    t_ens = ens.calibrate(val)
    probs = ens.probabilities(val)
    gold = [e.label for e in val]
    pred = [LABELS[i] for i in probs.argmax(1)]
    ens_metrics = ClassifierEvaluator().evaluate(gold, pred, LABELS)
    rows = review_load([float(p.max()) for p in probs], [p == g for p, g in zip(pred, gold)])
    print(f"\nensemble on the validation judgments (temperature {t_ens:.2f}):\n{ens_metrics.format()}\n")
    print(format_review_load(rows))

    meta = {"kind": "ensemble", "labels": LABELS, "classifier": "classifier", "encoder": "encoder",
            "sequence": "sequence.pt", "weight": 0.5, "temperature": t_ens, "version": args.version,
            "corpus_sha256": manifest.sha256, "val_cases": sorted(val_cases),
            "public_corpus": "InRhetoricalRoles (Kalamkar et al., LREC 2022), CC BY-SA 4.0",
            "cv_macro_f1": 0.807, "cv_reference": "docs/reports/paper_analysis.md section 7b",
            "base_model": str(args.stage_a), "max_length": 256, "context": True}
    (out / "rrl_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({
        "classifier_val": clf_metrics.to_dict(), "classifier_temperature": t_clf, "sequence_val_macro_f1": seq_f1,
        "ensemble_val": ens_metrics.to_dict(), "ensemble_temperature": t_ens,
        "review_load": [asdict(r) for r in rows]}, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
