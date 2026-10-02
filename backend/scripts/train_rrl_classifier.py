"""Fine-tune InLegalBERT on a frozen corpus version and report held-out metrics.

Trains on the version's train split, keeps the epoch with the best validation macro-F1, then
evaluates on the test split (cases never seen in training) and prints accuracy, per-class
precision/recall/F1, macro-F1 and the confusion matrix. Saves the checkpoint plus metrics.json.

Usage (from the repo root):
    python backend/scripts/train_rrl_classifier.py --version v0.1
    python backend/scripts/train_rrl_classifier.py --version v0.1 --max-steps 5      # smoke test
    python backend/scripts/train_rrl_classifier.py --version v0.1 --cv 5 --epochs 2  # case-grouped CV only
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation.label_scheme import LabelScheme  # noqa: E402
from app.annotation.store import AnnotationStore  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.labeling_assistant.classifier import DEFAULT_MODEL, RRLClassifier, TrainConfig  # noqa: E402
from app.labeling_assistant.dataset import RRLDataset  # noqa: E402
from app.labeling_assistant.evaluator import ClassifierEvaluator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v0.1", help="frozen corpus version to train on")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--freeze-layers", type=int, default=8, help="bottom encoder layers to freeze (0 = none)")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps (smoke test)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--cv", type=int, default=0, help="run K-fold case-grouped cross-validation instead of a single split")
    ap.add_argument("--out", type=Path, default=None, help="checkpoint directory (default data/models/rrl/<version>-<time>)")
    args = ap.parse_args()

    s = get_settings()
    store = AnnotationStore(s.annotation_store_dir)
    manifest, sentences = store.load_version(args.version, verify=True)
    labels = list(LabelScheme.LABELS)
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      freeze_layers=args.freeze_layers, max_steps=args.max_steps, seed=args.seed)
    corpus = {"version": args.version, "sha256": manifest.sha256, "cases": len(manifest.case_ids)}
    print(f"corpus {args.version} (sha256 {manifest.sha256[:12]}…): {len(manifest.case_ids)} cases, {len(sentences)} sentences")
    print(f"config: {asdict(cfg)}")

    if args.cv:
        examples = RRLDataset.examples_from(sentences)

        def train_and_predict(train, test):  # fresh model per fold
            clf = RRLClassifier(labels, model_name=args.model)
            clf.train(clf.dataset(train), None, cfg)
            return [p.label for p in clf.predict(test)]

        report = ClassifierEvaluator().cross_validate(examples, args.cv, args.seed, train_and_predict, labels)
        print(f"\n=== {args.cv}-fold cross-validation, grouped by case ===")
        print(report.format())
        out = args.out or s.rrl_model_dir.parent / "rrl_cv" / f"{args.version}-cv{args.cv}-{datetime.now():%Y%m%d-%H%M%S}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"corpus": corpus, "config": asdict(cfg), "k": args.cv,
                                   "folds": [m.to_dict() for m in report.folds], "fold_cases": report.fold_cases,
                                   "macro_f1_mean_sd": report.macro_f1, "accuracy_mean_sd": report.accuracy}, indent=2),
                       encoding="utf-8")
        print(f"\nwrote {out}")
        return 0

    clf = RRLClassifier(labels, model_name=args.model)
    train_ds = RRLDataset.from_version(store, args.version, "train", clf.tokenizer, labels)
    val_ds = RRLDataset.from_version(store, args.version, "val", clf.tokenizer, labels)
    test_ds = RRLDataset.from_version(store, args.version, "test", clf.tokenizer, labels)
    print(f"splits: train {len(train_ds)} sentences / {len(train_ds.case_ids)} cases, "
          f"val {len(val_ds)} / {len(val_ds.case_ids)}, test {len(test_ds)} / {len(test_ds.case_ids)}")

    report = clf.train(train_ds, val_ds, cfg)
    report.corpus = corpus
    test_metrics, _ = clf.evaluate(test_ds)
    print(f"\n=== HELD-OUT TEST: {', '.join(test_ds.case_ids)} (best epoch {report.best_epoch}) ===")
    print(test_metrics.format())

    out = args.out or s.rrl_model_dir / f"{args.version}-{datetime.now():%Y%m%d-%H%M%S}"
    clf.save(out, extra={"corpus": corpus, "test_metrics": test_metrics.to_dict(), "test_cases": test_ds.case_ids})
    (out / "metrics.json").write_text(json.dumps({"corpus": corpus, "train_report": report.to_dict(),
                                                  "test": test_metrics.to_dict()}, indent=2), encoding="utf-8")
    print(f"\nsaved checkpoint to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
