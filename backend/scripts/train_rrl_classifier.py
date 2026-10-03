"""Fine-tune InLegalBERT on a frozen corpus version and report held-out metrics.

Trains on the version's train split, keeps the epoch with the best validation macro-F1, fits the
confidence temperature on the validation cases, then evaluates on the test split (cases never seen
in training): accuracy, per-class precision/recall/F1, macro-F1, the confusion matrix, and the
review-load table (how much mandatory review each confidence bar would leave, and how accurate the
skipped labels are). Saves the checkpoint plus metrics.json.

Only human-checked labels are trained and evaluated on (manual annotation, or assisted sentences a
reviewer saw); unchecked assisted labels still serve as neighbouring context.

Usage (from the repo root; the GPU environment is much faster):
    python backend/scripts/train_rrl_classifier.py --version v0.1
    .venv-gpu\\Scripts\\python backend/scripts/train_rrl_classifier.py --version v0.2 --context \\
        --freeze-layers 0 --max-length 256 --epochs 6 --bf16
    python backend/scripts/train_rrl_classifier.py --version v0.1 --max-steps 5      # smoke test
    python backend/scripts/train_rrl_classifier.py --version v0.1 --cv 5             # case-grouped CV only
"""

import argparse
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation.label_scheme import LabelScheme  # noqa: E402
from app.annotation.store import AnnotationStore  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.labeling_assistant.classifier import DEFAULT_MODEL, RRLClassifier, TrainConfig  # noqa: E402
from app.labeling_assistant.dataset import RRLDataset, is_human_checked  # noqa: E402
from app.labeling_assistant.evaluator import ClassifierEvaluator, format_review_load, review_load  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v0.1", help="frozen corpus version to train on")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--freeze-layers", type=int, default=8, help="bottom encoder layers to freeze (0 = none)")
    ap.add_argument("--max-length", type=int, default=128, help="tokens per input (sentence + context)")
    ap.add_argument("--context", action="store_true", help="read each sentence with its neighbours and position")
    ap.add_argument("--include-unreviewed", action="store_true", help="also train on assisted labels no one checked")
    ap.add_argument("--no-calibrate", action="store_true", help="skip temperature scaling of confidences")
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast on CUDA (saves GPU memory)")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps (smoke test)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--cv", type=int, default=0, help="run K-fold case-grouped cross-validation instead of a single split")
    ap.add_argument("--cv-val-cases", type=int, default=3, help="training cases per fold held out for epoch choice + calibration")
    ap.add_argument("--out", type=Path, default=None, help="checkpoint directory (default data/models/rrl/<version>-<time>)")
    args = ap.parse_args()

    s = get_settings()
    store = AnnotationStore(s.annotation_store_dir)
    manifest, sentences = store.load_version(args.version, verify=True)
    labels = list(LabelScheme.LABELS)
    human_only = not args.include_unreviewed
    calibrate = not args.no_calibrate
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, freeze_layers=args.freeze_layers,
                      max_steps=args.max_steps, seed=args.seed, bf16=args.bf16)
    setup = {"context": args.context, "max_length": args.max_length, "human_only": human_only, "calibrate": calibrate}
    corpus = {"version": args.version, "sha256": manifest.sha256, "cases": len(manifest.case_ids)}
    checked = sum(is_human_checked(x) for x in sentences)
    print(f"corpus {args.version} (sha256 {manifest.sha256[:12]}…): {len(manifest.case_ids)} cases, "
          f"{len(sentences)} sentences ({checked} human-checked)")
    print(f"config: {asdict(cfg)}\nsetup: {setup}")

    def new_classifier() -> RRLClassifier:
        return RRLClassifier(labels, model_name=args.model, max_length=args.max_length, context=args.context)

    if args.cv:
        pool = RRLDataset.examples_from(sentences)  # every sentence: neighbours for context
        gold = [e for e, x in zip(pool, sentences) if is_human_checked(x) or not human_only]
        confidences: list[float] = []
        correct: list[bool] = []
        temperatures: list[float] = []

        def train_and_predict(train, test):  # fresh model per fold; test cases are never seen
            train_cases = sorted({e.case_id for e in train})
            val_cases = set(random.Random(args.seed + len(confidences)).sample(train_cases, args.cv_val_cases))
            fit = [e for e in train if e.case_id not in val_cases]
            val = [e for e in train if e.case_id in val_cases]
            ctx = [e for e in pool if e.case_id in {x.case_id for x in train} | {x.case_id for x in test}]
            clf = new_classifier()
            clf.train(clf.dataset(fit, ctx), clf.dataset(val, ctx), cfg)
            if calibrate:
                temperatures.append(clf.calibrate(clf.dataset(val, ctx)))
            test_cases = {e.case_id for e in test}
            preds = clf.predict([e for e in pool if e.case_id in test_cases])  # whole judgments: full context
            by_id = {p.sentence_id: p for p in preds}
            out = [by_id[e.sentence_id] for e in test]
            confidences.extend(p.confidence for p in out)
            correct.extend(p.label == e.label for p, e in zip(out, test))
            return [p.label for p in out]

        report = ClassifierEvaluator().cross_validate(gold, args.cv, args.seed, train_and_predict, labels)
        rows = review_load(confidences, correct)
        print(f"\n=== {args.cv}-fold cross-validation, grouped by case ===")
        print(report.format())
        print(f"\nreview load, pooled over all {len(confidences)} held-out sentences"
              + (f" (temperatures {', '.join(f'{t:.2f}' for t in temperatures)})" if temperatures else ""))
        print(format_review_load(rows))
        out = args.out or s.rrl_model_dir.parent / "rrl_cv" / f"{args.version}-cv{args.cv}-{datetime.now():%Y%m%d-%H%M%S}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"corpus": corpus, "config": asdict(cfg), "setup": setup, "k": args.cv,
                                   "folds": [m.to_dict() for m in report.folds], "fold_cases": report.fold_cases,
                                   "macro_f1_mean_sd": report.macro_f1, "accuracy_mean_sd": report.accuracy,
                                   "temperatures": temperatures, "review_load": [asdict(r) for r in rows]}, indent=2),
                       encoding="utf-8")
        print(f"\nwrote {out}")
        return 0

    clf = new_classifier()
    splits = {name: RRLDataset.from_version(store, args.version, name, clf.tokenizer, labels, args.max_length,
                                            context=args.context, human_only=human_only)
              for name in ("train", "val", "test")}
    train_ds, val_ds, test_ds = splits["train"], splits["val"], splits["test"]
    print(f"splits: train {len(train_ds)} sentences / {len(train_ds.case_ids)} cases, "
          f"val {len(val_ds)} / {len(val_ds.case_ids)}, test {len(test_ds)} / {len(test_ds.case_ids)}")

    report = clf.train(train_ds, val_ds, cfg)
    report.corpus = corpus
    if calibrate:
        print(f"confidence temperature fitted on the validation cases: {clf.calibrate(val_ds):.3f}")
    test_metrics, _ = clf.evaluate(test_ds)
    probs = clf._probabilities(test_ds)
    rows = review_load([float(p.max()) for p in probs],
                       [labels[int(p.argmax())] == e.label for p, e in zip(probs, test_ds.examples)])
    print(f"\n=== HELD-OUT TEST: {', '.join(test_ds.case_ids)} (best epoch {report.best_epoch}) ===")
    print(test_metrics.format())
    print("\nreview load on the test cases:")
    print(format_review_load(rows))

    out = args.out or s.rrl_model_dir / f"{args.version}-{datetime.now():%Y%m%d-%H%M%S}"
    clf.save(out, extra={"corpus": corpus, "setup": setup, "test_metrics": test_metrics.to_dict(),
                         "test_cases": test_ds.case_ids, "review_load": [asdict(r) for r in rows]})
    (out / "metrics.json").write_text(json.dumps({"corpus": corpus, "setup": setup, "train_report": report.to_dict(),
                                                  "test": test_metrics.to_dict(),
                                                  "review_load": [asdict(r) for r in rows]}, indent=2), encoding="utf-8")
    print(f"\nsaved checkpoint to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
