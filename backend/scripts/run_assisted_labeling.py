"""Predict rhetorical roles for unlabelled judgments and queue them for human review.

For every collected judgment that has no labels yet and is not already queued: segment it,
predict each sentence's role with the trained classifier, then let ReviewSelector flag
low-confidence predictions (always reviewed) and a random audit sample of confident ones.
Review them in the app's Review tab, then sign off each case.

Usage (from the repo root):
    python backend/scripts/run_assisted_labeling.py                       # newest checkpoint, all new cases
    python backend/scripts/run_assisted_labeling.py --cases sc-2023-rajo --tau 0.8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation.collector import JudgmentCollector  # noqa: E402
from app.annotation.models import sentence_id  # noqa: E402
from app.annotation.store import AnnotationStore  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.labeling_assistant.classifier import RRLClassifier  # noqa: E402
from app.labeling_assistant.dataset import RRLExample  # noqa: E402
from app.labeling_assistant.segmenter import AssistedSegmenter  # noqa: E402
from app.labeling_assistant.review_queue import HumanReviewQueue  # noqa: E402
from app.labeling_assistant.review_selector import ReviewSelector  # noqa: E402


def latest_checkpoint(model_dir: Path) -> Path:
    runs = sorted((p for p in model_dir.glob("*") if (p / "rrl_meta.json").exists()), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit(f"no trained checkpoint in {model_dir}; run backend/scripts/train_rrl_classifier.py first")
    return runs[-1]


def main() -> int:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=None, help="checkpoint directory (default: newest)")
    ap.add_argument("--cases", nargs="*", help="case ids (default: every unlabelled, unqueued case)")
    ap.add_argument("--tau", type=float, default=s.review_tau_conf, help="confidence below which review is mandatory")
    ap.add_argument("--audit-rate", type=float, default=s.review_audit_rate, help="share of confident predictions audited")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    collector, store = JudgmentCollector(s.raw_judgments_dir), AnnotationStore(s.annotation_store_dir)
    queue = HumanReviewQueue(store, s.review_max_audit_error)
    labelled, queued = set(store.case_ids()), set(queue.case_ids())
    cases = args.cases or [e["case_id"] for e in collector.index() if e["case_id"] not in labelled | queued]
    if not cases:
        print("nothing to do: every collected judgment is already labelled or queued")
        return 0

    checkpoint = args.model or latest_checkpoint(s.rrl_model_dir)
    clf = RRLClassifier.load(checkpoint)
    selector = ReviewSelector(tau_conf=args.tau, audit_rate=args.audit_rate, seed=args.seed)
    segmenter = AssistedSegmenter()
    print(f"model {checkpoint.name} (labels {clf.labels}); tau_conf {args.tau}, audit rate {args.audit_rate:.0%}\n")

    for case_id in cases:
        if case_id in labelled or case_id in queued:
            print(f"SKIP   {case_id}: already {'labelled' if case_id in labelled else 'queued'}")
            continue
        sentences = store.load_segments(case_id) or segmenter.segment(collector.load(case_id).raw_text, case_id)
        examples = [RRLExample(sentence_id(case_id, i), case_id, i, t) for i, t in enumerate(sentences)]
        predictions = clf.predict(examples)
        batch = selector.select(predictions, case_id)
        queue.enqueue(batch, model=checkpoint.name)
        confs = sorted(p.confidence for p in predictions)
        print(f"QUEUED {case_id}: {len(predictions)} sentences -> {len(batch.mandatory)} low-confidence (review), "
              f"{len(batch.audit)} spot-checks, {len(batch.auto)} accepted unless the spot-check fails; "
              f"median confidence {confs[len(confs) // 2]:.2f}")
    print("\nReview them in the app: Review tab (http://localhost:5173/#review)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
