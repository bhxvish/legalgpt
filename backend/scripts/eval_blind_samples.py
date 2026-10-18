"""InLegalBERT's accuracy on blind samples: labels the model could not influence.

About 1 uploaded judgment in 10 (REVIEW_BLIND_RATE; a stable hash of the case id, see
app/labeling_assistant/router.py) is labelled in the Label tab with the suggestions hidden. Here those
labels (by people, source "manual"; never "model:*" or "claude") are compared with what the model
predicted for the same sentences when the judgment was uploaded (the review queue).

Reported, pooled over blind-sample judgments: sentences compared, accuracy, macro-F1, per-role F1,
and accuracy on the sentences the model was confident about (the ones a reviewer is not asked to
check: confidence >= the queue's threshold) vs the rest. Judgments sent back from review and then
labelled by hand are listed separately: they were chosen *because* the model did badly, so they are
not an estimate of its accuracy.

Usage:  python backend/scripts/eval_blind_samples.py [--out data/reports/blind_samples.json]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import REPO_ROOT, get_settings  # noqa: E402

AI = ("model:", "claude")


def human_labels(store, case_id: str) -> dict[str, list]:
    """annotator -> that person's latest manual labels on the case."""
    out = {}
    for name in store.annotators(case_id):
        if name.startswith(AI):
            continue
        labels = [s for s in store.load_case(case_id, annotator=name) if s.source == "manual"]
        if labels:
            out[name] = labels
    return out


def compare(store, queue, case_id: str) -> list[dict]:
    case = queue.load(case_id)
    pred = {it.prediction.sentence_id: it.prediction for it in case.items}
    rows = []
    for name, labels in human_labels(store, case_id).items():
        for s in labels:
            p = pred.get(s.sentence_id)
            if p is not None:
                rows.append({"case": case_id, "annotator": name, "gold": s.label, "pred": p.label,
                             "confident": p.confidence >= case.tau_conf})
    return rows


def scores(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    labels = sorted({r["gold"] for r in rows} | {r["pred"] for r in rows})
    f1 = {}
    for lab in labels:
        tp = sum(r["gold"] == r["pred"] == lab for r in rows)
        fp = sum(r["pred"] == lab != r["gold"] for r in rows)
        fn = sum(r["gold"] == lab != r["pred"] for r in rows)
        f1[lab] = round(2 * tp / (2 * tp + fp + fn), 3) if tp + fp + fn else None
    gold_labels = [lab for lab in labels if any(r["gold"] == lab for r in rows)]
    acc = lambda rs: round(sum(r["gold"] == r["pred"] for r in rs) / len(rs), 3) if rs else None  # noqa: E731
    return {
        "n": len(rows), "judgments": len({r["case"] for r in rows}), "accuracy": acc(rows),
        "macro_f1": round(sum(f1[lab] or 0 for lab in gold_labels) / len(gold_labels), 3),
        "per_role_f1": f1,
        "confident": {"n": sum(r["confident"] for r in rows), "accuracy": acc([r for r in rows if r["confident"]])},
        "flagged": {"n": sum(not r["confident"] for r in rows), "accuracy": acc([r for r in rows if not r["confident"]])},
        "confusion": dict(Counter(f"{r['gold']} -> {r['pred']}" for r in rows if r["gold"] != r["pred"]).most_common(8)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "reports" / "blind_samples.json")
    args = ap.parse_args()
    from app.annotation.store import AnnotationStore
    from app.labeling_assistant.review_queue import HumanReviewQueue
    from app.labeling_assistant.router import is_blind_sample

    s = get_settings()
    store = AnnotationStore(s.annotation_store_dir)
    queue = HumanReviewQueue(store, s.review_max_audit_error)
    groups: dict[str, list[str]] = defaultdict(list)
    for cid in queue.case_ids():
        # a judgment sent back was first checked with the suggestions visible: never a blind sample
        if queue.load(cid).status == "escalated":
            groups["sent_back"].append(cid)
        elif is_blind_sample(cid, s.review_blind_rate):
            groups["blind"].append(cid)
    report = {"blind_rate": s.review_blind_rate, "blind_samples": groups["blind"],
              "blind": scores([r for c in groups["blind"] for r in compare(store, queue, c)]),
              "sent_back": scores([r for c in groups["sent_back"] for r in compare(store, queue, c)])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    b = report["blind"]
    print(f"blind samples in the queue: {len(groups['blind'])} ({', '.join(groups['blind']) or 'none yet'})")
    if not b["n"]:
        print("no blind-sample labels yet: open a 'Blind sample' judgment in the Label tab and label it; then run this again")
    else:
        print(f"blind: {b['n']} sentences in {b['judgments']} judgment(s): accuracy {b['accuracy']}, macro-F1 {b['macro_f1']}")
        print(f"  confident (not shown to reviewers): {b['confident']}; flagged: {b['flagged']}")
        print(f"  per role: {b['per_role_f1']}")
    sb = report["sent_back"]
    if sb["n"]:
        print(f"sent back, then labelled by hand (selected for poor model output; not an accuracy estimate): "
              f"{sb['n']} sentences, accuracy {sb['accuracy']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
