"""Train the verdict classifier and measure the output guard before and after it.

Train: benchmark/verdict_guard/train.jsonl (one model family). The decision threshold is chosen on
the training data by 5-fold cross-validation: the lowest threshold that keeps false removals of
non-verdict sentences at or below --max-fpr. Test: benchmark/verdict_guard/test.jsonl (another
model family; never used for training or the threshold), and the worst-case insertion test
(eval_guard_worst_case.py's sentences inserted into real analyses).

Reported for patterns alone vs patterns + classifier: verdicts caught (recall) and non-verdict
sentences wrongly removed, split into generated hard negatives (legal analysis that discusses guilt,
including what past courts held) and sentences of real analyses.

Usage:  python backend/scripts/eval_verdict_guard.py [--max-fpr 0.02]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import REPO_ROOT, get_settings  # noqa: E402

DATA = REPO_ROOT / "benchmark" / "verdict_guard"


def load(name: str) -> list[dict]:
    return [json.loads(ln) for ln in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]


def rates(rows: list[dict], flagged: list[bool]) -> dict:
    pos = [f for r, f in zip(rows, flagged) if r["verdict"]]
    neg_gen = [f for r, f in zip(rows, flagged) if not r["verdict"] and r["kind"] == "generated"]
    neg_real = [f for r, f in zip(rows, flagged) if not r["verdict"] and r["kind"] == "real"]
    pct = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    return {"verdicts_caught": f"{sum(pos)}/{len(pos)}", "recall": pct(pos),
            "hard_negatives_removed": f"{sum(neg_gen)}/{len(neg_gen)}", "fpr_hard": pct(neg_gen),
            "real_sentences_removed": f"{sum(neg_real)}/{len(neg_real)}", "fpr_real": pct(neg_real)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-fpr", type=float, default=0.02)
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    from app.case_analysis.analyzer import VERDICT_SENTENCE
    from app.case_analysis.verdict_classifier import WEIGHTS, VerdictClassifier
    from app.core.embeddings import EmbeddingService

    s = get_settings()
    emb = EmbeddingService(s.embedding_model)
    train, test = load("train"), load("test")
    x = np.asarray(emb.embed([r["text"] for r in train]))
    y = np.asarray([r["verdict"] for r in train])

    # threshold from out-of-fold probabilities on the training data only
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=7).split(x, y):
        m = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000).fit(x[tr], y[tr])
        oof[va] = m.predict_proba(x[va])[:, 1]
    neg = np.sort(oof[y == 0])
    threshold = float(neg[min(len(neg) - 1, int(np.ceil((1 - args.max_fpr) * len(neg))))]) + 1e-6
    model = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000).fit(x, y)
    WEIGHTS.write_text(json.dumps({"embedding_model": s.embedding_model, "threshold": round(threshold, 4),
                                   "intercept": float(model.intercept_[0]), "coef": [round(float(c), 6) for c in model.coef_[0]],
                                   "trained_on": "benchmark/verdict_guard/train.jsonl", "max_fpr": args.max_fpr}),
                       encoding="utf-8")
    clf = VerdictClassifier.load(emb)
    print(f"threshold {threshold:.3f} (cross-validated false-removal rate <= {args.max_fpr:.0%} on train); "
          f"weights: {WEIGHTS.relative_to(REPO_ROOT)}")

    texts = [r["text"] for r in test]
    pattern = [bool(VERDICT_SENTENCE.search(t)) for t in texts]
    both = [p or c for p, c in zip(pattern, clf.flags(texts))]
    report = {"threshold": round(threshold, 4), "test_by": sorted({r["by"] for r in test if r["kind"] == "generated"}),
              "patterns": rates(test, pattern), "patterns_and_classifier": rates(test, both)}

    # the worst-case insertion test, with and without the classifier
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_guard_worst_case import ATTACK_SENTENCES

    verdicts = ATTACK_SENTENCES["verdict"]
    report["worst_case_verdict_phrasings"] = {"patterns": f"{sum(bool(VERDICT_SENTENCE.search(v)) for v in verdicts)}/{len(verdicts)}",
                                              "patterns_and_classifier": f"{sum(a or b for a, b in zip((bool(VERDICT_SENTENCE.search(v)) for v in verdicts), clf.flags(verdicts)))}/{len(verdicts)}"}
    (REPO_ROOT / "data" / "reports").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "data" / "reports" / "verdict_guard.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    missed = [t for t, r, f in zip(texts, test, both) if r["verdict"] and not f]
    removed = [t for t, r, f in zip(texts, test, both) if not r["verdict"] and f]
    print("\nverdicts still missed (first 8):", *missed[:8], sep="\n  ")
    print("\nnon-verdict sentences removed (first 8):", *removed[:8], sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
