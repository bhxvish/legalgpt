"""Significance of the sentence-role ablation: paired tests on the same held-out sentences.

Reads the saved cross-validation predictions (train_rrl_classifier.py --save-predictions and
rrl_sequence_model.py) and reports, per row: pooled macro-F1 and accuracy, per-class F1, and, against
the previous row and against the current model (full context):
  - exact McNemar test on per-sentence correctness (same sentences, same folds);
  - a 95% confidence interval for the macro-F1 difference from a bootstrap over judgments
    (judgments are resampled, not sentences, since sentences of one judgment are not independent).

Usage (from the repo root):
    python backend/scripts/rrl_ablation_report.py [--version v0.3] [--boot 2000]
"""

import argparse
import glob
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CV = REPO / "data" / "models" / "rrl_cv"
LABELS = ["Facts", "Law Applied", "Precedent", "Argument", "Ruling"]


def latest(pattern: str) -> Path:
    files = sorted(glob.glob(str(CV / pattern)))
    if not files:
        raise SystemExit(f"no file matches {pattern}")
    return Path(files[-1])


def load_rows(version: str) -> dict[str, dict[str, tuple[str, str, str]]]:
    """row name -> sentence_id -> (case_id, gold, predicted)."""
    rows: dict[str, dict] = {}
    for tag, name in [("sentence", "Sentence only"), ("position", "+ position only"), ("neighbours", "+ neighbours only"),
                      ("full", "+ position + neighbours")]:
        d = json.loads(latest(f"{version}-cv5-abl-{tag}-*.json").read_text(encoding="utf-8"))
        rows[name] = {p["sentence_id"]: (p["case_id"], p["gold"], p["pred"]) for p in d["predictions"]}
    seq = json.loads(latest(f"{version}-cv5-sequence-*.json").read_text(encoding="utf-8"))
    case_of = {sid: v[0] for sid, v in rows["Sentence only"].items()}
    argmax = lambda probs: max(probs, key=probs.get)  # noqa: E731
    rows["BiLSTM sequence model"] = {p["sentence_id"]: (case_of[p["sentence_id"]], p["gold"], argmax(p["probabilities"]))
                                     for p in seq["predictions"]}
    if seq["predictions"] and "ensemble" in seq["predictions"][0]:
        rows["Ensemble (+ both, + sequence)"] = {p["sentence_id"]: (case_of[p["sentence_id"]], p["gold"], argmax(p["ensemble"]))
                                                 for p in seq["predictions"]}
    ids = set.intersection(*(set(r) for r in rows.values()))
    return {k: {sid: v for sid, v in r.items() if sid in ids} for k, r in rows.items()}


def f1s(gold: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = []
    for c in range(len(LABELS)):
        tp = np.sum((pred == c) & (gold == c))
        fp = np.sum((pred == c) & (gold != c))
        fn = np.sum((pred != c) & (gold == c))
        out.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return np.array(out)


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """a, b: per-sentence correctness. Returns (only a right, only b right, two-sided exact p)."""
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = n01 + n10
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return n01, n10, p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v0.3")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    rows = load_rows(args.version)
    ids = sorted(next(iter(rows.values())))
    cases = np.array([rows["Sentence only"][s][0] for s in ids])
    gold = np.array([LABELS.index(rows["Sentence only"][s][1]) for s in ids])
    pred = {k: np.array([LABELS.index(r[s][2]) for s in ids]) for k, r in rows.items()}
    names = list(rows)
    uniq = np.unique(cases)
    idx_by_case = {c: np.where(cases == c)[0] for c in uniq}
    rng = np.random.default_rng(13)
    boots = [np.concatenate([idx_by_case[c] for c in rng.choice(uniq, len(uniq))]) for _ in range(args.boot)]

    def macro(i, name):
        return f1s(gold[i], pred[name][i]).mean()

    out = {"n_sentences": len(ids), "n_judgments": len(uniq), "rows": []}
    print(f"{len(ids)} held-out sentences from {len(uniq)} judgments (5-fold CV, each sentence predicted once)\n")
    print(f"{'model':32} {'macro-F1':>8} {'acc':>6}   " + "  ".join(f"{c[:8]:>8}" for c in LABELS))
    for name in names:
        per = f1s(gold, pred[name])
        acc = float(np.mean(pred[name] == gold))
        print(f"{name:32} {per.mean():8.3f} {acc:6.3f}   " + "  ".join(f"{x:8.3f}" for x in per))
        out["rows"].append({"model": name, "macro_f1": float(per.mean()), "accuracy": acc,
                            "per_class_f1": dict(zip(LABELS, map(float, per)))})
    print()
    comparisons = [(names[i - 1], names[i]) for i in range(1, 4)]
    full = "+ position + neighbours"
    comparisons += [(full, n) for n in names[4:]] + [("Sentence only", names[-1])]
    out["comparisons"] = []
    for a, b in comparisons:
        ca, cb = pred[a] == gold, pred[b] == gold
        n01, n10, p = mcnemar_exact(ca, cb)
        diffs = np.array([macro(i, b) - macro(i, a) for i in boots])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        d = macro(np.arange(len(ids)), b) - macro(np.arange(len(ids)), a)
        print(f"{b} vs {a}: macro-F1 {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}); "
              f"McNemar: {n10} sentences fixed, {n01} broken, p = {p:.2g}")
        out["comparisons"].append({"from": a, "to": b, "macro_f1_diff": float(d), "ci95": [float(lo), float(hi)],
                                   "fixed": n10, "broken": n01, "mcnemar_p": p})
    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
