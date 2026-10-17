"""Round 2 of the sentence-role experiments: longer training, public-corpus transfer, ensembles.

Every fine-tuned run (train_rrl_classifier.py --save-predictions) and every sequence-model run
(rrl_sequence_model.py) saved its held-out probabilities on the same 5 folds, so any fine-tuned run
can be averaged with any sequence run (fixed 50/50 weights, nothing fitted on test folds). This
reports every single model and every such ensemble, then tests the chosen final system against the
deployed model and against round 1's best, with the same tests as rrl_ablation_report.py.

The final system is fixed in advance as "everything": the transferred fine-tuned model (pub-full)
averaged with the BiLSTM on transferred vectors pretrained on public judgments (seq-pubenc). The
full grid is reported so the choice can be checked, not to pick the best cell after the fact.

Usage:  python backend/scripts/rrl_round2_report.py [--version v0.3] [--json-out FILE]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rrl_ablation_report import LABELS, f1s, latest, mcnemar_exact  # noqa: E402

FINE_TUNED = {"abl-full": "deployed (InLegalBERT, context, 6 ep, 256 tok)", "long": "longer training (10 ep, 384 tok)",
              "pub-full": "public transfer", "pub-long": "public transfer + longer"}
SEQUENCE = {"2026": "BiLSTM (round 1)", "seq-pubbilstm": "BiLSTM pretrained on public",
            "seq-pubenc": "BiLSTM pretrained on public + transferred vectors"}


def probs_of(path: Path) -> dict[str, tuple[str, np.ndarray]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {p["sentence_id"]: (p["gold"], np.array([p["probabilities"][lab] for lab in LABELS])) for p in d["predictions"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v0.3")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    v = args.version
    ft = {name: probs_of(latest(f"{v}-cv5-{tag}-*.json")) for tag, name in FINE_TUNED.items()}
    seq = {name: probs_of(latest(f"{v}-cv5-sequence-{tag}*.json")) for tag, name in SEQUENCE.items()}
    case_of = {p["sentence_id"]: p["case_id"] for p in json.loads(latest(f"{v}-cv5-abl-full-*.json").read_text(encoding="utf-8"))["predictions"]}
    ids = sorted(set.intersection(*(set(x) for x in [*ft.values(), *seq.values()])))
    gold = np.array([LABELS.index(ft[FINE_TUNED["abl-full"]][s][0]) for s in ids])
    P = {name: np.stack([d[s][1] for s in ids]) for name, d in {**ft, **seq}.items()}
    pred = {name: p.argmax(1) for name, p in P.items()}
    for a in ft:
        for b in seq:
            pred[f"{a} + {b}"] = ((P[a] + P[b]) / 2).argmax(1)

    cases = np.array([case_of[s] for s in ids])
    uniq = np.unique(cases)
    by_case = {c: np.where(cases == c)[0] for c in uniq}
    rng = np.random.default_rng(13)
    boots = [np.concatenate([by_case[c] for c in rng.choice(uniq, len(uniq))]) for _ in range(args.boot)]
    macro = lambda name, i=slice(None): f1s(gold[i], pred[name][i]).mean()  # noqa: E731

    out = {"n_sentences": len(ids), "n_judgments": len(uniq), "single": [], "ensembles": [], "comparisons": []}
    print(f"{len(ids)} held-out sentences, {len(uniq)} judgments\n\nSingle models (macro-F1 / accuracy):")
    for name in [*ft, *seq]:
        acc = float(np.mean(pred[name] == gold))
        print(f"  {name:55} {macro(name):.3f}  {acc:.3f}")
        out["single"].append({"model": name, "macro_f1": float(macro(name)), "accuracy": acc,
                              "per_class_f1": dict(zip(LABELS, map(float, f1s(gold, pred[name]))))})
    print("\nEnsembles, fine-tuned (rows) x sequence (columns), macro-F1:")
    print(" " * 48 + "".join(f"{c[:22]:>24}" for c in seq))
    for a in ft:
        print(f"  {a:46}" + "".join(f"{macro(f'{a} + {b}'):24.3f}" for b in seq))
        for b in seq:
            n = f"{a} + {b}"
            out["ensembles"].append({"fine_tuned": a, "sequence": b, "macro_f1": float(macro(n)),
                                     "accuracy": float(np.mean(pred[n] == gold)),
                                     "per_class_f1": dict(zip(LABELS, map(float, f1s(gold, pred[n]))))})

    deployed = FINE_TUNED["abl-full"]
    round1 = f"{deployed} + {SEQUENCE['2026']}"
    final = f"{FINE_TUNED['pub-full']} + {SEQUENCE['seq-pubenc']}"
    print(f"\nFinal system (fixed in advance): {final}")
    print("  per class: " + ", ".join(f"{lab} {x:.3f}" for lab, x in zip(LABELS, f1s(gold, pred[final]))))
    for a, b in [(deployed, FINE_TUNED["pub-full"]), (SEQUENCE["2026"], SEQUENCE["seq-pubenc"]),
                 (deployed, final), (round1, final)]:
        ca, cb = pred[a] == gold, pred[b] == gold
        n01, n10, p = mcnemar_exact(ca, cb)
        diffs = np.array([macro(b, i) - macro(a, i) for i in boots])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        d = macro(b) - macro(a)
        print(f"  {b}\n    vs {a}: macro-F1 {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}); {n10} fixed, {n01} broken, McNemar p = {p:.2g}")
        out["comparisons"].append({"from": a, "to": b, "macro_f1_diff": float(d), "ci95": [float(lo), float(hi)],
                                   "fixed": n10, "broken": n01, "mcnemar_p": p})
    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
