"""How faithful are the classifier's explanations? ERASER-style deletion tests.

On held-out test-case sentences (corpus split "test", never trained on), for the role the model
predicts, rank the sentence's words by
  ig         Integrated Gradients (the Review tab's explanation)
  occlusion  leave-one-word-out probability drop (a LIME-like perturbation method)
  random     a random order (baseline; mean of 5 orders)
and measure (DeYoung et al., 2020):
  comprehensiveness@k  p(role | sentence) - p(role | sentence without its top-k% words)   higher = better
  sufficiency@k        p(role | sentence) - p(role | only its top-k% words)              lower  = better
for k = 10%, 20%, 50% (at least one word). The context segment is kept as it is: these test the
sentence words the Review tab highlights. Also reports how often IG and occlusion agree on the
single most important word, and their Spearman rank correlation.

Usage (GPU environment recommended):
    .venv-gpu\\Scripts\\python backend/scripts/eval_explanations.py --n 100
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation.store import AnnotationStore  # noqa: E402
from app.config import REPO_ROOT, get_settings  # noqa: E402
from app.labeling_assistant.classifier import RRLClassifier  # noqa: E402
from app.labeling_assistant.dataset import RRLDataset  # noqa: E402
from app.labeling_assistant.explain import ClassifierExplainer  # noqa: E402
from app.labeling_assistant.intake import latest_checkpoint  # noqa: E402

KS = (0.1, 0.2, 0.5)


def keep_text(text: str, words, keep: set[int]) -> str:
    return " ".join(text[w.start:w.end] for i, w in enumerate(words) if i in keep) or "."


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 and np.std(ra) and np.std(rb) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--steps", type=int, default=128)
    args = ap.parse_args()
    s = get_settings()
    store = AnnotationStore(s.annotation_store_dir)
    version = sorted(p.name for p in (s.annotation_store_dir / "versions").iterdir() if p.is_dir())[-1]
    manifest, sentences = store.load_version(version, verify=True)
    test_cases = set(manifest.splits["test"]) if hasattr(manifest, "splits") else set(manifest["splits"]["test"])
    examples = RRLDataset.examples_from([x for x in sentences if x.case_id in test_cases])
    by_case = defaultdict(list)
    for e in examples:
        by_case[e.case_id].append(e)
    rng = random.Random(args.seed)
    pool = [e for e in examples if 4 <= len(e.text.split()) <= 60]
    chosen = rng.sample(pool, min(args.n, len(pool)))

    ckpt = latest_checkpoint(s.rrl_model_dir)
    ex = ClassifierExplainer(RRLClassifier.load(ckpt))
    comp, suff = defaultdict(list), defaultdict(list)
    top1_agree, rho, ctx_shares, gaps = [], [], [], []
    for n, e in enumerate(chosen, 1):
        ctx = ex._context(e, by_case[e.case_id])
        probs = ex.probabilities(e.text, ctx)
        target = int(np.argmax(probs))
        p0 = float(probs[target])
        ig = ex.integrated_gradients(e.text, ctx, target, steps=args.steps)
        words, share = ex.word_scores(e.text, ig["enc"], ig["token_scores"])
        if len(words) < 2:
            continue
        ctx_shares.append(share)
        gaps.append(ig["completeness_gap"])
        occ = ex.occlusion(e.text, ctx, target, words)
        orders = {"ig": list(np.argsort([-w.score for w in words])), "occlusion": list(np.argsort([-o for o in occ]))}
        top1_agree.append(int(orders["ig"][0] == orders["occlusion"][0]))
        rho.append(spearman([w.score for w in words], occ))
        rand_orders = [rng.sample(range(len(words)), len(words)) for _ in range(5)]
        for k in KS:
            m = max(1, round(k * len(words)))
            for name, order in list(orders.items()) + [("random", None)]:
                runs = rand_orders if order is None else [order]
                c_vals, s_vals = [], []
                for o in runs:
                    top = set(o[:m])
                    without = keep_text(e.text, words, set(range(len(words))) - top)
                    only = keep_text(e.text, words, top)
                    c_vals.append(p0 - float(ex.probabilities(without, ctx)[target]))
                    s_vals.append(p0 - float(ex.probabilities(only, ctx)[target]))
                comp[(name, k)].append(mean(c_vals))
                suff[(name, k)].append(mean(s_vals))
        if n % 10 == 0:
            print(f"{n}/{len(chosen)}", flush=True)

    lines = ["# How faithful are the classifier's explanations?", "",
             f"{len(ctx_shares)} sentences from the held-out test cases of corpus {version} ({', '.join(sorted(test_cases))}), "
             f"classifier `{ckpt.name}`, explaining the role it predicts. ERASER deletion metrics (DeYoung et al., 2020); "
             "the context segment is unchanged.", "",
             "| method | comprehensiveness@10% | @20% | @50% | sufficiency@10% | @20% | @50% |", "|---|---|---|---|---|---|---|"]
    for name in ("ig", "occlusion", "random"):
        lines.append(f"| {name} | " + " | ".join(f"{mean(comp[(name, k)]):.3f}" for k in KS) + " | "
                     + " | ".join(f"{mean(suff[(name, k)]):.3f}" for k in KS) + " |")
    valid_rho = [r for r in rho if r == r]
    lines += ["", "Comprehensiveness: higher is better (removing the words the method ranks highest should hurt the "
              "prediction most). Sufficiency: lower is better (keeping only those words should preserve it).", "",
              f"- IG and occlusion pick the same most important word in **{100 * mean(top1_agree):.0f}%** of sentences; "
              f"mean Spearman rank correlation of their word rankings **{mean(valid_rho):.2f}**.",
              f"- Mean share of attribution on the context segment (position + neighbouring sentences): "
              f"**{100 * mean(ctx_shares):.0f}%**.",
              f"- Mean IG completeness gap at {args.steps} steps: {mean(gaps):.4f} logits.",
              "", "Reproduce: `.venv-gpu\\Scripts\\python backend/scripts/eval_explanations.py --n 100`"]
    out = REPO_ROOT / "docs" / "reports" / "classifier_explanations.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPO_ROOT / "data" / "reports").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "data" / "reports" / "classifier_explanations.json").write_text(json.dumps(
        {"comprehensiveness": {f"{a}@{b}": mean(v) for (a, b), v in comp.items()},
         "sufficiency": {f"{a}@{b}": mean(v) for (a, b), v in suff.items()},
         "top1_agree": mean(top1_agree), "spearman": mean(valid_rho), "context_share": mean(ctx_shares)}, indent=1))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
