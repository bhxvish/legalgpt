"""Train the scope check (app/core/scope.py).

Training data:
- app/core/scope_examples.json: 50 + 50 questions;
- the benchmark's practice scenarios (eval_scenario_bench.py "tune" + "check"; never the 60
  held-out benchmark scenarios) as in-scope;
- app/core/scope_civil_extra.json: everyday civil / non-IPC matters and general questions, as
  out-of-scope.

Reports 5-fold cross-validated accuracy, then fits on everything. The threshold is set on the
out-of-scope *validation* questions (eval_scenario_bench "oos-tune", not used for training): 0.05
above the highest of them, and never below 0.5. So no validation civil question reaches it, and the
query rewrite, which uses the same gate, cannot make one answerable. Writes app/core/scope_model.json.

Usage (from the repo root):
    python backend/scripts/train_scope.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.core.embeddings import EmbeddingService  # noqa: E402
from app.core.retriever import QueryExpander  # noqa: E402
from app.core.scope import DEFAULT_MODEL, ScopeCheck, train  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "app" / "core" / "scope_examples.json"
CIVIL = Path(__file__).resolve().parents[1] / "app" / "core" / "scope_civil_extra.json"
L2 = 1.0
MARGIN = 0.05


def main() -> int:
    s = get_settings()
    data = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_scenario_bench import splits

    pool = splits()
    held_out = {q["id"] for q in pool["final"]}
    practice = [q["question"] for q in pool["tune"] + pool["check"] if q["id"] not in held_out]
    civil = json.loads(CIVIL.read_text(encoding="utf-8"))["out_of_scope"]
    pos, neg = data["in_scope"] + practice, data["out_of_scope"] + civil
    texts = pos + neg
    y = np.array([1.0] * len(pos) + [0.0] * len(neg))
    emb = EmbeddingService(s.embedding_model)
    X = emb.embed([QueryExpander().expand(t) for t in texts]).astype(np.float64)

    rng = np.random.default_rng(13)
    folds = np.array_split(rng.permutation(len(y)), 5)
    correct = 0
    for k, test in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(y)), test)
        w, b = train(X[tr], y[tr], l2=L2)
        pred = (1 / (1 + np.exp(-(X[test] @ w + b)))) >= 0.5
        correct += int(np.sum(pred == y[test].astype(bool)))
    print(f"5-fold CV accuracy on {len(y)} examples: {correct / len(y):.1%}")

    w, b = train(X, y, l2=L2)
    val = [q["question"] for q in pool["oos-tune"]]
    p_val = 1 / (1 + np.exp(-(emb.embed([QueryExpander().expand(t) for t in val]).astype(np.float64) @ w + b)))
    threshold = max(0.5, float(np.ceil((p_val.max() + MARGIN) * 100) / 100))
    print(f"out-of-scope validation ({len(val)} questions): highest p {p_val.max():.2f} -> threshold {threshold:.2f}")
    model = ScopeCheck(w, b, threshold, s.embedding_model)
    DEFAULT_MODEL.write_text(json.dumps(model.to_dict()), encoding="utf-8")
    print(f"wrote {DEFAULT_MODEL.name} ({len(w)} weights)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
