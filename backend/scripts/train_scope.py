"""Train the scope check (app/core/scope.py) on app/core/scope_examples.json.

Reports 5-fold cross-validated accuracy on the training examples, then fits on all of them and
writes app/core/scope_model.json. Re-run after changing the examples or the embedding model.

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
L2 = 1.0


def main() -> int:
    s = get_settings()
    data = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    texts = data["in_scope"] + data["out_of_scope"]
    y = np.array([1.0] * len(data["in_scope"]) + [0.0] * len(data["out_of_scope"]))
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
    model = ScopeCheck(w, b, 0.5, s.embedding_model)
    DEFAULT_MODEL.write_text(json.dumps(model.to_dict()), encoding="utf-8")
    print(f"wrote {DEFAULT_MODEL.name} ({len(w)} weights)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
