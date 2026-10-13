"""ScopeCheck: is a question about an offence under the IPC?

Everyday fact patterns ("someone slapped me at the bus stop") are written in a different style from
the statute, so their similarity to the right section is often below the retrieval floor (0.55),
which was calibrated on short legal questions - and a plain lower floor would also let civil
questions ("my landlord won't return my deposit") through, because they look just as similar.

So below the floor the retriever asks a second question: is this about a crime? A logistic
regression on the question's embedding (trained by scripts/train_scope.py on
scope_examples.json) answers it. The weights are a small JSON file, so the check is cheap,
deterministic and published - a verifier can recompute it from the (public) query embedding.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path(__file__).with_name("scope_model.json")


@dataclass
class ScopeCheck:
    weights: np.ndarray
    bias: float
    threshold: float = 0.5
    embedding_model: str = ""

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL) -> "ScopeCheck | None":
        if not Path(path).exists():
            return None
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.asarray(d["weights"], dtype=np.float64), float(d["bias"]), float(d["threshold"]),
                   d.get("embedding_model", ""))

    def probability(self, embedding) -> float:
        z = float(np.dot(self.weights, np.asarray(embedding, dtype=np.float64)) + self.bias)
        return float(1.0 / (1.0 + np.exp(-z)))

    def in_scope(self, embedding) -> bool:
        return self.probability(embedding) >= self.threshold

    def to_dict(self) -> dict:
        return {"weights": [round(float(w), 6) for w in self.weights], "bias": round(self.bias, 6),
                "threshold": self.threshold, "embedding_model": self.embedding_model}


def train(X: np.ndarray, y: np.ndarray, l2: float = 1.0, steps: int = 3000, lr: float = 0.5) -> tuple[np.ndarray, float]:
    """L2-regularised logistic regression by gradient descent (numpy only; tiny data)."""
    w, b = np.zeros(X.shape[1]), 0.0
    n = len(y)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / n + l2 * w / n)
        b -= lr * float(np.mean(p - y))
    return w, b
