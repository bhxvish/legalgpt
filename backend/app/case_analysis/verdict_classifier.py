"""A small classifier that backs up the verdict patterns of the case-analysis output guard.

The patterns (analyzer.VERDICT_SENTENCE) catch the usual wordings ("the accused is guilty", "must be
convicted") but not paraphrases ("There is no doubt that the accused did it"). This is a logistic
regression on the sentence embeddings the app already computes (all-MiniLM-L6-v2), trained by
backend/scripts/eval_verdict_guard.py on benchmark/verdict_guard/train.jsonl; its weights are
verdict_clf.json next to this file. A sentence is removed when the patterns *or* the classifier
flag it. If the weights or the embedder are missing, the guard uses the patterns alone.
"""

import json
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).with_name("verdict_clf.json")


class VerdictClassifier:
    def __init__(self, embedder, coef: list[float], intercept: float, threshold: float) -> None:
        self.embedder = embedder
        self.coef = np.asarray(coef, dtype=np.float32)
        self.intercept, self.threshold = float(intercept), float(threshold)

    @classmethod
    def load(cls, embedder, path: Path = WEIGHTS) -> "VerdictClassifier | None":
        if embedder is None or not path.exists():
            return None
        w = json.loads(path.read_text(encoding="utf-8"))
        return cls(embedder, w["coef"], w["intercept"], w["threshold"])

    def probabilities(self, sentences: list[str]) -> np.ndarray:
        if not sentences:
            return np.zeros(0)
        x = np.asarray(self.embedder.embed(sentences), dtype=np.float32)
        return 1 / (1 + np.exp(-(x @ self.coef + self.intercept)))

    def flags(self, sentences: list[str]) -> list[bool]:
        return [bool(p >= self.threshold) for p in self.probabilities(sentences)]
