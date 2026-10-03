"""ClassifierEvaluator: accuracy, per-class precision/recall/F1, macro-F1, confusion matrix,
and case-grouped k-fold cross-validation."""

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class Metrics:
    labels: list[str]
    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassMetrics]
    confusion: np.ndarray  # rows = true label, columns = predicted label
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": self.labels,
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "per_class": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in vars(v).items()}
                          for k, v in self.per_class.items()},
            "confusion": self.confusion.tolist(),
        }

    def format(self) -> str:
        """Human-readable report: headline numbers, per-class table, confusion matrix."""
        w = max(len(label) for label in self.labels) + 2
        lines = [
            f"n = {self.n}   accuracy = {self.accuracy:.3f}   macro-F1 = {self.macro_f1:.3f}",
            "",
            "class".ljust(w) + "precision".rjust(10) + "recall".rjust(8) + "F1".rjust(7) + "support".rjust(9),
        ]
        for label in self.labels:
            m = self.per_class[label]
            lines.append(label.ljust(w) + f"{m.precision:10.3f}{m.recall:8.3f}{m.f1:7.3f}{m.support:9d}")
        lines += ["", "confusion (rows = true, columns = predicted)", " " * w + "".join(l[:9].rjust(10) for l in self.labels)]
        for label, row in zip(self.labels, self.confusion):
            lines.append(label.ljust(w) + "".join(str(v).rjust(10) for v in row))
        return "\n".join(lines)


@dataclass
class CVReport:
    k: int
    seed: int
    folds: list[Metrics] = field(default_factory=list)
    fold_cases: list[list[str]] = field(default_factory=list)

    @property
    def macro_f1(self) -> tuple[float, float]:
        scores = [m.macro_f1 for m in self.folds]
        return float(np.mean(scores)), float(np.std(scores))

    @property
    def accuracy(self) -> tuple[float, float]:
        scores = [m.accuracy for m in self.folds]
        return float(np.mean(scores)), float(np.std(scores))

    def format(self) -> str:
        f1, f1_sd = self.macro_f1
        acc, acc_sd = self.accuracy
        rows = [f"fold {i + 1}: accuracy {m.accuracy:.3f}  macro-F1 {m.macro_f1:.3f}  (n={m.n}, {len(c)} cases)"
                for i, (m, c) in enumerate(zip(self.folds, self.fold_cases))]
        return "\n".join(rows + [f"mean over {self.k} folds: accuracy {acc:.3f} ± {acc_sd:.3f}   macro-F1 {f1:.3f} ± {f1_sd:.3f}"])


@dataclass
class ReviewLoadRow:
    bar: float  # ReviewSelector.tau_conf
    skip_share: float  # share of sentences at or above the bar (only the audit sample of these is reviewed)
    skip_accuracy: float | None  # how often those confident predictions are right
    review_share: float  # share below the bar: mandatory review


def review_load(confidences: Sequence[float], correct: Sequence[bool],
                bars: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)) -> list[ReviewLoadRow]:
    """What each confidence bar would mean on held-out sentences: how much skips mandatory review
    and how accurate the skipped labels are. This is the number that decides human workload."""
    n = len(confidences)
    rows = []
    for bar in bars:
        hi = [c for conf, c in zip(confidences, correct) if conf >= bar]
        rows.append(ReviewLoadRow(bar, len(hi) / n if n else 0.0,
                                  sum(hi) / len(hi) if hi else None, 1 - len(hi) / n if n else 0.0))
    return rows


def format_review_load(rows: Sequence[ReviewLoadRow]) -> str:
    lines = ["bar   skip review   accuracy of skipped   must review"]
    for r in rows:
        acc = f"{r.skip_accuracy:.1%}" if r.skip_accuracy is not None else "  -  "
        lines.append(f"{r.bar:.2f}  {r.skip_share:10.1%}   {acc:>19}   {r.review_share:10.1%}")
    return "\n".join(lines)


class ClassifierEvaluator:
    def evaluate(self, y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Metrics:
        if len(y_true) != len(y_pred):
            raise ValueError(f"{len(y_true)} true labels but {len(y_pred)} predictions")
        if not y_true:
            raise ValueError("nothing to evaluate")
        labels = list(labels)
        index = {label: i for i, label in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[index[t], index[p]] += 1
        per_class: dict[str, ClassMetrics] = {}
        for i, label in enumerate(labels):
            tp = int(cm[i, i])
            predicted, actual = int(cm[:, i].sum()), int(cm[i, :].sum())
            precision = tp / predicted if predicted else 0.0
            recall = tp / actual if actual else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[label] = ClassMetrics(precision, recall, f1, actual)
        # macro-F1 over classes present in the gold labels (absent classes would only add zeros)
        present = [l for l in labels if per_class[l].support > 0]
        macro_f1 = float(np.mean([per_class[l].f1 for l in present]))
        return Metrics(labels, float(np.trace(cm) / cm.sum()), macro_f1, per_class, cm, int(cm.sum()))

    def cross_validate(
        self,
        ds: Sequence[Any],
        k: int,
        seed: int,
        train_and_predict: Callable[[list[Any], list[Any]], list[str]],
        labels: Sequence[str],
        case_of: Callable[[Any], str] = lambda ex: ex.case_id,
        label_of: Callable[[Any], str] = lambda ex: ex.label,
    ) -> CVReport:
        """k-fold cross-validation grouped by case: every sentence of a judgment lands in the same
        fold, so no fold is tested on a case it was trained on. `train_and_predict(train, test)`
        trains a fresh model on `train` and returns predicted labels for `test`."""
        cases = sorted({case_of(ex) for ex in ds})
        if k < 2 or k > len(cases):
            raise ValueError(f"k must be between 2 and the number of cases ({len(cases)})")
        random.Random(seed).shuffle(cases)
        folds = [cases[i::k] for i in range(k)]
        report = CVReport(k=k, seed=seed)
        for fold_cases in folds:
            held_out = set(fold_cases)
            train = [ex for ex in ds if case_of(ex) not in held_out]
            test = [ex for ex in ds if case_of(ex) in held_out]
            predicted = train_and_predict(train, test)
            report.folds.append(self.evaluate([label_of(ex) for ex in test], predicted, labels))
            report.fold_cases.append(sorted(fold_cases))
        return report
