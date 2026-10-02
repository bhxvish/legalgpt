"""AgreementCalculator: inter-annotator agreement on a double-annotated subset."""

from collections.abc import Sequence

import numpy as np

from app.annotation.label_scheme import LabelScheme
from app.annotation.store import AnnotationStore


class AgreementCalculator:
    def __init__(self, labels: Sequence[str] = LabelScheme.LABELS) -> None:
        self.labels = list(labels)

    def cohen_kappa(self, annotator_a_labels: Sequence[str], annotator_b_labels: Sequence[str]) -> float:
        """Cohen's kappa = (p_o - p_e) / (1 - p_e) for two annotators over the same items.

        Returns 1.0 when both annotators used one and the same label throughout (p_e = 1, perfect
        agreement), where the formula itself is undefined."""
        cm = self.confusion_matrix(annotator_a_labels, annotator_b_labels)
        n = cm.sum()
        if n == 0:
            raise ValueError("no items to compare")
        p_o = np.trace(cm) / n
        p_e = float((cm.sum(axis=1) * cm.sum(axis=0)).sum()) / (n * n)
        if p_e >= 1.0:
            return 1.0
        return float((p_o - p_e) / (1.0 - p_e))

    def confusion_matrix(self, a: Sequence[str], b: Sequence[str]) -> np.ndarray:
        """Rows: annotator A's label; columns: annotator B's label; order = self.labels."""
        if len(a) != len(b):
            raise ValueError(f"annotators labelled different numbers of items ({len(a)} vs {len(b)})")
        index = {label: i for i, label in enumerate(self.labels)}
        unknown = sorted({x for x in (*a, *b) if x not in index})
        if unknown:
            raise ValueError(f"labels not in the scheme: {unknown}")
        cm = np.zeros((len(self.labels), len(self.labels)), dtype=int)
        for x, y in zip(a, b):
            cm[index[x], index[y]] += 1
        return cm

    @staticmethod
    def paired_labels(
        store: AnnotationStore, case_ids: Sequence[str], annotator_a: str, annotator_b: str
    ) -> tuple[list[str], list[str]]:
        """Labels from both annotators on the sentences they both labelled, aligned."""
        a_out: list[str] = []
        b_out: list[str] = []
        for case_id in case_ids:
            a = {s.sentence_id: s.label for s in store.load_case(case_id, annotator=annotator_a)}
            b = {s.sentence_id: s.label for s in store.load_case(case_id, annotator=annotator_b)}
            for sid in sorted(a.keys() & b.keys()):
                a_out.append(a[sid])
                b_out.append(b[sid])
        return a_out, b_out
