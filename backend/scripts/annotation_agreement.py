"""Cohen's kappa between two annotators on the cases they both labelled.

Usage (from the repo root):
    python backend/scripts/annotation_agreement.py alice bob
    python backend/scripts/annotation_agreement.py alice bob --cases state-v-kumar another-case
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation.agreement import AgreementCalculator  # noqa: E402
from app.annotation.store import AnnotationStore  # noqa: E402
from app.config import get_settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotator_a")
    ap.add_argument("annotator_b")
    ap.add_argument("--cases", nargs="*", help="limit to these case ids (default: all shared cases)")
    args = ap.parse_args()

    store = AnnotationStore(get_settings().annotation_store_dir)
    cases = args.cases or [c for c in store.case_ids() if {args.annotator_a, args.annotator_b} <= set(store.annotators(c))]
    calc = AgreementCalculator()
    a, b = calc.paired_labels(store, cases, args.annotator_a, args.annotator_b)
    if not a:
        print(f"{args.annotator_a} and {args.annotator_b} have no sentences labelled in common")
        return 1
    kappa = calc.cohen_kappa(a, b)
    cm = calc.confusion_matrix(a, b)
    print(f"cases: {cases}")
    print(f"sentences labelled by both: {len(a)}; raw agreement {sum(x == y for x, y in zip(a, b)) / len(a):.1%}")
    print(f"Cohen's kappa: {kappa:.3f}\n")
    width = max(len(label) for label in calc.labels) + 2
    print(f"{args.annotator_a} \\ {args.annotator_b}".ljust(width) + "".join(label[:9].rjust(10) for label in calc.labels))
    for label, row in zip(calc.labels, cm):
        print(label.ljust(width) + "".join(str(v).rjust(10) for v in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
