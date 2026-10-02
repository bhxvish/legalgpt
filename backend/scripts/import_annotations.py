"""Import a team annotation spreadsheet (.xlsx) into the AnnotationStore.

Usage (from the repo root):
    python backend/scripts/import_annotations.py data/inbox/legaltech_dataset.xlsx --dry-run
    python backend/scripts/import_annotations.py data/inbox/legaltech_dataset.xlsx --annotator team-sheet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation.collector import JudgmentCollector  # noqa: E402
from app.annotation.importer import import_annotations  # noqa: E402
from app.annotation.store import AnnotationStore  # noqa: E402
from app.config import get_settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xlsx", type=Path)
    ap.add_argument("--annotator", default="team-sheet", help="name recorded for rows whose annotator cell is empty")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    s = get_settings()
    report = import_annotations(
        args.xlsx, JudgmentCollector(s.raw_judgments_dir), AnnotationStore(s.annotation_store_dir),
        default_annotator=args.annotator, dry_run=args.dry_run,
    )
    for c in report.cases:
        extra = f" — {c.detail}" if c.detail else ""
        print(f"{c.status.upper():9} {c.case_id:8} {c.sentences:4} sentences  {c.title[:55]}{extra}")
    for p in report.problems:
        print(f"NOTE      {p}")
    total = sum(c.sentences for c in report.imported)
    verb = "would import" if args.dry_run else "imported"
    print(f"\n{verb} {len(report.imported)}/{len(report.cases)} cases, {total} labelled sentences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
