"""Batch-ingest a folder of judgment text files into data/raw_judgments/ via JudgmentCollector.

Each .txt file is normalized, screened (duplicate / non-English / non-criminal / too short) and,
if eligible, stored with its metadata. The source folder is never modified.

Usage (from the repo root):
    python backend/scripts/collect_judgments.py data/inbox
    python backend/scripts/collect_judgments.py data/inbox --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation.collector import JudgmentCollector  # noqa: E402
from app.config import get_settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="screen and report only; store nothing")
    args = ap.parse_args()

    files = sorted(p for p in args.folder.glob("*.txt") if p.is_file())
    if not files:
        print(f"no .txt files in {args.folder}")
        return 1
    collector = JudgmentCollector(get_settings().raw_judgments_dir)
    accepted = 0
    for path in files:
        record = collector.ingest_file(path)
        if args.dry_run:
            ok, reason = collector.is_eligible(record)
        else:
            record = collector.collect(record)
            ok, reason = record.selection_status == "eligible", record.rejection_reason or "eligible"
        accepted += ok
        print(f"{'ACCEPT' if ok else 'REJECT'}  {record.case_id}")
        print(f"        court={record.court or '?'} year={record.decision_year or '?'} "
              f"chars={len(record.raw_text)} IPC sections={record.sections_cited or '-'}")
        if not ok:
            print(f"        reason: {reason}")
    action = "would be collected" if args.dry_run else f"collected into {collector.store_dir}"
    print(f"\n{accepted}/{len(files)} {action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
