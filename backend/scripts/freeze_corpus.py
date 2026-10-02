"""Freeze the current annotations as an immutable, hashed corpus version.

Writes data/annotation_store/versions/<version>/{labels.jsonl,manifest.json}. Phases 3 and 4
read the manifest to record exactly which snapshot they trained on.

Usage (from the repo root):
    python backend/scripts/freeze_corpus.py v0.1 --notes "seed set, 3 cases"
    python backend/scripts/freeze_corpus.py --verify v0.1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation.store import AnnotationStore  # noqa: E402
from app.config import get_settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version")
    ap.add_argument("--notes", default="")
    ap.add_argument("--seed", type=int, default=13, help="seed for the train/val/test case split")
    ap.add_argument("--verify", action="store_true", help="re-check an existing version's digests instead")
    args = ap.parse_args()

    store = AnnotationStore(get_settings().annotation_store_dir)
    if args.verify:
        manifest, sentences = store.load_version(args.version, verify=True)
        print(f"version {args.version}: OK — {len(manifest.case_ids)} cases, {len(sentences)} sentences, sha256 {manifest.sha256}")
        return 0

    manifest = store.freeze(args.version, split_seed=args.seed, notes=args.notes)
    incomplete = [c for c, e in manifest.cases.items() if not e.complete]
    print(f"froze {args.version}: {len(manifest.case_ids)} cases, {manifest.total_sentences} labelled sentences")
    print(f"  sha256       {manifest.sha256}")
    print(f"  labels       {manifest.label_counts}")
    print(f"  splits       " + ", ".join(f"{k}={len(v)}" for k, v in manifest.splits.items()))
    if incomplete:
        print(f"  WARNING: {len(incomplete)} case(s) not fully labelled: {incomplete}")
    print(f"  manifest     {store.root / 'versions' / args.version / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
