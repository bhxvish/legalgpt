"""Build the zip that the Colab notebook trains from (data/colab/legalgpt_colab.zip).

Contents: the backend code, the IPC corpus and its search index, the benchmark, the cached teacher
rewrites and the eval questions. Not included unless --with-judgments: the judgment corpus from
the Review tab (Module 1), because uploading it puts those texts on Google's servers. Without it,
train with --no-judgments (the ~95 judgment-based examples are left out). No .env, keys or models.

Usage (from the repo root):
    python backend/scripts/colab/make_package.py [--with-judgments]
"""

import argparse
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
INCLUDE = [
    "backend/app", "backend/scripts", "backend/requirements.txt",
    "data/corpus", "data/chroma_db", "benchmark", "docs/eval/scenario_questions.json",
    "data/reports/bench/rewrites",
]
SKIP_PARTS = {"__pycache__", ".pytest_cache", "node_modules"}


def files(rel: str):
    p = REPO / rel
    if p.is_file():
        yield p
    elif p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file() and not SKIP_PARTS & set(f.parts) and f.suffix not in {".pyc", ".bak"}:
                yield f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-judgments", action="store_true", help="also include data/annotation_store/versions")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "colab" / "legalgpt_colab.zip")
    args = ap.parse_args()
    include = INCLUDE + (["data/annotation_store/versions"] if args.with_judgments else [])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in include:
            for f in files(rel):
                name = f.relative_to(REPO).as_posix()
                if name.endswith(".env") or "/.env" in name:
                    continue
                z.write(f, name)
                n += 1
    print(f"{args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {n} files; judgments {'included' if args.with_judgments else 'not included'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
