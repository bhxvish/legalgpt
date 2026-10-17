"""Build the zip for the sentence-role (InLegalBERT) experiments on Colab: data/colab/legalgpt_rrl.zip.

Contents: the backend code and scripts, and one frozen annotation snapshot (default v0.3: the
annotated judgments, digest-checked when loaded). No .env, keys, models or the IPC index.
Uploading it puts the judgment texts and their labels on Google's servers.

Usage (from the repo root):
    python backend/scripts/colab/make_rrl_package.py [--version v0.3]
"""

import argparse
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
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
    ap.add_argument("--version", default="v0.3", help="annotation snapshot to include")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "colab" / "legalgpt_rrl.zip")
    args = ap.parse_args()
    snapshot = f"data/annotation_store/versions/{args.version}"
    if not (REPO / snapshot / "manifest.json").exists():
        raise SystemExit(f"no snapshot {snapshot}")
    include = ["backend/app", "backend/scripts", "backend/requirements.txt", snapshot]
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
    print(f"{args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {n} files; snapshot {args.version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
