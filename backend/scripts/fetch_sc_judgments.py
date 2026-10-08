"""Fetch Supreme Court judgments from the CC-BY-4.0 Hugging Face dataset
`labofsahil/Indian-Supreme-Court-Judgments` into data/inbox/ as plain text.

The dataset stores each year's PDFs in one large tar (hundreds of MB). This script reads only
the tar headers it needs and downloads the chosen PDFs with HTTP range requests, then extracts
and cleans the text of the Supreme Court Reports layout:

- the editorial headnote is cut (the body starts after the "JUDGMENT" line or "The Judgment of
  the Court was delivered by"), since it is the reporter's summary, not the court's text;
- page numbers and repeated running headers ("SUPREME COURT REPORTS", "[2023] 11 S.C.R.") go;
- ligature splits from PDF extraction ("ﬁ led") are re-joined;
- PDFs whose text layer is garbled (words run together) are skipped.

Usage (from the repo root):
    python backend/scripts/fetch_sc_judgments.py --year 2023 --list-criminal 40
    python backend/scripts/fetch_sc_judgments.py --year 2023 --files 2023_8_152_182_EN.pdf 2023_11_484_506_EN.pdf
"""

import argparse
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import REPO_ROOT  # noqa: E402
from app.labeling_assistant.pdf_text import clean_judgment_text, looks_garbled  # noqa: E402

DATASET = "https://huggingface.co/datasets/labofsahil/Indian-Supreme-Court-Judgments/resolve/main"
UA = {"User-Agent": "legaltech-research"}
_PARTY_STATE = re.compile(r"\bSTATE\b|\bUNION TERRITORY\b|C\.?B\.?I\b|\bN\.?C\.?T\b", re.I)
_NON_CRIMINAL = re.compile(r"TAX|REVENUE|BANK|LIMITED|LTD|INSURANCE|ELECTRICITY|MUNICIPAL|AUTHORITY|UNIVERSITY|COMMISSIONER|CORPORATION", re.I)


def fetch(url: str, start: int | None = None, length: int | None = None) -> bytes:
    headers = dict(UA)
    if start is not None:
        headers["Range"] = f"bytes={start}-{start + length - 1}"  # type: ignore[operator]
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
        return r.read()


def tar_locations(tar_url: str, wanted: set[str]) -> dict[str, tuple[int, int]]:
    """(data offset, size) of each wanted member, read header by header from the start."""
    found: dict[str, tuple[int, int]] = {}
    offset, pax_name = 0, None
    while len(found) < len(wanted):
        h = fetch(tar_url, offset, 512)
        if not h.strip(b"\0"):
            break
        size = int(h[124:136].rstrip(b"\0 ") or b"0", 8)
        data_at = offset + 512
        if h[156:157] == b"x":  # PAX header: may hold the next member's path
            for rec in fetch(tar_url, data_at, size).decode("utf-8", "replace").split("\n"):
                if " path=" in rec:
                    pax_name = rec.split(" path=", 1)[1]
        else:
            name = (pax_name or h[:100].rstrip(b"\0").decode()).rsplit("/", 1)[-1]
            if name in wanted:
                found[name] = (data_at, size)
            pax_name = None
        offset = data_at + ((size + 511) // 512) * 512
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--files", nargs="*", default=[], help="PDF names from the year's index to fetch")
    ap.add_argument("--list-criminal", type=int, default=0, metavar="N",
                    help="list likely criminal cases (State is a party) among the first N files of the tar")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "inbox")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    tar_url = f"{DATASET}/data/tar/year={args.year}/english/english.tar"
    index = json.loads(fetch(f"{DATASET}/data/tar/year={args.year}/english/english.index.json"))
    order = [f for part in index["parts"] for f in part["files"]]
    meta = {r["path"]: r for r in pq.read_table(io.BytesIO(fetch(f"{DATASET}/metadata/parquet/year={args.year}/metadata.parquet"))).to_pylist()}

    if args.list_criminal:
        for pos, name in enumerate(order[: args.list_criminal]):
            m = meta.get(name.replace("_EN.pdf", ""))
            if m and _PARTY_STATE.search(m["title"]) and not _NON_CRIMINAL.search(m["title"]):
                print(f"{pos:4} {name:28} {m['decision_date']}  {m['title'][:80]}")
        return 0

    from pdfminer.high_level import extract_text

    args.out.mkdir(parents=True, exist_ok=True)
    for name, (start, size) in tar_locations(tar_url, set(args.files)).items():
        m = meta[name.replace("_EN.pdf", "")]
        text = clean_judgment_text(extract_text(io.BytesIO(fetch(tar_url, start, size))))
        if looks_garbled(text):
            print(f"SKIP   {name}: PDF text layer is garbled (words run together)")
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", m["title"].split(" versus ")[0].split("@")[0].lower()).strip("-")[:40]
        out = args.out / f"sc-{args.year}-{slug}.txt"  # source: labofsahil/Indian-Supreme-Court-Judgments (CC-BY-4.0)
        # judgment-style header only; provenance (dataset + PDF name) is the file's source_path
        header = f"Supreme Court of India\n{m['title']}\nDecided on {m['decision_date']}\n\n"
        out.write_text(header + text + "\n", encoding="utf-8")
        print(f"WROTE  {out.name}: {len(text)} chars (PDF {size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
