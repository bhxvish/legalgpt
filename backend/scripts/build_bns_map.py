"""Build the IPC -> BNS section map from the official BPRD correspondence table.

Source: "Correspondence Table and Comparison Summary of the Bharatiya Nyaya Sanhita, 2023 (BNS) to
the Indian Penal Code, 1860 (IPC)", Bureau of Police Research and Development (Ministry of Home
Affairs), prepared at CAPT Bhopal; copy at keralaprisons.gov.in (data/bns/, not committed).
The PDF's table has four columns (BNS section | subject | IPC section | summary of comparison);
rows are rebuilt from the text positions (pdfminer). Only the section correspondence and a short
subject title are kept (no comparison text): backend/app/core/bns_map.json.

Usage:  python backend/scripts/build_bns_map.py [--pdf data/bns/comparison_summary_BNS_to_IPC.pdf]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "backend" / "app" / "core" / "bns_map.json"
X_SUBJECT, X_IPC, X_SUMMARY = 100, 290, 345  # column starts, in PDF points
BNS_CELL = re.compile(r"^(\d{1,3})(\(\d+\))?(\([a-z]+\))?(?:\s|$)")


def rows_from_pdf(pdf: Path) -> list[dict]:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LTTextLine

    rows = []
    for page in extract_pages(str(pdf)):
        lines = []
        for el in page:
            if isinstance(el, LTTextContainer):
                for line in el:
                    if isinstance(line, LTTextLine) and line.get_text().strip():
                        lines.append((page.height - line.y1, line.x0, line.get_text().strip()))
        starts = sorted((y, t) for y, x, t in lines if x < X_SUBJECT and BNS_CELL.match(t))
        for i, (y, cell) in enumerate(starts):
            end = starts[i + 1][0] - 3 if i + 1 < len(starts) else page.height
            band = [(yy, x, t) for yy, x, t in lines if y - 3 <= yy < end]
            col = lambda lo, hi: " ".join(t for _, x, t in sorted(band) if lo <= x < hi)  # noqa: E731
            rows.append({"bns": cell.split()[0], "subject": col(X_SUBJECT, X_IPC), "ipc": col(X_IPC, X_SUMMARY),
                         "summary": col(X_SUMMARY, 10_000)})
    return rows


def ipc_numbers(cell: str) -> list[str]:
    cell = re.sub(r"\(\d+\)|\([a-z]+\)", "", cell)  # 376(2) -> 376
    return [n.upper() for n in re.findall(r"\b(\d{1,3}[A-Za-z]{0,2})\b", cell) if not n.isdigit() or int(n) <= 511]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", type=Path, default=REPO_ROOT / "data" / "bns" / "comparison_summary_BNS_to_IPC.pdf")
    args = ap.parse_args()
    rows = rows_from_pdf(args.pdf)
    ipc_to_bns: dict[str, list[dict]] = defaultdict(list)
    new_in_bns = []
    for r in rows:
        nums = ipc_numbers(r["ipc"])
        if re.match(r"\s*new\b", r["ipc"], re.I) or not nums:
            new_in_bns.append({"bns": r["bns"], "subject": r["subject"][:120]})
            continue
        for n in dict.fromkeys(nums):
            if all(m["bns"] != r["bns"] for m in ipc_to_bns[n]):  # one row can list a section twice
                ipc_to_bns[n].append({"bns": r["bns"], "subject": r["subject"][:120]})
    out = {
        "source": ("Correspondence Table and Comparison Summary of the Bharatiya Nyaya Sanhita, 2023 (BNS) to the Indian "
                   "Penal Code, 1860 (IPC), Bureau of Police Research and Development (Ministry of Home Affairs), "
                   "CAPT Bhopal"),
        "note": "BNS in force from 1 July 2024. Check the Sanhita's text where it matters; numbering changed throughout.",
        "ipc_to_bns": dict(sorted(ipc_to_bns.items(), key=lambda kv: (int(re.match(r"\d+", kv[0]).group()), kv[0]))),
        "new_in_bns": new_in_bns,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} rows -> {len(out['ipc_to_bns'])} IPC sections mapped, {len(new_in_bns)} BNS provisions new; wrote {OUT}")
    for sec in ("34", "120B", "302", "304", "304A", "304B", "307", "376", "379", "406", "420", "498A", "124A", "377", "497"):
        print(f"  IPC {sec:5} -> {[m['bns'] for m in out['ipc_to_bns'].get(sec, [])] or 'no BNS entry'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
