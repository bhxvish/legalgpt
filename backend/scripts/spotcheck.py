"""Human spot-check of the AI-judged results: one person labels a sample blind, then agreement is scored.

  make    writes data/reports/spotcheck/spotcheck.xlsx: 50 items in random order, AI labels hidden
            - 38 citations from the faithfulness evaluation (19 citing a judgment, 19 citing an IPC
              section): does the SOURCE support what the SENTENCE attributes to it?
              SUPPORTED / PARTIAL / UNSUPPORTED (same definitions the two AI judges had)
            - 12 Hindi questions (machine-translated): does the Hindi say the same as the English?
              SAME / MINOR / WRONG
          A separate key file (spotcheck_key.json) holds the AI labels; do not open it before scoring.
  score   reads your labels from the sheet and reports, against each AI judge: agreement, Cohen's
          kappa (3-way and supported-vs-not), and the items you disagree on; for Hindi, the share
          you rated SAME. Writes data/reports/spotcheck/summary.json.

Usage:  python backend/scripts/spotcheck.py make
        python backend/scripts/spotcheck.py score
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "data" / "reports" / "spotcheck"
SHEET, KEY = OUT / "spotcheck.xlsx", OUT / "spotcheck_key.json"
CITE = ["SUPPORTED", "PARTIAL", "UNSUPPORTED"]
HINDI = ["SAME", "MINOR", "WRONG"]

GUIDE = [
    "How to fill this in (about 2 hours; do it in one or two sittings):",
    "",
    "1. Work down the 'items' sheet. For each row, read the SENTENCE, then the SOURCE, and pick your label",
    "   from the dropdown in the YOUR LABEL column. Optionally add a short note.",
    "2. Citation rows (kind = judgment / statute): judge ONLY whether the SOURCE supports what the SENTENCE",
    "   attributes to it. Ignore whether you think the law is right; ignore claims tied to other [numbers].",
    "     SUPPORTED   - the source states, or directly and obviously implies, everything the sentence says.",
    "     PARTIAL     - the source supports part of it, or the sentence overstates / generalises it.",
    "     UNSUPPORTED - the source does not say it, is about something else, or contradicts it.",
    "3. Hindi rows (kind = hindi): does the Hindi question ask the same thing as the English one?",
    "     SAME  - same facts and meaning (wording may differ).",
    "     MINOR - a small slip that would not change which offence applies.",
    "     WRONG - a fact or meaning changed (who did what, the act, the harm).",
    "4. Do NOT open spotcheck_key.json (the AI labels) until you have finished every row.",
    "5. Save the file, then run:  python backend/scripts/spotcheck.py score",
]


def make(n_hindi: int = 12, seed: int = 11) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    rng = random.Random(seed)
    claims_path = REPO_ROOT / "data" / "reports" / "faithfulness" / "claims.jsonl"
    claims = [json.loads(ln) for ln in claims_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    hindi = [json.loads(ln) for ln in (REPO_ROOT / "benchmark" / "hindi" / "questions_hi.jsonl").read_text(encoding="utf-8").splitlines()]
    hindi = rng.sample([h for h in hindi if h.get("question_hi")], n_hindi)
    items, key = [], {}
    for c in claims:
        src = (c["title"] + " - judgment passage" if c["kind"] == "judgment" else c["citation"]) + ":\n" + c["source"]
        items.append({"kind": c["kind"], "a": c["sentence"], "b": src})
        key_labels = {judge: v["label"] for judge, v in c["judges"].items()}
        items[-1]["key"] = {"type": "citation", "ai": key_labels}
    for h in hindi:
        items.append({"kind": "hindi", "a": h["question_en"], "b": h["question_hi"],
                      "key": {"type": "hindi", "translated_by": h.get("translated_by", "")}})
    rng.shuffle(items)

    wb = Workbook()
    guide = wb.active
    guide.title = "how to"
    for i, line in enumerate(GUIDE, 1):
        guide.cell(i, 1, line).font = Font(bold=i == 1, size=12 if i == 1 else 11)
    guide.column_dimensions["A"].width = 110
    ws = wb.create_sheet("items")
    head = ["id", "kind", "SENTENCE (or English question)", "SOURCE (or Hindi question)", "YOUR LABEL", "note"]
    ws.append(head)
    for col, w in zip("ABCDEF", (8, 10, 60, 80, 16, 30)):
        ws.column_dimensions[col].width = w
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F2937")
    dv_c = DataValidation(type="list", formula1='"' + ",".join(CITE) + '"', allow_blank=True)
    dv_h = DataValidation(type="list", formula1='"' + ",".join(HINDI) + '"', allow_blank=True)
    ws.add_data_validation(dv_c)
    ws.add_data_validation(dv_h)
    yellow = PatternFill("solid", fgColor="FFF7CC")
    for i, it in enumerate(items, 1):
        iid = f"S{i:02d}"
        ws.append([iid, it["kind"], it["a"], it["b"], "", ""])
        r = ws.max_row
        for col in "CD":
            ws[f"{col}{r}"].alignment = Alignment(wrap_text=True, vertical="top")
        ws[f"E{r}"].fill = yellow
        (dv_h if it["kind"] == "hindi" else dv_c).add(f"E{r}")
        key[iid] = it["key"]
    ws.freeze_panes = "C2"
    OUT.mkdir(parents=True, exist_ok=True)
    wb.save(SHEET)
    KEY.write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {SHEET} ({len(items)} items: {sum(i['kind'] != 'hindi' for i in items)} citations, "
          f"{sum(i['kind'] == 'hindi' for i in items)} Hindi) and the hidden key {KEY.name}")


def kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def score() -> None:
    from openpyxl import load_workbook

    ws = load_workbook(SHEET)["items"]
    key = json.loads(KEY.read_text(encoding="utf-8"))
    mine = {row[0]: (row[1], str(row[4] or "").strip().upper()) for row in ws.iter_rows(min_row=2, values_only=True) if row[0]}
    missing = [i for i, (_, lab) in mine.items() if not lab]
    if missing:
        print(f"{len(missing)} item(s) not labelled yet: {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}")
    cites = {i: lab for i, (kind, lab) in mine.items() if kind != "hindi" and lab in CITE}
    judges = sorted({j for k in key.values() if k["type"] == "citation" for j in k["ai"]})
    yes = lambda xs: ["SUPPORTED" if x == "SUPPORTED" else "NOT" for x in xs]  # noqa: E731
    out = {"citations_labelled": len(cites), "by_judge": {}}
    for j in judges:
        ids = [i for i in cites if key[i]["ai"].get(j) in CITE]
        h, a = [cites[i] for i in ids], [key[i]["ai"][j] for i in ids]
        out["by_judge"][j] = {"n": len(ids), "agreement": round(sum(x == y for x, y in zip(h, a)) / len(ids), 3) if ids else None,
                              "kappa_3way": kappa(h, a), "kappa_binary": kappa(yes(h), yes(a)),
                              "disagreements": [{"id": i, "you": cites[i], "ai": key[i]["ai"][j]} for i in ids if cites[i] != key[i]["ai"][j]]}
    both = [i for i in cites if len({key[i]["ai"].get(j) for j in judges}) == 1]
    out["you_vs_both_judges_where_they_agree"] = {
        "n": len(both), "agreement": round(sum(cites[i] == key[i]["ai"][judges[0]] for i in both) / len(both), 3) if both else None}
    out["your_labels"] = {kind: dict(Counter(lab for i, (k, lab) in mine.items() if (k == "judgment") == (kind == "judgment") and k != "hindi" and lab))
                          for kind in ("judgment", "statute")}
    hin = [lab for kind, lab in mine.values() if kind == "hindi" and lab in HINDI]
    out["hindi"] = {"n": len(hin), **dict(Counter(hin))}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"citations labelled: {len(cites)}")
    for j, v in out["by_judge"].items():
        print(f"  vs {j}: agreement {v['agreement']}, kappa {v['kappa_3way']} (3-way) / {v['kappa_binary']} (supported vs not), "
              f"{len(v['disagreements'])} disagreement(s)")
    w = out["you_vs_both_judges_where_they_agree"]
    print(f"  where both AI judges agree ({w['n']} items), you agree with them: {w['agreement']}")
    print(f"Hindi translations: {out['hindi']}")
    print(f"wrote {OUT / 'summary.json'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["make", "score"])
    args = ap.parse_args()
    make() if args.step == "make" else score()
    return 0


if __name__ == "__main__":
    sys.exit(main())
