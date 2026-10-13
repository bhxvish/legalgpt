"""Agreement of the fact check with a labelled test set (docs/eval/verification_cases.json).

Each case is a fact pattern, the section to check and the verdict a careful reader expects
(CONSISTENT / INCONSISTENT / INSUFFICIENT). Runs the real VerificationService (hosted model
extracts the facts twice, Prolog checks them) and reports the agreement rate, a confusion table
and, for every miss, which elements came out differently. About 8k Groq tokens per case.

Usage (from the repo root):
    python backend/scripts/eval_verification.py
    python backend/scripts/eval_verification.py --cases docs/eval/verification_cases.json --only v01 v02
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import REPO_ROOT, get_settings  # noqa: E402
from app.main import build_verification_service  # noqa: E402

VERDICTS = ("CONSISTENT", "INCONSISTENT", "INSUFFICIENT")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, default=REPO_ROOT / "docs" / "eval" / "verification_cases.json")
    ap.add_argument("--only", nargs="*", help="case ids to run")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "reports" / "verification_eval")
    args = ap.parse_args()

    service = build_verification_service(get_settings())
    if service is None:
        print("verification is unavailable (SWI-Prolog / pyswip missing)")
        return 1
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]
    rows = []
    for c in cases:
        t = time.perf_counter()
        try:
            r = service.verify(c["text"], c["section"])
            got, elements = r.status, {e.predicate: e.status for e in r.elements}
            facts = {k: v["value"] for k, v in r.facts.items()}
        except Exception as e:  # rate limit or extraction failure: record, keep going
            got, elements, facts = f"ERROR: {type(e).__name__}: {e}"[:200], {}, {}
        rows.append({**c, "got": got, "elements": elements, "facts": facts, "seconds": round(time.perf_counter() - t, 1)})
        print(f"{c['id']} s.{c['section']:5} expected {c['expected']:12} got {got} ({rows[-1]['seconds']}s)")

    scored = [r for r in rows if r["got"] in VERDICTS]
    agree = sum(r["got"] == r["expected"] for r in scored)
    lines = ["# Fact check against a labelled test set", "",
             f"{len(rows)} fact patterns from `{args.cases.relative_to(REPO_ROOT).as_posix()}` "
             f"({dict(Counter(r['expected'] for r in rows))}); fact extraction by "
             f"{get_settings().groq_model}, two runs, checked in Prolog.", "",
             f"**Agreement: {agree}/{len(scored)} = {agree / len(scored):.0%}**"
             + (f" ({len(rows) - len(scored)} case(s) failed to run)" if len(scored) < len(rows) else ""), "",
             "| expected \\ got | " + " | ".join(VERDICTS) + " |", "|---|---|---|---|"]
    for exp in VERDICTS:
        lines.append(f"| {exp} | " + " | ".join(str(sum(r["expected"] == exp and r["got"] == g for r in scored))
                                                for g in VERDICTS) + " |")
    lines += ["", "| case | section | expected | got | element results |", "|---|---|---|---|---|"]
    for r in rows:
        els = ", ".join(f"{k}: {v}" for k, v in r["elements"].items())
        mark = "" if r["got"] == r["expected"] else " ✗"
        lines.append(f"| {r['id']} | {r['section']} | {r['expected']} | {r['got']}{mark} | {els} |")
    misses = [r for r in scored if r["got"] != r["expected"]]
    if misses:
        lines += ["", "## Disagreements", ""]
        lines += [f"- **{r['id']}** (expected {r['expected']}: {r['why']}); got {r['got']}. Elements: "
                  + ", ".join(f"{k}={v}" for k, v in r["elements"].items()) for r in misses]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nagreement {agree}/{len(scored)}; wrote {args.out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
