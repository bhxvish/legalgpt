"""Run neuro-symbolic verification on real annotated cases and write a report to sanity-check.

The case description is the case's own sentences labelled "Facts" in the annotation store
(the court's narration — not its ruling, so the check is not handed the answer).

Usage (from the repo root):
    python backend/scripts/verify_case.py case20:304A case01:279 case31:323
    python backend/scripts/verify_case.py case20:304A --out data/reports/verification_examples.md
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation.collector import JudgmentCollector  # noqa: E402
from app.annotation.store import AnnotationStore  # noqa: E402
from app.config import REPO_ROOT, get_settings  # noqa: E402
from app.core.llm_client import GroqClient  # noqa: E402
from app.verification.fact_extractor import FactExtractor  # noqa: E402
from app.verification.prolog_engine import PrologEngine  # noqa: E402
from app.verification.verification_service import DISCLAIMER, VerificationService  # noqa: E402

ICON = {"satisfied": "✅", "violated": "❌", "missing": "❔"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help="case_id:section, e.g. case20:304A")
    ap.add_argument("--max-chars", type=int, default=6000, help="truncate long fact narratives")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "reports" / "verification_examples.md")
    args = ap.parse_args()

    s = get_settings()
    store, collector = AnnotationStore(s.annotation_store_dir), JudgmentCollector(s.raw_judgments_dir)
    llm = GroqClient(s.groq_api_key, s.groq_model, max_tokens=4096, temperature=0.0)
    service = VerificationService(FactExtractor(llm), PrologEngine(s.prolog_rules_dir))

    md = ["# Verification examples (for a team member to sanity-check)", "",
          f"Generated {datetime.now():%Y-%m-%d %H:%M} with {llm.model_id}. {DISCLAIMER}", ""]
    for target in args.targets:
        case_id, section = target.split(":")
        facts_text = " ".join(x.text for x in store.load_case(case_id) if x.label == "Facts")[: args.max_chars]
        title = collector.load(case_id).title or case_id
        result = service.verify(facts_text, section)
        print(f"\n=== {case_id} ({title}) — cited s.{result.cited_section}: {result.status}")
        for e in result.elements:
            print(f"  {e.status:9} {e.predicate:28} found={e.found:11} | {e.evidence[:90]}")
        print("  other sections:", ", ".join(f"{a.section}={a.status}" for a in result.alternatives))
        for w in result.warnings:
            print("  warning:", w)

        md += [f"## {title} ({case_id}) — cited section {result.cited_section}: **{result.status}**", "",
               "<details><summary>Case description given to the extractor (the case's Facts sentences)</summary>", "",
               facts_text, "", "</details>", "",
               "| | element | found | evidence quote |", "|---|---|---|---|"]
        md += [f"| {ICON[e.status]} | {e.description} | {e.found} | {e.evidence.replace('|', '/') or '—'} |" for e in result.elements]
        md += ["", "Other encoded sections: " + ", ".join(f"s.{a.section} {a.status.lower()}" for a in result.alternatives), ""]
        if result.warnings:
            md += ["Warnings:", *[f"- {w}" for w in result.warnings], ""]
        md += ["**Sanity check:** does the verdict match your own reading of the facts? Note disagreements here.", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
