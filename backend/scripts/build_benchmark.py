"""Build the IPC benchmark in benchmark/ (data files, labelling sheets, statistics).

  statute_qa.jsonl      punishment and definition questions built from the IPC text (gold by construction)
  scenarios.jsonl       everyday fact-pattern questions with acceptable sections (silver labels)
  out_of_scope.jsonl    civil / non-IPC questions that should not be answered as IPC offences
  fact_patterns.jsonl   generated fact patterns with rule-computed verdicts, counterfactual and
                        paraphrase variants (groups), plus the 23 hand-written verification cases
  labelling/*.csv       sheets for legal reviewers to confirm or correct each silver label

Usage (from the repo root):
    python backend/scripts/build_benchmark.py
"""

import csv
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.benchmark.factgen import generate  # noqa: E402
from app.benchmark.scenario_data import GROUPS, OUT_OF_SCOPE, SCENARIOS  # noqa: E402
from app.benchmark.statute import build as build_statute  # noqa: E402
from app.config import REPO_ROOT, get_settings  # noqa: E402
from calibrate_retrieval import OUT_OF_SCOPE as SHORT_OOS  # noqa: E402

OUT = REPO_ROOT / "benchmark"


def write_jsonl(rows: list[dict], name: str) -> None:
    with (OUT / name).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_sheet(rows: list[dict], cols: list[str], name: str) -> None:
    (OUT / "labelling").mkdir(parents=True, exist_ok=True)
    with (OUT / "labelling" / name).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["reviewer", "agree (yes/no)", "corrected label", "note"])
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") if not isinstance(r.get(c), list) else " ".join(r[c]) for c in cols})


def main() -> int:
    s = get_settings()
    OUT.mkdir(exist_ok=True)

    statute = [asdict(q) for q in build_statute(s.corpus_chunks_path)]
    write_jsonl(statute, "statute_qa.jsonl")

    earlier = json.loads((REPO_ROOT / "docs" / "eval" / "scenario_questions.json").read_text(encoding="utf-8"))["questions"]
    scenarios = [{"id": f"sc-{i:03d}", "question": q["q"], "acceptable": q["expected"], "group": "",
                  "split": q["split"], "source": "scenario_questions.json (2026-10-13)"}
                 for i, q in enumerate(x for x in earlier if x["split"] in ("dev", "test"))]
    scenarios += [{"id": f"sc-{len(scenarios) + i:03d}", "question": q, "acceptable": GROUPS[g], "group": g,
                   "split": "test", "source": "scenario_data.py (2026-10-14)"} for i, (q, g) in enumerate(SCENARIOS)]
    write_jsonl(scenarios, "scenarios.jsonl")

    oos = [q["q"] for q in earlier if q["split"] == "oos"] + list(SHORT_OOS) + list(OUT_OF_SCOPE)
    write_jsonl([{"id": f"oos-{i:03d}", "question": q} for i, q in enumerate(dict.fromkeys(oos))], "out_of_scope.jsonl")

    facts = [asdict(c) for c in generate()]
    natural = json.loads((REPO_ROOT / "docs" / "eval" / "verification_cases.json").read_text(encoding="utf-8"))["cases"]
    facts += [{"id": c["id"], "section": c["section"], "group": f"natural-{c['id']}", "kind": "natural", "change": {},
               "text": c["text"], "facts": {}, "gold_verdict": c["expected"], "gold_elements": {}} for c in natural]
    write_jsonl(facts, "fact_patterns.jsonl")

    write_sheet(scenarios, ["id", "question", "group", "acceptable"], "scenarios.csv")
    write_sheet(statute, ["id", "kind", "question", "section", "title", "gold_terms"], "statute_qa.csv")
    write_sheet([{**f, "elements": "; ".join(f"{k}={v}" for k, v in f["facts"].items() if v != "unknown")} for f in facts],
                ["id", "section", "kind", "text", "elements", "gold_verdict"], "fact_patterns.csv")
    write_sheet([{"id": r["id"], "question": r["question"]} for r in
                 (json.loads(line) for line in (OUT / "out_of_scope.jsonl").read_text(encoding="utf-8").splitlines())],
                ["id", "question"], "out_of_scope.csv")

    stats = {"statute_qa": dict(Counter(q["kind"] for q in statute)), "scenarios": dict(Counter(q["split"] for q in scenarios)),
             "out_of_scope": len(dict.fromkeys(oos)), "fact_patterns": dict(Counter(f["kind"] for f in facts)),
             "fact_pattern_verdicts": dict(Counter(f["gold_verdict"] for f in facts)),
             "fact_pattern_sections": dict(Counter(f["section"] for f in facts))}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
