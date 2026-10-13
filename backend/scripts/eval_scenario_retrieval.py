"""Retrieval on everyday fact-pattern questions (docs/eval/scenario_questions.json).

For each question: is any expected section among the retrieved sources (hit@sources), its rank,
whether retrieval refused (no source above the floor), and the best similarity. Reports dev and
test splits separately - the test split is for the final measurement only.

Usage (from the repo root):
    python backend/scripts/eval_scenario_retrieval.py              # both splits
    python backend/scripts/eval_scenario_retrieval.py --split dev --show
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import REPO_ROOT, get_settings  # noqa: E402
from app.core.retriever import Retriever, build_retriever  # noqa: E402


def evaluate(retriever: Retriever, questions: list[dict]) -> dict:
    rows = []
    for it in questions:
        res = retriever.retrieve_with_diagnostics(it["q"])
        secs = [s.section for s in res.sources]
        rank = next((i + 1 for i, s in enumerate(secs) if s in it["expected"]), None)
        rows.append({**it, "sections": secs, "rank": rank, "refused": not secs, "best": round(res.best_similarity, 3)})
    n = len(rows)
    return {"n": n, "hit": sum(r["rank"] is not None for r in rows) / n, "refused": sum(r["refused"] for r in rows) / n,
            "top1": sum(r["rank"] == 1 for r in rows) / n, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--show", action="store_true", help="print every question")
    ap.add_argument("--out", type=Path, default=None, help="write the results as JSON")
    ap.add_argument("--no-scope", action="store_true", help="ablation: floor only, no scope check")
    ap.add_argument("--no-boilerplate", action="store_true", help="ablation: no state-amendment / Chapter I penalty")
    ap.add_argument("--scope-floor", type=float, default=None)
    ap.add_argument("--no-lexicon", action="store_true", help="ablation: no lay-to-statutory query expansion")
    args = ap.parse_args()
    s = get_settings()
    qs = json.loads((REPO_ROOT / "docs" / "eval" / "scenario_questions.json").read_text(encoding="utf-8"))["questions"]
    retriever = build_retriever(s, scope=not args.no_scope)
    if args.no_boilerplate:
        retriever.boilerplate_pref = {}
    if args.no_lexicon:
        retriever.expander.search_query = retriever.expander.expand
        import app.core.retriever as retriever_module

        retriever_module.statutory_terms = lambda q: []
    if args.scope_floor is not None:
        retriever.scope_floor = args.scope_floor
    out = {}
    for split in (["dev", "test"] if args.split == "all" else [args.split]):
        r = evaluate(retriever, [q for q in qs if q["split"] == split])
        out[split] = r
        print(f"{split}: {r['n']} questions  hit {r['hit']:.0%}  top-1 {r['top1']:.0%}  refused {r['refused']:.0%}")
        if args.show:
            for row in r["rows"]:
                mark = "ok " if row["rank"] else ("REF" if row["refused"] else "MISS")
                print(f"  {mark} best={row['best']:.2f} rank={row['rank']} want={row['expected']} got={row['sections']}  {row['q'][:70]}")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from calibrate_retrieval import OUT_OF_SCOPE

    oos = [q["q"] for q in qs if q["split"] == "oos"] + OUT_OF_SCOPE
    answered = [q for q in oos if retriever.retrieve(q)]
    out["out_of_scope"] = {"n": len(oos), "answered": answered}
    print(f"out of scope: {len(oos)} questions  wrongly answered {len(answered)}" + (f": {answered}" if answered else ""))
    if args.out:
        args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
