"""Is the model's stated reasoning faithful to what actually drives its answer?

On held-out scenarios (benchmark seed-7 sample), LegalGPT answers; the model then writes its IRAC
reasoning trace (app/explainability/reasoning.py), naming the key facts and the sections that apply.
Both claims are tested by perturbation (app/explainability/perturbation.py):
  - key facts: each fact is removed and the answer written again from the same sources; a key fact
    should change the conclusion, a fact it did not name should not (agreement = faithfulness);
  - sections: each section is left out in turn; sections the trace says apply should be decisive.
A control (the unchanged input answered again) separates real influence from run-to-run variation.

Usage (repo root): python backend/scripts/eval_reasoning.py --n 20 --model groq:openai/gpt-oss-120b
Cached per question in data/reports/explain/reasoning_<model>.jsonl (rerun to resume).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
OUT = ROOT / "data" / "reports" / "explain"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="groq:openai/gpt-oss-120b")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    from app.config import get_settings
    from app.core.models import SourceEvidence
    from app.explainability.perturbation import by_section, fact_influence, source_influence
    from app.explainability.reasoning import build_trace
    from app.explainability.router import ExplainRouter
    from app.main import build_chat_router
    from eval_bias import patiently, safe
    from run_baselines import sample

    chat = build_chat_router(get_settings())
    llm = chat.models["groq" if args.model.startswith("groq:") else args.model]
    explain = ExplainRouter(chat)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"reasoning_{safe(args.model)}.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
    items = [q for q in sample(7, 40, 20, 60, 30) if q["set"] == "scenario"][: args.n]
    for it in items:
        if it["id"] in done:
            continue
        run, raw = patiently(explain.run_question, it["question"], llm, "auto")
        if run.refused or not raw:
            row = {"id": it["id"], "refused": True}
        else:
            sources = [SourceEvidence(**s) for s in raw]
            trace = patiently(build_trace, it["question"], run.answer, sources, llm)
            facts = patiently(fact_influence, it["question"], sources, llm, chat.prompt_builder, "ipc",
                              trace.get("key_facts", []), run.answer)
            srcs = patiently(source_influence, it["question"], run.answer, sources, llm, chat.prompt_builder, "ipc")
            row = {"id": it["id"], "refused": False, "question": it["question"], "answer": run.answer,
                   "statute_sections": [s.section for s in by_section(sources)], "trace": trace, "facts": facts, "sources": srcs}
        done[it["id"]] = row
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not row["refused"]:
            print(f"{it['id']}: faithfulness {row['facts']['faithfulness']['agreement']}, key facts "
                  f"{row['facts']['faithfulness']['key_facts_confirmed']}/{row['facts']['faithfulness']['key_facts']}, "
                  f"decisive {[r['section'] for r in row['sources']['sources'] if r['decisive']]}, "
                  f"applies {[r['section'] for r in row['trace'].get('rules', []) if r['applies']]}", flush=True)

    rows = [done[it["id"]] for it in items if it["id"] in done and not done[it["id"]]["refused"] and "error" not in done[it["id"]]["trace"]]
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    kf = [r["facts"]["faithfulness"] for r in rows]
    applies_dec, applies_all, dec_all = 0, 0, 0
    for r in rows:
        applies = {x["section"] for x in r["trace"]["rules"] if x["applies"]}
        decisive = {x["section"] for x in r["sources"]["sources"] if x["decisive"]}
        applies_dec += len(applies & decisive)
        applies_all += len(applies)
        dec_all += len(decisive)
    quoted = sum(len(r["trace"]["key_facts"]) for r in rows)
    unquoted = sum(len(r["trace"].get("unquoted_key_facts", [])) for r in rows)
    summary = {
        "model": args.model, "questions": len(rows), "refused": sum(1 for it in items if done.get(it["id"], {}).get("refused")),
        "trace": {"invalid_items_removed_per_trace": mean([r["trace"]["removed"] for r in rows]),
                  "key_facts_verbatim": round(quoted / max(1, quoted + unquoted), 3)},
        "facts": {"agreement_mean": mean([f["agreement"] for f in kf if f["agreement"] is not None]),
                  "key_facts_confirmed": sum(f["key_facts_confirmed"] for f in kf), "key_facts": sum(f["key_facts"] for f in kf),
                  "unnamed_influential": sum(f["unnamed_but_influential"] for f in kf), "unnamed": sum(f["unnamed"] for f in kf),
                  "control_changed": round(sum(r["facts"]["control"]["changed"] for r in rows) / max(1, len(rows)), 3)},
        "sections": {"applies_and_decisive": applies_dec, "applies": applies_all, "decisive": dec_all,
                     "precision": round(applies_dec / applies_all, 3) if applies_all else None,
                     "recall": round(applies_dec / dec_all, 3) if dec_all else None,
                     "control_changed": round(sum(r["sources"]["control"]["changed"] for r in rows) / max(1, len(rows)), 3)},
    }
    (OUT / f"reasoning_summary_{safe(args.model)}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
