"""Retrieval on the benchmark's everyday scenarios, with splits that keep the benchmark held out.

The end-to-end benchmark (run_baselines.py, seed 7) scores 60 test scenarios and 30 out-of-scope
questions. Retrieval is designed on the *other* questions only:

  tune     dev scenarios + half of the unused test scenarios (by id)    used while designing
  check    the other half of the unused test scenarios                  confirms a change generalises
  final    the 60 benchmark scenarios                                   scored once, at the end
  oos-tune out-of-scope questions not in the benchmark sample, plus docs/eval and calibration sets
  oos-final the 30 benchmark out-of-scope questions

For each scenario: hit = an acceptable section is among the retrieved sources; wrong = sources but
none acceptable; refused = no sources. For out-of-scope questions, "answered" means sources were
returned (the chat would then answer as if it were an IPC offence).

Usage (from the repo root):
    python backend/scripts/eval_scenario_bench.py                    # tune, check, oos-tune
    python backend/scripts/eval_scenario_bench.py --splits tune --show
    python backend/scripts/eval_scenario_bench.py --splits final oos-final   # once, at the end
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.benchmark.metrics import load_jsonl  # noqa: E402
from app.config import REPO_ROOT, get_settings  # noqa: E402

BENCH = REPO_ROOT / "benchmark"


def splits(seed: int = 7) -> dict[str, list[dict]]:
    from run_baselines import sample

    held = {it["id"] for it in sample(seed, 40, 20, 60, 30)}
    scen = load_jsonl(BENCH / "scenarios.jsonl")
    spare = sorted((q for q in scen if q["split"] == "test" and q["id"] not in held), key=lambda q: q["id"])
    oos = load_jsonl(BENCH / "out_of_scope.jsonl")
    from calibrate_retrieval import OUT_OF_SCOPE

    docs = json.loads((REPO_ROOT / "docs" / "eval" / "scenario_questions.json").read_text(encoding="utf-8"))["questions"]
    oos_tune = ([q["question"] for q in oos if q["id"] not in held] + [q["q"] for q in docs if q["split"] == "oos"]
                + list(OUT_OF_SCOPE))
    return {
        "tune": [q for q in scen if q["split"] == "dev"] + spare[0::2],
        "check": spare[1::2],
        "final": [q for q in scen if q["id"] in held],
        "oos-tune": [{"question": q} for q in dict.fromkeys(oos_tune)],
        "oos-final": [q for q in oos if q["id"] in held],
    }


class CachedRewriter:
    """LLM rewrites cached on disk (data/reports/bench/rewrites/), so settings can be compared
    without calling the model again."""

    def __init__(self, spec: str, patient: bool = False) -> None:
        from app.benchmark.quota import QuotaWaiter

        self.waiter = QuotaWaiter(True, 30.0) if patient else None
        from app.core.openai_compat import OpenAICompatClient, parse_spec

        provider, model = parse_spec(spec) if ":" in spec and spec.split(":")[0] != "groq" else ("groq", spec.split(":", 1)[-1])
        if provider == "groq":
            from app.core.llm_client import GroqClient

            self.llm = GroqClient(get_settings().groq_api_key, model, max_tokens=2048)
        else:
            self.llm = OpenAICompatClient(provider, model, max_tokens=2048)
        name = re.sub(r"[^\w.-]+", "_", spec)
        self.path = REPO_ROOT / "data" / "reports" / "bench" / "rewrites" / f"{name}.json"
        self.cache = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def __call__(self, question: str):
        from app.core.models import Message
        from app.core.query_rewriter import REWRITE_PROMPT, parse

        if question not in self.cache:
            import time

            from app.core.llm_client import LLMRateLimitError

            attempt = 0
            while True:
                try:
                    ask = getattr(self.llm, "generate_brief", self.llm.generate)  # as QueryRewriter does
                    reply = ask([Message("system", REWRITE_PROMPT), Message("user", question)])
                    break
                except LLMRateLimitError as e:  # free tiers: per-minute limits; daily ones with --patient
                    if "daily" in str(e).lower():
                        if self.waiter and self.waiter.wait(e):
                            continue
                        raise
                    attempt += 1
                    if attempt == 6:
                        raise
                    time.sleep(20 * attempt)
            self.cache[question] = reply
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache, indent=0, ensure_ascii=False), encoding="utf-8")
        return parse(self.cache[question], self.llm.model_id)


def evaluate(retriever, items: list[dict], oos: bool = False, rewriter=None) -> dict:
    rows = []
    for it in items:
        rw = rewriter(it["question"]) if rewriter else None
        res = retriever.retrieve_with_diagnostics(it["question"], rewrite=rw)
        secs = [s.section for s in res.sources]
        if oos:
            rows.append({"q": it["question"], "sections": secs, "answered": bool(secs), "rewrite": res.rewrite})
            continue
        ok = set(it["acceptable"])
        rank = next((i + 1 for i, s in enumerate(secs) if s in ok), None)
        rows.append({"q": it["question"], "want": it["acceptable"], "sections": secs, "rank": rank,
                     "refused": not secs, "wrong": bool(secs) and rank is None, "best": round(res.best_similarity, 3),
                     "rewrite": res.rewrite})
    n = len(rows) or 1
    if oos:
        return {"n": len(rows), "answered": sum(r["answered"] for r in rows) / n, "rows": rows}
    return {"n": len(rows), "hit": sum(r["rank"] is not None for r in rows) / n, "top1": sum(r["rank"] == 1 for r in rows) / n,
            "wrong": sum(r["wrong"] for r in rows) / n, "refused": sum(r["refused"] for r in rows) / n, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="*", default=["tune", "check", "oos-tune"],
                    choices=["tune", "check", "final", "oos-tune", "oos-final"])
    ap.add_argument("--show", action="store_true", help="print every miss")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--rewrite", metavar="MODEL", help="LLM query rewrite, e.g. gemini:gemini-3.5-flash-lite or groq:openai/gpt-oss-120b")
    ap.add_argument("--patient", action="store_true", help="wait out daily free-tier limits instead of stopping")
    ap.add_argument("--gate", type=float, nargs="*", help="rewrite gates (scope probability) to compare")
    args = ap.parse_args()
    from app.core.retriever import build_retriever

    retriever = build_retriever(get_settings())
    rewriter = CachedRewriter(args.rewrite, args.patient) if args.rewrite else None
    data, out = splits(), {}
    for gate in (args.gate or [retriever.rewrite_gate]):
        retriever.rewrite_gate = gate
        if args.gate:
            print(f"--- rewrite gate {gate}")
        report(retriever, rewriter, data, args, out)
    if args.out:
        args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


def report(retriever, rewriter, data, args, out) -> None:
    for name in args.splits:
        r = evaluate(retriever, data[name], oos=name.startswith("oos"), rewriter=rewriter)
        out[name] = r
        if name.startswith("oos"):
            print(f"{name:9} n={r['n']:3}  wrongly answered {r['answered']:.0%}")
            if args.show:
                for row in r["rows"]:
                    if row["answered"]:
                        print(f"   ANSWERED {row['sections'][:4]}  {row['q'][:90]}")
        else:
            print(f"{name:9} n={r['n']:3}  hit {r['hit']:.0%}  top-1 {r['top1']:.0%}  wrong {r['wrong']:.0%}  refused {r['refused']:.0%}")
            if args.show:
                for row in r["rows"]:
                    if row["rank"] is None:
                        tag = "REF " if row["refused"] else "MISS"
                        rw = row["rewrite"] and [o["section"] for o in row["rewrite"]["offences"]]
                        print(f"   {tag} best={row['best']:.2f} want={row['want']} got={row['sections'][:4]} rw={rw}  {row['q'][:80]}")


if __name__ == "__main__":
    raise SystemExit(main())
