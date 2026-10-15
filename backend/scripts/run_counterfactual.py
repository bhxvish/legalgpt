"""Counterfactual test: LLM fact extraction + Prolog rules vs the same LLM deciding the verdict directly.

On benchmark/fact_patterns.jsonl (gold verdicts computed by the rules):
  pipeline   the app's verifier: the LLM extracts quoted facts, Prolog applies the section's elements
  direct     the LLM gets the facts, the section's elements and the definitions of the three verdicts,
             and answers the verdict itself (JSON)

Measured per arm:
  accuracy             verdict = gold, overall and per case kind
  flip correctness     a one-fact flip is right AND its base case is right (did the verdict change
                       exactly as the law says?)
  omit correctness     the same for an omitted fact (should become INSUFFICIENT)
  paraphrase stability base and paraphrase get the same verdict
  unsafe errors        a definite verdict (CONSISTENT / INCONSISTENT) that is wrong, vs cautious
                       errors (INSUFFICIENT when the facts were decisive)

Answers are cached per (model, arm, runs) and resumed. Usage:
    python backend/scripts/run_counterfactual.py run --model qwen/qwen3.8-27b --frames 0
    python backend/scripts/run_counterfactual.py score
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.benchmark.metrics import load_jsonl  # noqa: E402
from app.config import REPO_ROOT, get_settings  # noqa: E402

BENCH = REPO_ROOT / "benchmark" / "fact_patterns.jsonl"
OUT = REPO_ROOT / "data" / "reports" / "bench" / "counterfactual"
VERDICTS = ("CONSISTENT", "INCONSISTENT", "INSUFFICIENT")
DIRECT_SYSTEM = (
    "You are a careful legal analyst applying the Indian Penal Code. Decide whether the facts described satisfy every "
    "element of the given section. Use only the facts stated; do not assume anything that is not stated.\n"
    "Verdicts:\n"
    "- CONSISTENT: every element is clearly established by the facts.\n"
    "- INCONSISTENT: the facts contradict at least one element.\n"
    "- INSUFFICIENT: no element is contradicted, but at least one element is not established by the facts.\n"
    'Reply with one JSON object only: {"elements": {"<element>": "satisfied|violated|missing", ...}, "verdict": "CONSISTENT|INCONSISTENT|INSUFFICIENT"}'
)


def select(frames: list[int] | None, natural: bool) -> list[dict]:
    rows = load_jsonl(BENCH)
    out = []
    for r in rows:
        if r["kind"] == "natural":
            if natural:
                out.append(r)
            continue
        if frames is None or int(re.search(r"-f(\d+)$", r["group"]).group(1)) in frames:
            out.append(r)
    return out


def _parse_direct(text: str) -> tuple[str | None, dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            v = str(d.get("verdict", "")).upper()
            return (v if v in VERDICTS else None), d.get("elements", {})
        except json.JSONDecodeError:
            pass
    v = next((x for x in VERDICTS if x in text.upper()), None)
    return v, {}


def make_client(model: str, provider: str = "groq", route: str | None = None):
    """The model on Groq (default), or the same model through OpenRouter pinned to one host (`route`),
    e.g. when the Groq free tier is used up. Rows record the route (`via`)."""
    from app.core.llm_client import GroqClient

    if provider == "groq":
        return GroqClient(get_settings().groq_api_key, model, max_tokens=4096, temperature=0.0)
    from app.core.openai_compat import OpenAICompatClient

    pin = {"provider": {"order": [route], "allow_fallbacks": False}} if route else None
    return OpenAICompatClient(provider, model, max_tokens=4096, temperature=0.0, extra_body=pin)


def make_arm(arm: str, model: str, runs: int, provider: str = "groq", route: str | None = None):
    from app.core.models import Message
    from app.verification.fact_extractor import FactExtractor
    from app.verification.prolog_engine import PrologEngine
    from app.verification.verification_service import VerificationService

    s = get_settings()
    engine = PrologEngine(s.prolog_rules_dir)
    sections = {sec.section: sec for sec in engine.sections()}
    if arm == "pipeline":
        client = make_client(model, provider, route)
        svc = VerificationService(FactExtractor(client), engine, extraction_runs=runs)

        def run(case):
            r = svc.verify(case["text"], case["section"])
            return r.status, {e.predicate: e.status for e in r.elements}
        run.llm = client
        return run
    llm = make_client(model, provider, route)

    def run(case):
        sec = sections[case["section"]]
        elements = "\n".join(f"- {e.predicate}: {e.description}" for e in sec.elements)
        user = f"Section {sec.section} ({sec.title}). Elements:\n{elements}\n\nFacts:\n{case['text']}"
        return _parse_direct(llm.generate([Message("system", DIRECT_SYSTEM), Message("user", user)]))
    run.llm = llm
    return run


def cache_path(model: str, arm: str, runs: int) -> Path:
    name = re.sub(r"[^\w.-]+", "_", model)
    return OUT / (f"{name}__{arm}" + (f"__runs{runs}" if arm == "pipeline" else "") + ".jsonl")


def run(args) -> None:
    from app.core.llm_client import LLMRateLimitError

    from app.benchmark.quota import QuotaWaiter

    waiter = QuotaWaiter(args.patient, args.max_wait_hours)
    cases = select(args.frames, args.natural)
    if args.match_direct:  # only the cases the direct arm answered, so the two arms are paired
        direct = OUT / f"{args.model.replace('/', '_')}__direct.jsonl"
        done = {r["id"] for r in load_jsonl(direct)} if direct.exists() else set()
        cases = [c for c in cases if c["id"] in done]
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        path = cache_path(args.model, arm, args.runs)
        done = {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
        todo = [c for c in cases if c["id"] not in done]
        print(f"{args.model} {arm}: {len(done)} cached, {len(todo)} to run")
        if not todo:
            continue
        fn = make_arm(arm, args.model, args.runs, args.provider, args.route)
        usage = getattr(getattr(fn, "llm", None), "usage", None)  # OpenAI-compatible providers count tokens
        cost = lambda: (usage["in"] * args.price[0] + usage["out"] * args.price[1]) / 1e6 if usage else 0.0
        with path.open("a", encoding="utf-8") as f:
            for c in todo:
                if args.budget_usd is not None and cost() + 0.02 > args.budget_usd:
                    print(f"  stopped: ${cost():.3f} spent, the ${args.budget_usd} budget is nearly reached")
                    return
                t, pred, els, err = time.perf_counter(), None, {}, ""
                attempt = 0
                while attempt < 6:
                    try:
                        pred, els = fn(c)
                        break
                    except LLMRateLimitError as e:
                        if "daily" in str(e).lower():
                            if waiter.wait(e):
                                continue
                            print(f"  stopped (daily limit): {e}")
                            return
                        attempt += 1
                        time.sleep(20 * attempt)
                    except Exception as e:  # extraction failure etc.: record and continue
                        err = f"{type(e).__name__}: {e}"[:300]
                        break
                row = {"id": c["id"], "pred": pred, "elements": els, "error": err, "seconds": round(time.perf_counter() - t, 1)}
                if args.provider != "groq":
                    row["via"] = args.provider + (f" (host {args.route})" if args.route else "")
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            if usage:
                print(f"  {arm}: {usage['in']} input + {usage['out']} output tokens; ${cost():.3f}")


def score(args) -> None:
    gold = {r["id"]: r for r in load_jsonl(BENCH)}
    lines = ["| model | arm | n | accuracy | base | paraphrase | flip | omit | natural | flip correct (with base) | "
             "omit correct (with base) | paraphrase stable | unsafe errors | cautious errors | no verdict |",
             "|" + "---|" * 15]
    summary = []
    for path in sorted(OUT.glob("*.jsonl")):
        res = {r["id"]: r for r in load_jsonl(path)}
        rows = [(gold[i], r) for i, r in res.items() if i in gold]
        if not rows:
            continue
        ok = {g["id"]: r["pred"] == g["gold_verdict"] for g, r in rows}
        pct = lambda xs: f"{100 * sum(xs) / len(xs):.0f}% ({sum(xs)}/{len(xs)})" if xs else "-"
        by_kind = defaultdict(list)
        for g, r in rows:
            by_kind[g["kind"]].append(ok[g["id"]])
        base = {g["group"]: g["id"] for g, _ in rows if g["kind"] == "base"}
        para = {g["group"]: g["id"] for g, _ in rows if g["kind"] == "paraphrase"}
        flip_ok = [ok[g["id"]] and ok.get(base.get(g["group"]), False) for g, _ in rows if g["kind"] == "flip" and g["group"] in base]
        omit_ok = [ok[g["id"]] and ok.get(base.get(g["group"]), False) for g, _ in rows if g["kind"] == "omit" and g["group"] in base]
        stable = [res[base[gr]]["pred"] == res[para[gr]]["pred"] for gr in base if gr in para]
        unsafe = [r["pred"] in ("CONSISTENT", "INCONSISTENT") and r["pred"] != g["gold_verdict"] for g, r in rows]
        cautious = [r["pred"] == "INSUFFICIENT" and g["gold_verdict"] != "INSUFFICIENT" for g, r in rows]
        none = [r["pred"] is None for _, r in rows]
        model, _, rest = path.stem.partition("__")
        lines.append(f"| {model} | {rest} | {len(rows)} | {pct(list(ok.values()))} | {pct(by_kind['base'])} | "
                     f"{pct(by_kind['paraphrase'])} | {pct(by_kind['flip'])} | {pct(by_kind['omit'])} | {pct(by_kind['natural'])} | "
                     f"{pct(flip_ok)} | {pct(omit_ok)} | {pct(stable)} | {pct(unsafe)} | {pct(cautious)} | {pct(none)} |")
        summary.append({"model": model, "arm": rest, "n": len(rows), "accuracy": sum(ok.values()) / len(rows),
                        "flip_correct": sum(flip_ok) / len(flip_ok) if flip_ok else None,
                        "omit_correct": sum(omit_ok) / len(omit_ok) if omit_ok else None,
                        "paraphrase_stable": sum(stable) / len(stable) if stable else None,
                        "unsafe": sum(unsafe) / len(rows), "cautious": sum(cautious) / len(rows)})
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["run", "score"])
    ap.add_argument("--model", default="qwen/qwen3.8-27b")
    ap.add_argument("--arms", nargs="+", default=["direct", "pipeline"], choices=["direct", "pipeline"])
    ap.add_argument("--runs", type=int, default=1, help="extraction runs for the pipeline arm (the app uses 2)")
    ap.add_argument("--frames", type=int, nargs="*", default=None, help="frame indices to include (default: all)")
    ap.add_argument("--match-direct", action="store_true",
                    help="run only the cases the direct arm already answered (a paired comparison)")
    ap.add_argument("--natural", action="store_true", help="also run the 23 hand-written cases")
    ap.add_argument("--patient", action="store_true", help="wait out daily limits instead of stopping")
    ap.add_argument("--max-wait-hours", type=float, default=12.0)
    ap.add_argument("--provider", default="groq", choices=["groq", "openrouter"],
                    help="where the model runs; openrouter when the Groq free tier is used up")
    ap.add_argument("--route", help="OpenRouter host to pin, no fallbacks (e.g. Cerebras)")
    ap.add_argument("--budget-usd", type=float, help="stop before spending more than this (needs --price)")
    ap.add_argument("--price", type=float, nargs=2, default=[0.0, 0.0], metavar=("IN", "OUT"),
                    help="USD per million input / output tokens")
    args = ap.parse_args()
    {"run": run, "score": score}[args.step](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
