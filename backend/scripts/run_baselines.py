"""LegalGPT vs closed-book chatbots on the benchmark (benchmark/*.jsonl).

Systems (any number, each answers the same seeded sample; answers are cached and resumed):
  legalgpt:<groq model>      the app's legal path: retrieval + the model, refusal when nothing is retrieved
  legalgpt-lora              retrieval + the local LoRA adapter (GPU environment)
  closed:<groq model>[@effort]  the model alone, with a neutral "answer about Indian law, cite sections" prompt;
                             @low/@medium/@high sets gpt-oss reasoning effort (gpt-oss-20b at the default
                             effort often spends the whole output budget on hidden reasoning)
  legalgpt+<provider>:<model>  LegalGPT's pipeline with another provider's model as the LLM
  closed+<provider>:<model>    that model alone, closed-book
                             providers (app/core/openai_compat.py): gemini (GEMINI_API_KEY, free tier),
                             openai (OPENAI_API_KEY), openrouter (OPENROUTER_API_KEY);
                             e.g. legalgpt+gemini:gemini-3.5-flash-lite
  anthropic:<model>          Claude closed-book (ANTHROPIC_API_KEY)
  --smoke                    4 questions per system first, to check keys and model names
  --budget-usd 1 --price 0.75 4.5
                             stop before a paid run can exceed $1 (prices per million tokens)

Scored automatically (app/benchmark/metrics.py): correct section, correct punishment terms, wrong
punishment, invented sections, IPC/BNS confusion, scenario hits, and civil questions treated as
IPC offences. Usage:
    python backend/scripts/run_baselines.py run --systems legalgpt:openai/gpt-oss-120b closed:openai/gpt-oss-120b
    python backend/scripts/run_baselines.py score
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.benchmark.metrics import load_jsonl, score_oos, score_scenario, score_statute  # noqa: E402
from app.config import REPO_ROOT, get_settings  # noqa: E402

BENCH = REPO_ROOT / "benchmark"
OUT = REPO_ROOT / "data" / "reports" / "bench" / "baselines"
CLOSED_SYSTEM = ("You are a helpful legal assistant for Indian law. Answer the user's question accurately and concisely. "
                 "When a statutory provision applies, cite it by section number.")
MARKER = re.compile(r"\[(\d+)\]")
# Reasoning models (gpt-oss) spend part of the output budget on hidden reasoning; at the app's 2048
# gpt-oss-20b often returned nothing. Every Groq system gets the same, larger budget.
MAX_TOKENS = 4096


def sample(seed: int, n_pun: int, n_def: int, n_sc: int, n_oos: int) -> list[dict]:
    rng = random.Random(seed)
    st = load_jsonl(BENCH / "statute_qa.jsonl")
    pun = rng.sample([q for q in st if q["kind"] == "punishment"], n_pun)
    dfn = rng.sample([q for q in st if q["kind"] == "definition"], n_def)
    sc = rng.sample([q for q in load_jsonl(BENCH / "scenarios.jsonl") if q["split"] == "test"], n_sc)
    oos = rng.sample(load_jsonl(BENCH / "out_of_scope.jsonl"), n_oos)
    return ([{"set": "statute", **q} for q in pun + dfn] + [{"set": "scenario", **q} for q in sc]
            + [{"set": "oos", **q} for q in oos])


class HttpChat:
    """Minimal closed-book clients for other providers (no extra dependencies)."""

    def __init__(self, provider: str, model: str) -> None:
        import httpx

        self.provider, self.model, self.http = provider, model, httpx.Client(timeout=120)
        env = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}[provider]
        self.key = os.environ.get(env) or _dotenv(env)
        if not self.key:
            raise SystemExit(f"{env} is not set: add it to .env to include {provider}:{model}")

    def ask(self, question: str) -> str:
        if self.provider == "openai":
            r = self.http.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {self.key}"},
                               json={"model": self.model, "messages": [{"role": "system", "content": CLOSED_SYSTEM},
                                                                       {"role": "user", "content": question}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        if self.provider == "anthropic":
            r = self.http.post("https://api.anthropic.com/v1/messages",
                               headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"},
                               json={"model": self.model, "max_tokens": 1024, "system": CLOSED_SYSTEM,
                                     "messages": [{"role": "user", "content": question}]})
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json()["content"])
        r = self.http.post(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                           params={"key": self.key},
                           json={"systemInstruction": {"parts": [{"text": CLOSED_SYSTEM}]},
                                 "contents": [{"role": "user", "parts": [{"text": question}]}]})
        r.raise_for_status()
        return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])


def _dotenv(name: str) -> str:
    p = REPO_ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"')
    return ""


def make_system(spec: str):
    """-> callable(question) -> (answer, extra_cited_sections)."""
    from app.core.llm_client import AdapterClient, GroqClient, latest_adapter
    from app.core.models import Message
    from app.core.prompt_builder import NOT_COVERED_REPLY, PromptBuilder
    from app.core.retriever import build_retriever

    from app.core.openai_compat import PROVIDERS, OpenAICompatClient

    s = get_settings()
    kind, _, model = spec.partition(":")
    if kind in ("openai", "gemini"):  # older spelling: openai:<model> = closed+openai:<model>
        kind = f"closed+{kind}"
    kind, _, provider = kind.partition("+")  # legalgpt+gemini:gemini-2.5-flash
    if provider and provider not in PROVIDERS:
        raise SystemExit(f"unknown provider {provider!r} in {spec!r}; use one of {', '.join(PROVIDERS)}")

    def hosted(name: str):
        return (OpenAICompatClient(provider, name, max_tokens=MAX_TOKENS) if provider
                else GroqClient(s.groq_api_key, name, max_tokens=MAX_TOKENS))

    if kind in ("legalgpt", "legalgpt-lora"):
        retriever = build_retriever(s)
        llm = hosted(model) if kind == "legalgpt" else AdapterClient(s.adapter_path or latest_adapter(s.lora_dir))

        def run(q: str):
            sources = retriever.retrieve(q)
            if not sources:
                return NOT_COVERED_REPLY, []
            answer = llm.generate(PromptBuilder().build(q, sources, "legal", []))
            by_marker = {x.marker: x.section for x in sources}
            return answer, [by_marker[int(m)] for m in MARKER.findall(answer) if int(m) in by_marker and by_marker[int(m)]]
        run.llm = llm
        return run
    if kind == "closed":
        model, _, effort = model.partition("@")  # closed:openai/gpt-oss-20b@low -> reasoning_effort="low" (Groq)
        llm = hosted(model)
        extra = {"reasoning_effort": effort} if effort and not provider else {}

        def ask(q: str):
            msgs = [Message("system", CLOSED_SYSTEM), Message("user", q)]
            return (llm._create(msgs, **extra).choices[0].message.content or "") if extra else llm.generate(msgs), []
        ask.llm = llm
        return ask
    if kind == "anthropic":
        client = HttpChat(kind, model)
        return lambda q: (client.ask(q), [])
    raise SystemExit(f"unknown system {spec!r}")


def slug(spec: str) -> str:
    return re.sub(r"[^\w.-]+", "_", spec)


def run(args) -> None:
    from app.core.llm_client import LLMRateLimitError

    from app.benchmark.quota import QuotaWaiter

    waiter = QuotaWaiter(args.patient, args.max_wait_hours)
    items = sample(args.seed, *((1, 1, 1, 1) if args.smoke else args.sample))
    out = OUT / "smoke" if args.smoke else OUT
    if args.smoke:  # a fresh 4-question check of keys and model names; never touches the real caches
        for f in out.glob("*.jsonl"):
            f.unlink()
    out.mkdir(parents=True, exist_ok=True)
    spent = 0.0  # USD in this invocation, from the providers' token counts and --price
    cost = lambda u: (u["in"] * args.price[0] + u["out"] * args.price[1]) / 1e6
    worst = cost({"in": 6000, "out": MAX_TOKENS})  # the most one answer can cost
    for spec in args.systems:
        path = out / f"{slug(spec)}.jsonl"
        done = {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
        todo = [it for it in items if it["id"] not in done]
        print(f"{spec}: {len(done)} cached, {len(todo)} to answer")
        if not todo:
            continue
        system = make_system(spec)
        usage = getattr(getattr(system, "llm", None), "usage", None)
        with path.open("a", encoding="utf-8") as f:
            for it in todo:
                if args.budget_usd is not None and spent + worst > args.budget_usd:
                    print(f"  stopped: ${spent:.3f} spent; one more answer could pass the ${args.budget_usd} budget")
                    return
                before = dict(usage) if usage else None
                t = time.perf_counter()
                answer = None
                attempt = 0
                while attempt < 6:
                    try:
                        answer, extra = system(it["question"])
                        break
                    except LLMRateLimitError as e:
                        if "daily" in str(e).lower():
                            if waiter.wait(e):
                                continue
                            print(f"  stopped (daily limit): {e}")
                            break
                        attempt += 1
                        time.sleep(20 * attempt)  # per-minute limit: wait and retry
                    except Exception as e:  # keep going; record the failure
                        answer, extra = f"ERROR: {type(e).__name__}: {e}", []
                        break
                if answer is None:
                    break
                row = {"id": it["id"], "set": it["set"], "answer": answer, "extra_cited": extra,
                       "seconds": round(time.perf_counter() - t, 2)}
                if usage:
                    row["tokens"] = {k: usage[k] - before[k] for k in usage}
                    spent += cost(row["tokens"])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if args.smoke:
                    ok = "FAILED" if answer.startswith("ERROR") or not answer.strip() else "ok"
                    print(f"  [{ok}] {it['set']}: {it['question'][:60]}\n        -> {answer[:160]!r}")
            if usage:
                print(f"  {spec}: {usage['in']} input + {usage['out']} output tokens; ${spent:.3f} spent so far")


def score(args) -> None:
    from app.core.corpus import load_chunks

    ipc = {c.section for c in load_chunks(get_settings().corpus_chunks_path) if c.section}
    items = {it["id"]: it for it in sample(args.seed, *args.sample)}
    rows = []
    for path in sorted(OUT.glob("*.jsonl")):
        allans = [a for a in load_jsonl(path) if a["id"] in items and not a["answer"].startswith("ERROR")]
        ans = [a for a in allans if a["answer"].strip()]
        if not ans:
            continue
        st = [score_statute(items[a["id"]], a["answer"], a["extra_cited"], ipc) | {"kind": items[a["id"]]["kind"]}
              for a in ans if a["set"] == "statute"]
        pun = [r for r in st if r["kind"] == "punishment"]
        sc = [score_scenario(items[a["id"]], a["answer"], a["extra_cited"], ipc) for a in ans if a["set"] == "scenario"]
        oo = [score_oos(a["answer"], a["extra_cited"]) for a in ans if a["set"] == "oos"]
        pct = lambda xs, k: f"{100 * sum(bool(x[k]) for x in xs) / len(xs):.0f}%" if xs else "-"
        rows.append({"system": path.stem, "n": len(ans), "empty": len(allans) - len(ans), "statute_n": len(st), "section_correct": pct(st, "section_correct"),
                     "punishment_correct": pct(pun, "punishment_correct"), "punishment_wrong": pct(pun, "punishment_wrong"),
                     "invented": pct(st + sc, "invented"), "bns_confusion": pct(st, "bns_confusion"),
                     "mentions_bns": pct(st + sc, "mentions_bns"), "scenario_n": len(sc), "scenario_hit": pct(sc, "hit"),
                     "scenario_wrong_only": pct(sc, "wrong_only"), "scenario_refused": pct(sc, "refused"),
                     "oos_n": len(oo), "oos_criminalised": pct(oo, "criminalised"), "oos_refused": pct(oo, "refused"),
                     "mean_seconds": round(sum(a["seconds"] for a in ans) / len(ans), 1)})
    cols = list(rows[0]) if rows else []
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n".join(md))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["run", "score"])
    ap.add_argument("--systems", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--patient", action="store_true", help="wait out daily limits instead of stopping")
    ap.add_argument("--smoke", action="store_true",
                    help="answer just 4 questions per system (in a separate folder) to check keys and model names")
    ap.add_argument("--max-wait-hours", type=float, default=12.0)
    ap.add_argument("--budget-usd", type=float, help="stop before spending more than this (needs --price)")
    ap.add_argument("--price", type=float, nargs=2, default=[0.0, 0.0], metavar=("IN", "OUT"),
                    help="USD per million input / output tokens, for --budget-usd and the cost printout")
    ap.add_argument("--sample", type=int, nargs=4, default=[40, 20, 60, 30], metavar=("PUN", "DEF", "SCEN", "OOS"))
    args = ap.parse_args()
    {"run": run, "score": score}[args.step](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
