"""Hindi questions: retrieval and answers on a Hindi version of the benchmark's held-out questions.

Data: the 60 benchmark scenarios and 30 out-of-scope questions (run_baselines.py, seed 7),
machine-translated into Hindi by --translate-model (a different model from the one under test), in
benchmark/hindi/questions_hi.jsonl. The translations were not checked by a native speaker; the file
says so.

Retrieval, per question (hit = an acceptable section among the sources; for out-of-scope questions,
"answered" = any sources, which should not happen):
  en            the English question, with the LLM query rewrite (the app on English)
  hi-raw        the Hindi question as is, no rewrite (MiniLM is an English model)
  hi-rewrite    the Hindi question with the rewrite reading the Hindi (the app before this change)
  hi-translate  the Hindi question translated to English first, then the rewrite (the app now)
Rewrites and translations are by --model and cached under data/reports/hindi/.

--answers: also runs the chat end to end on the Hindi questions (the chat's own translation, rewrite
and answer, with --model) and reports: refused, written in Hindi (>= 50% Devanagari letters), cites
an acceptable section, citation numbers that exist; for out-of-scope questions, refused.

Usage:  python backend/scripts/eval_hindi.py [--answers] [--limit N]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import REPO_ROOT, get_settings  # noqa: E402

DATA = REPO_ROOT / "benchmark" / "hindi" / "questions_hi.jsonl"
OUT = REPO_ROOT / "data" / "reports" / "hindi"

TO_HINDI = """Translate the following question into natural, everyday Hindi (Devanagari script), the way an
ordinary person in India would type it. Keep names, amounts and dates. Do not add section numbers or legal
terms that are not in the original. Reply with the Hindi translation only."""


def safe(spec: str) -> str:
    return re.sub(r"[^\w.-]+", "_", spec)


def client(spec: str):
    from app.core.openai_compat import OpenAICompatClient, parse_spec

    provider, model = parse_spec(spec) if spec.split(":")[0] != "groq" else ("groq", spec.split(":", 1)[1])
    if provider == "groq":
        from app.core.llm_client import GroqClient

        return GroqClient(get_settings().groq_api_key, model, max_tokens=2048)
    return OpenAICompatClient(provider, model, max_tokens=2048)


def patiently(fn, *a):
    """Free tiers: wait out per-minute limits, and daily ones for as long as the provider says
    (QuotaWaiter: up to 6 hours of waiting in all)."""
    from app.benchmark.quota import QuotaWaiter
    from app.core.llm_client import LLMClientError, LLMRateLimitError

    global _WAITER
    _WAITER = globals().get("_WAITER") or QuotaWaiter(True, 6.0)
    attempt = 0
    while True:
        try:
            return fn(*a)
        except (LLMRateLimitError, LLMClientError) as e:
            if "daily" in str(e).lower():
                if not _WAITER.wait(e):
                    raise
                continue
            attempt += 1
            if attempt == 8:
                raise
            time.sleep(15 * attempt)


def build_dataset(spec: str) -> list[dict]:
    """questions_hi.jsonl, translating only what is not there yet."""
    from app.core.language import detect
    from app.core.models import Message
    from run_baselines import sample

    held = sample(7, 40, 20, 60, 30)
    items = [{"id": it["id"], "kind": "scenario" if it["id"].startswith("sc-") else "oos", "question_en": it["question"],
              "acceptable": it.get("acceptable", [])} for it in held if it["id"].startswith(("sc-", "oos-"))]
    have = {}
    if DATA.exists():
        have = {r["id"]: r for r in map(json.loads, DATA.read_text(encoding="utf-8").splitlines()) if r.get("question_hi")}
    llm = client(spec)
    ask = getattr(llm, "generate_brief", llm.generate)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for it in items:
        if it["id"] in have:
            out.append(have[it["id"]])
            continue
        hi = ""
        for _ in range(3):
            hi = re.sub(r"<think>.*?</think>", "", patiently(ask, [Message("system", TO_HINDI), Message("user", it["question_en"])]) or "",
                        flags=re.DOTALL).strip().strip('"')
            if detect(hi) == "hi":
                break
        out.append({**it, "question_hi": hi if detect(hi) == "hi" else "", "translated_by": llm.model_id,
                    "checked_by_native_speaker": False})
        DATA.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
        print(f"  translated {len(out)}/{len(items)}", flush=True)
    DATA.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    return [r for r in out if r["question_hi"]]


class CachedTranslator:
    """Hindi -> English as the chat does it (language.Translator's prompt), cached on disk."""

    def __init__(self, spec: str) -> None:
        self.llm = client(spec)
        self.path = OUT / f"translations_{safe(spec)}.json"
        self.cache = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def __call__(self, question: str) -> str | None:
        from app.core.language import Translator

        if question not in self.cache:
            self.cache[question] = patiently(Translator().to_english, question, self.llm, True)
            OUT.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache, indent=0, ensure_ascii=False), encoding="utf-8")
        return self.cache[question]


def retrieval(retriever, items, rewriter, translator) -> dict:
    conditions = {
        "en": lambda q: (q["question_en"], rewriter(q["question_en"])),
        "hi-raw": lambda q: (q["question_hi"], None),
        "hi-rewrite": lambda q: (q["question_hi"], rewriter(q["question_hi"])),
        "hi-translate": lambda q: ((t := translator(q["question_hi"]) or q["question_hi"]), rewriter(t)),
    }
    out = {}
    for name, make in conditions.items():
        rows = []
        for q in items:
            text, rw = make(q)
            secs = [s.section for s in retriever.retrieve_with_diagnostics(text, rewrite=rw).sources]
            rows.append({"id": q["id"], "kind": q["kind"], "sections": secs,
                         "hit": bool(set(secs) & set(q["acceptable"])), "answered": bool(secs)})
        sc = [r for r in rows if r["kind"] == "scenario"]
        oos = [r for r in rows if r["kind"] == "oos"]
        out[name] = {"scenarios": len(sc), "hit": sum(r["hit"] for r in sc), "refused": sum(not r["answered"] for r in sc),
                     "oos": len(oos), "oos_answered": sum(r["answered"] for r in oos), "rows": rows}
        print(f"  {name:13} hit {out[name]['hit']}/{len(sc)}  refused {out[name]['refused']}/{len(sc)}  "
              f"out-of-scope answered {out[name]['oos_answered']}/{len(oos)}", flush=True)
    return out


def answers(items, spec: str) -> dict:
    from app.core.chat_router import ChatRequest
    from app.core.language import devanagari_share
    from app.main import build_chat_router

    chat = build_chat_router(get_settings())
    name = next((k for k, m in chat.models.items() if m.model_id == spec.split(":", 1)[-1] or k == spec), None)
    if name is None:
        raise SystemExit(f"model {spec} is not configured in the app (have: {', '.join(chat.models)})")
    path = OUT / f"answers_{safe(spec)}.jsonl"
    done = {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())} if path.exists() else {}
    rows = []
    for q in items:
        if q["id"] in done and not done[q["id"]].get("error"):
            rows.append(done[q["id"]])
            continue
        ev: dict = {"answer": ""}
        for _ in range(4):
            ev = {"answer": ""}
            for block in chat.events(ChatRequest(question=q["question_hi"], model=name)):
                head, data = block.split("\n", 1)
                kind, data = head[len("event: "):], json.loads(data[len("data: "):].strip())
                if kind == "token":
                    ev["answer"] += data["text"]
                else:
                    ev[kind] = data
            if "error" not in ev:
                break
            time.sleep(30)
        markers = {s["marker"]: s["section"] for s in ev.get("sources", []) if s.get("kind") != "judgment"}
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", ev["answer"])}
        all_markers = {s["marker"] for s in ev.get("sources", [])}
        row = {"id": q["id"], "kind": q["kind"], "refused": bool(ev.get("meta", {}).get("refused")),
               "hindi": devanagari_share(ev["answer"]) >= 0.5, "devanagari": round(devanagari_share(ev["answer"]), 2),
               "cites_acceptable": any(markers.get(n) in q["acceptable"] for n in cited),
               "bad_markers": sorted(cited - all_markers), "translation": ev.get("retrieval", {}).get("translation"),
               "answer": ev["answer"], "error": ev.get("error", {}).get("message") if "error" in ev else None}
        rows.append(row)
        done[q["id"]] = row
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in done.values()) + "\n", encoding="utf-8")
        print(f"  {q['id']:10} {'REFUSED' if row['refused'] else 'answered'}  hindi={row['devanagari']:.2f}  "
              f"cites-acceptable={row['cites_acceptable']}{'  ERROR ' + row['error'][:60] if row['error'] else ''}", flush=True)
    sc = [r for r in rows if r["kind"] == "scenario" and not r["error"]]
    oos = [r for r in rows if r["kind"] == "oos" and not r["error"]]
    return {"scenarios": len(sc), "answered": sum(not r["refused"] for r in sc),
            "in_hindi": sum(r["hindi"] for r in sc if not r["refused"]),
            "cites_acceptable": sum(r["cites_acceptable"] for r in sc),
            "bad_markers": sum(bool(r["bad_markers"]) for r in rows),
            "oos": len(oos), "oos_refused": sum(r["refused"] for r in oos),
            "refusals_in_hindi": sum(r["hindi"] for r in rows if r["refused"] and not r["error"]),
            "refused_total": sum(r["refused"] for r in rows if not r["error"]),
            "errors": sum(bool(r["error"]) for r in rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--translate-model", default="gemini:gemini-3.8-flash", help="English -> Hindi, for the data set")
    ap.add_argument("--model", default="groq:openai/gpt-oss-120b", help="the system under test: translation, rewrite, answer")
    ap.add_argument("--answers", action="store_true", help="also run the chat end to end")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from app.core.retriever import build_retriever
    from eval_scenario_bench import CachedRewriter

    items = build_dataset(args.translate_model)[: args.limit]
    print(f"{len(items)} questions in Hindi ({sum(q['kind'] == 'scenario' for q in items)} scenarios)")
    retriever = build_retriever(get_settings())
    summary = {"data": str(DATA.relative_to(REPO_ROOT)), "model": args.model, "translate_model": args.translate_model,
               "retrieval": retrieval(retriever, items, CachedRewriter(args.model), CachedTranslator(args.model))}
    if args.answers:
        summary["answers"] = answers(items, args.model)
        print(json.dumps(summary["answers"], indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    slim = {**summary, "retrieval": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in summary["retrieval"].items()}}
    (OUT / "summary.json").write_text(json.dumps(slim, indent=1), encoding="utf-8")
    (OUT / "retrieval_rows.json").write_text(json.dumps({k: v["rows"] for k, v in summary["retrieval"].items()}, indent=0,
                                                        ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
