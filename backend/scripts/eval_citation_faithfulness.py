"""Citation faithfulness: does each cited judgment passage support the sentence that cites it?

1. Answers: the chat (--model, the app default) answers the benchmark's held-out scenarios and
   punishment questions (run_baselines.py, seed 7). Cached in data/reports/faithfulness/.
2. Claims: every answer sentence that cites a JUDGMENT source [n] is paired with that passage. As a
   control, the same number of sentences citing an IPC section are paired with the section text.
   Deterministic check per judgment claim: does the sentence name the case it cites (party names
   found in the source's title), or another case?
3. Judges: two free models from different families from the one that answered (--judges), each
   labelling every pair blind to the other:
     SUPPORTED      the passage states or directly implies everything the sentence attributes to it
     PARTIAL        part of it, or the sentence overstates / generalises what the passage says
     UNSUPPORTED    the passage does not say it, is about something else, or contradicts it
   Reported: each judge's labels, the share both call SUPPORTED, agreement and Cohen's kappa
   (3-way and supported-vs-not), for judgments and the statute control separately.

Usage:  python backend/scripts/eval_citation_faithfulness.py [--limit N]
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import REPO_ROOT, get_settings  # noqa: E402

OUT = REPO_ROOT / "data" / "reports" / "faithfulness"
LABELS = ("SUPPORTED", "PARTIAL", "UNSUPPORTED")

JUDGE_PROMPT = """You check citations in answers about Indian criminal law. You get one SENTENCE from an answer and
the SOURCE passage it cites. Decide whether the source supports what the sentence attributes to it.

- SUPPORTED: the source states, or directly and obviously implies, everything the sentence attributes to it.
- PARTIAL: the source supports part of it, or the sentence overstates or generalises what the source says.
- UNSUPPORTED: the source does not say it, is about something else, or contradicts it.

Judge only the content attributed to the cited source; ignore claims the sentence attributes to other
[numbers]. Do not use outside knowledge about whether the claim is true in law.
Reply with JSON only: {"label": "SUPPORTED" | "PARTIAL" | "UNSUPPORTED", "reason": "<one short sentence>"}"""

def safe(spec: str) -> str:
    return re.sub(r"[^\w.-]+", "_", spec)


def client(spec: str):
    from app.core.openai_compat import OpenAICompatClient, parse_spec

    provider, model = ("groq", spec.split(":", 1)[1]) if spec.startswith("groq:") else parse_spec(spec)
    if provider == "groq":
        from app.core.llm_client import GroqClient

        return GroqClient(get_settings().groq_api_key, model, max_tokens=1024, temperature=0.0)
    return OpenAICompatClient(provider, model, max_tokens=1024, temperature=0.0)


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


def questions() -> list[dict]:
    from run_baselines import sample

    return [{"id": it["id"], "question": it["question"]} for it in sample(7, 40, 20, 60, 30)
            if it["id"].startswith(("sc-", "pun-"))]


def get_answers(items, model: str, cached_only: bool = False) -> list[dict]:
    from app.core.chat_router import ChatRequest
    from app.main import build_chat_router

    path = OUT / f"answers_{safe(model)}.jsonl"
    done = {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())} if path.exists() else {}
    todo = [] if cached_only else [q for q in items if q["id"] not in done]
    if todo:
        chat = build_chat_router(get_settings())
        name = next((k for k, m in chat.models.items() if k == model or f"groq:{m.model_id}" == model), None)
        if name is None:
            raise SystemExit(f"model {model} is not configured (have: {', '.join(chat.models)})")
        OUT.mkdir(parents=True, exist_ok=True)
        for i, q in enumerate(todo, 1):
            ev: dict = {}
            for attempt in range(5):
                ev = {"answer": ""}
                for block in chat.events(ChatRequest(question=q["question"], model=name)):
                    head, data = block.split("\n", 1)
                    kind, data = head[len("event: "):], json.loads(data[len("data: "):].strip())
                    if kind == "token":
                        ev["answer"] += data["text"]
                    elif kind in ("sources", "meta", "error"):
                        ev[kind] = data
                if "error" not in ev:
                    break
                if "daily" in ev["error"]["message"].lower():
                    from app.benchmark.quota import QuotaWaiter

                    QuotaWaiter(True, 1.0).wait(RuntimeError(ev["error"]["message"]))
                else:
                    time.sleep(20 * (attempt + 1))
            done[q["id"]] = {**q, "answer": ev["answer"], "sources": ev.get("sources", []),
                             "refused": bool(ev.get("meta", {}).get("refused")), "error": ev.get("error")}
            path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in done.values()) + "\n", encoding="utf-8")
            n_j = sum(s.get("kind") == "judgment" for s in done[q["id"]]["sources"])
            print(f"  answer {i}/{len(todo)} {q['id']:9} judgments={n_j}{' ERROR' if ev.get('error') else ''}", flush=True)
    return [done[q["id"]] for q in items if q["id"] in done and not done[q["id"]].get("error")]


def claims(answers: list[dict]) -> list[dict]:
    out = []
    from types import SimpleNamespace

    from app.explainability.case_names import check_case_names, party_words, sentences

    for a in answers:
        by_marker = {s["marker"]: s for s in a["sources"]}
        flagged = check_case_names(a["answer"], [SimpleNamespace(**s) for s in a["sources"]])
        for sent in sentences(a["answer"]):
            wrong = {f.cited_marker for f in flagged if f.sentence == sent[:400]}
            sent = sent.strip(" -*")
            for n in dict.fromkeys(int(m) for m in re.findall(r"\[(\d+)\]", sent)):
                src = by_marker.get(n)
                if not src or len(sent) < 25:
                    continue
                c = {"qid": a["id"], "sentence": sent[:800], "marker": n, "kind": src.get("kind", "statute"),
                     "source": src["text"][:2500], "title": src.get("title", ""), "citation": src.get("citation_path", "")}
                if c["kind"] == "judgment":
                    words = party_words(src.get("title", ""))
                    c["names_its_case"] = any(re.search(rf"\b{w}\b", sent.lower()) for w in words) if words else None
                    c["names_another_case"] = n in wrong  # the app's own check (case_names.py)
                out.append(c)
    return out


def judge(c: dict, llm) -> dict:
    from app.core.models import Message

    user = (f"SENTENCE (cites [{c['marker']}]):\n{c['sentence']}\n\nSOURCE [{c['marker']}]"
            f"{' - judgment: ' + c['title'] if c['kind'] == 'judgment' else ' - ' + c['citation']}:\n{c['source']}")
    ask = getattr(llm, "generate_brief", llm.generate)
    for _ in range(3):
        reply = re.sub(r"<think>.*?</think>", "", patiently(ask, [Message("system", JUDGE_PROMPT), Message("user", user)]) or "",
                       flags=re.DOTALL)
        m = re.search(r"\{.*\}", reply, re.DOTALL)
        try:
            data = json.loads(m.group()) if m else {}
        except json.JSONDecodeError:
            data = {}
        if str(data.get("label", "")).upper() in LABELS:
            return {"label": data["label"].upper(), "reason": str(data.get("reason", ""))[:300]}
    return {"label": None, "reason": "unreadable reply"}


def kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def summarise(rows: list[dict], judges: list[str]) -> dict:
    j1, j2 = judges
    both = [r for r in rows if r["judges"][j1]["label"] and r["judges"][j2]["label"]]
    a = [r["judges"][j1]["label"] for r in both]
    b = [r["judges"][j2]["label"] for r in both]
    yes = lambda xs: ["SUPPORTED" if x == "SUPPORTED" else "NOT" for x in xs]  # noqa: E731
    out = {"claims": len(rows), "judged_by_both": len(both),
           "labels": {j: dict(Counter(r["judges"][j]["label"] for r in both)) for j in judges},
           "both_supported": sum(x == y == "SUPPORTED" for x, y in zip(a, b)),
           "both_unsupported": sum(x == y == "UNSUPPORTED" for x, y in zip(a, b)),
           "agreement": round(sum(x == y for x, y in zip(a, b)) / len(both), 3) if both else None,
           "kappa_3way": kappa(a, b), "kappa_binary": kappa(yes(a), yes(b))}
    if rows and rows[0]["kind"] == "judgment":
        out["names_its_case"] = sum(bool(r.get("names_its_case")) for r in rows)
        out["names_another_case"] = sum(bool(r.get("names_another_case")) for r in rows)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="groq", help="the chat model whose answers are checked (an app model id)")
    ap.add_argument("--judges", nargs=2, default=["gemini:gemini-3.5-flash-lite", "groq:qwen/qwen3.8-27b"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cached-answers", action="store_true", help="judge only the answers already generated")
    args = ap.parse_args()

    answers = get_answers(questions()[: args.limit], args.model, args.cached_answers)
    print(f"{len(answers)} answers; {sum(not a['refused'] for a in answers)} not refused; "
          f"{sum(any(s.get('kind') == 'judgment' for s in a['sources']) for a in answers)} with judgment sources")
    all_claims = claims(answers)
    jc = [c for c in all_claims if c["kind"] == "judgment"]
    sc = [c for c in all_claims if c["kind"] != "judgment"]
    sc = sc[:: max(1, len(sc) // max(1, len(jc)))][: len(jc)]  # an evenly spread control of equal size
    print(f"{len(jc)} judgment claims, {len(sc)} statute claims (control)")
    cache_path = OUT / "judgements_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    judges = {spec: client(spec) for spec in args.judges}
    rows = []
    for i, c in enumerate(jc + sc, 1):
        r = {**c, "judges": {}}
        for spec, llm in judges.items():
            key = f"{spec}|{c['marker']}|{c['sentence']}|{c['source'][:200]}"
            if key not in cache:
                cache[key] = judge(c, llm)
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
            r["judges"][spec] = cache[key]
        rows.append(r)
        print(f"  {i}/{len(jc) + len(sc)} {c['kind']:9} " + " ".join(str(v['label']) for v in r["judges"].values()), flush=True)
    summary = {"model": args.model, "judges": args.judges, "answers": len(answers),
               "judgment": summarise([r for r in rows if r["kind"] == "judgment"], args.judges),
               "statute_control": summarise([r for r in rows if r["kind"] != "judgment"], args.judges)}
    (OUT / "claims.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
