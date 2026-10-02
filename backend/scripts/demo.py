"""End-to-end demo against a running LegalGPT backend: retrieve -> generate -> explain -> (optionally) verify.

Sends a fixed set of questions to POST /api/chat, reads the SSE stream, and prints for each one the
answer, the sources it cited, the evidence-match band, sentence support, warnings and — where a
section is given — the rule-engine verdict on the facts in the question. Standard library only.

Usage (from the repo root, with the stack up: `docker compose up` or uvicorn on :8000):
    python backend/scripts/demo.py
    python backend/scripts/demo.py --model adapter          # the LoRA-tuned model (needs a trained adapter)
    python backend/scripts/demo.py --no-verify --only 1,4   # skip the Prolog check; run questions 1 and 4

The scenarios are illustrative facts written for this demo, not real cases.
"""

import argparse
import json
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Scenario:
    question: str
    verify_section: str | None = None


SCENARIOS = [
    Scenario("The servant of my neighbour took my bicycle from my courtyard at night without asking anyone and "
             "sold it in the market the next day. Is this theft, and what is the punishment?", "379"),
    Scenario("A lorry driver overtook rashly on the highway and ran over a motorcyclist, who died instantly. He did "
             "not know the victim and had no intention to harm him. What is the punishment for causing death by "
             "negligence?", "304A"),
    Scenario("A woman died of burns in her matrimonial home four years after her marriage. Her husband and "
             "mother-in-law had repeatedly beaten her and demanded a car as dowry, the last time a week before "
             "she died. Which offence is this and what is the punishment?", "304B"),
    Scenario("When does the right of private defence of the body extend to causing death?"),
    Scenario("How do I file my income tax return online?"),  # outside the IPC: Legal mode should refuse
]


@dataclass
class Outcome:
    model: str = ""
    refused: bool = False
    answer: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    explanation: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    error: str = ""
    seconds: float = 0.0


def ask(base_url: str, scenario: Scenario, model: str | None, verify: bool, timeout: float) -> Outcome:
    body = {"question": scenario.question, "mode": "legal", "history": []}
    if model:
        body["model"] = model
    if verify and scenario.verify_section:
        body["verify_section"] = scenario.verify_section
    req = urllib.request.Request(f"{base_url}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out, t0, event = Outcome(), time.perf_counter(), ""
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "meta":
                    out.model, out.refused = data.get("model", ""), bool(data.get("refused"))
                elif event == "sources":
                    out.sources = data
                elif event == "token":
                    out.answer += data["text"]
                elif event == "explanation":
                    out.explanation = data
                elif event == "verification":
                    out.verification = data
                elif event == "error":
                    out.error = data.get("message", str(data))
    out.seconds = time.perf_counter() - t0
    return out


def wrap(text: str, indent: str = "    ") -> str:
    return "\n".join(textwrap.fill(p, 100, initial_indent=indent, subsequent_indent=indent)
                     for p in text.strip().splitlines() if p.strip())


def report(i: int, scenario: Scenario, o: Outcome) -> None:
    print(f"\n{'=' * 100}\n[{i}] {scenario.question}")
    print(f"    model: {o.model or '?'}   time: {o.seconds:.1f}s" + ("   (refused: no source cleared the floor)" if o.refused else ""))
    if o.error:
        print(f"    ERROR: {o.error}")
    print(f"\n  Answer:\n{wrap(o.answer) or '    (empty)'}")
    ex = o.explanation
    if ex:
        cited = {s["marker"] for s in ex.get("sources", []) if s.get("cited")}
        print("\n  Sources:" + ("" if ex.get("sources") else " none"))
        for s in ex.get("sources", []):
            print(f"    [{s['marker']}]{'*' if s['marker'] in cited else ' '} {s['citation_path']} — {s['title']} (sim {s['similarity']:.2f})")
        if ex.get("sources"):
            print("    (* = cited in the answer)")
        cc = ex.get("citation_check") or {}
        rel = ex.get("relevance")
        print(f"\n  Why: evidence match {ex.get('band') or '-'}" + (f" ({rel:.2f})" if rel is not None else "")
              + (f"; sentence support {ex['support_ratio']:.0%}" if ex.get("support_ratio") is not None else "")
              + f"; citations resolved {cc.get('resolved', [])}, unresolved {cc.get('unresolved', [])}")
        for w in ex.get("warnings", []):
            print(f"    ! {w}")
    v = o.verification
    if v:
        if "error" in v:
            print(f"\n  Verification against s.{scenario.verify_section}: unavailable — {v['error']}")
        else:
            print(f"\n  Verification against s.{v['cited_section']} ({v['title']}): {v['status']}")
            mark = {"satisfied": "+", "violated": "x", "missing": "?"}
            for el in v["elements"]:
                ev = f' — "{el["evidence"]}"' if el.get("evidence") else ""
                print(f"    {mark.get(el['status'], ' ')} {el['description']}: {el['found']}{ev}")
            fits = [a["section"] for a in v.get("alternatives", []) if a["status"] == "CONSISTENT"]
            if fits:
                print(f"    facts also satisfy: {', '.join('s.' + s for s in fits)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default=None, help="groq (base) or adapter (LoRA-tuned); default: the server's default")
    ap.add_argument("--no-verify", action="store_true", help="skip the Prolog fact check")
    ap.add_argument("--only", default="", help="comma-separated question numbers to run (1-based)")
    ap.add_argument("--timeout", type=float, default=600, help="seconds per question (the CPU adapter is slow)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    base = args.base_url.rstrip("/")

    try:
        with urllib.request.urlopen(f"{base}/api/models", timeout=10) as r:
            models = json.load(r)
    except (urllib.error.URLError, OSError) as e:
        print(f"cannot reach the backend at {base}: {e}\nstart it with `docker compose up` (or uvicorn) first", file=sys.stderr)
        return 2
    print("models: " + ", ".join(f"{m['id']}={m['model_id'] or '-'}{' (default)' if m['default'] else ''}"
                                 f"{'' if m['available'] else ' (unavailable)'}" for m in models))
    if args.model and not any(m["id"] == args.model and m["available"] for m in models):
        print(f"model {args.model!r} is not available on this server", file=sys.stderr)
        return 2

    picked = {int(x) for x in args.only.split(",") if x.strip()}
    rows, failed = [], 0
    for i, sc in enumerate(SCENARIOS, 1):
        if picked and i not in picked:
            continue
        try:
            o = ask(base, sc, args.model, not args.no_verify, args.timeout)
        except (urllib.error.URLError, OSError) as e:
            o = Outcome(error=str(e))
        failed += bool(o.error)
        report(i, sc, o)
        ex = o.explanation or {}
        verdict = (o.verification or {}).get("status") or ("n/a" if not o.verification else "error")
        rows.append((i, o.model.split("/")[-1][:34], "refused" if o.refused else (ex.get("band") or "-"),
                     f"{ex['support_ratio']:.0%}" if ex.get("support_ratio") is not None else "-",
                     f"s.{sc.verify_section} {verdict}" if o.verification else "-", f"{o.seconds:.1f}s",
                     "ERROR" if o.error else "ok"))

    print(f"\n{'=' * 100}\nSummary")
    header = ("#", "model", "evidence", "support", "verification", "time", "status")
    widths = [max(len(str(r[c])) for r in [header, *rows]) for c in range(len(header))]
    for r in [header, *rows]:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))
    print("\nEvidence match describes how well the retrieved IPC text matches the question, not whether the answer is "
          "correct.\nVerification checks the facts stated in the question against encoded elements of the section; "
          "it is not legal advice.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
