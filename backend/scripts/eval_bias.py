"""Bias evaluation: does the answer change when only the gender of the people changes?

benchmark/bias/pairs.jsonl: 40 scenarios with gendered people - 24 gender-sensitive (dowry, cruelty,
sexual offences, stalking...) and 16 controls where gender should not matter (theft, cheating, hurt...).
Each is swapped automatically (app/explainability/bias.py: husband <-> wife, he <-> she, names) and run
three times: original, swapped, and the original again (control, for run-to-run variation), by
  - the model alone (closed book: the sections it names), and
  - LegalGPT (the full pipeline: retrieval, rewrite, answer; the sections it relies on).
Each pair gets the bias check's verdict: no difference / difference required by law (gender-specific
sections) / run-to-run variation / possible bias (an unexplained difference, a flipped refusal, or
stereotyped wording).

--attribute caste|income|region|age: the first 10 control scenarios instead, where none of these should
matter. They name no caste, income, region or age, so one sentence about the accused is added to two
otherwise identical copies (bias.counterfactual). Both copies run twice, so a difference counts only if
it holds in both runs of one copy and neither run of the other; possible_bias_one_control keeps the
figure under the gender runs' rule (one control), for comparison. A definition and its punishment section
count as one offence (bias.same_offence); possible_bias_separate is the figure without that.

Second runs can only clear a difference, so they are made only when the first two runs differ.

--run: "" is without app/core/background.py; "fixed" is with it; "heldout" and "heldout-off" use the
wordings of benchmark/bias/held_out_wordings.json, which the filter was not built around, with and without it.

Usage (repo root): python backend/scripts/eval_bias.py --model groq:openai/gpt-oss-120b [--attribute caste]
Outputs are cached per run in data/reports/bias/ (rerun to resume); the summary goes to stdout and
data/reports/bias/summary_<model>.json (summary_<attribute>_<model>.json for an --attribute run).
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
OUT = ROOT / "data" / "reports" / "bias"
PAIRS = ROOT / "benchmark" / "bias" / "pairs.jsonl"
HELD_OUT = ROOT / "benchmark" / "bias" / "held_out_wordings.json"
SECTION = re.compile(r"\b(?:section|sec\.|s\.)\s*(\d{1,3}[A-Z]{0,2})\b", re.I)


OTHER_ACT = re.compile(r"(?i)crpc|criminal procedure|dowry prohibition|domestic violence|evidence act|bnss|"
                       r"information technology|\bIT Act|pocso|protection of women|hindu marriage|special marriage|"
                       r"\bD\.?P\.? Act|\bPWDVA|nagarik suraksha|sakshya")
BNS_WORD = re.compile(r"(?i)\bBNS\b|nyaya sanhita")


def offence_sections(text: str, valid: set[str]) -> list[str]:
    """IPC offence sections an answer names: sentences about other Acts are skipped, BNS numbers are
    mapped to their IPC equivalents (official table), and only IPC offences (s.107 on) count."""
    from app.core.bns import ipc_for

    out = []
    for sent in re.split(r"(?<=[.!?;:\n])\s+|\n", text or ""):
        if OTHER_ACT.search(sent) and not re.search(r"(?i)\bIPC\b|penal code", sent):
            continue
        for sec in (m.upper() for m in SECTION.findall(sent)):
            if BNS_WORD.search(sent) and not re.search(r"(?i)\bIPC\b|penal code", sent):
                out += [s for s in ipc_for(sec) if s in valid]
            elif sec in valid and int(re.match(r"\d+", sec).group()) >= 107:
                out.append(sec)
    return list(dict.fromkeys(out))


def whole(ask, text_of, tries: int = 6):
    """An answer that stopped mid-way (a stream can end early with no error) must not be compared: a
    missing section would read as a difference between the two versions. Asked again until it is whole."""
    from app.core.citations import looks_cut_off

    for attempt in range(tries):
        got = ask()
        if not looks_cut_off(text_of(got)):
            return got
        time.sleep(10 * (attempt + 1))
    raise RuntimeError("the answer kept stopping mid-way; try again later")


def safe(spec: str) -> str:
    return re.sub(r"[^\w.-]+", "_", spec)


def patiently(fn, *a):
    from app.benchmark.quota import QuotaWaiter
    from app.core.llm_client import LLMClientError, LLMRateLimitError

    global _WAITER
    _WAITER = globals().get("_WAITER") or QuotaWaiter(True, 8.0)
    attempt = 0
    while True:
        try:
            return fn(*a)
        except (LLMRateLimitError, LLMClientError) as e:
            if "daily" in str(e).lower() or "per day" in str(e).lower():
                if not _WAITER.wait(e):
                    raise
                continue
            attempt += 1
            if attempt == 8:
                raise
            time.sleep(15 * attempt)


class Cache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.rows[(r["id"], r["variant"])] = r

    def get(self, key, make):
        if key not in self.rows:
            row = {"id": key[0], "variant": key[1], **make()}
            self.rows[key] = row
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return self.rows[key]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="groq:openai/gpt-oss-120b", help="the chat model's id in /api/models, or groq:<model>")
    ap.add_argument("--systems", nargs="*", default=["closed", "legalgpt"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--attribute", default="gender", choices=["gender", "caste", "income", "region", "age"])
    ap.add_argument("--dry-run", action="store_true", help="print the versions that would be compared; no model calls")
    ap.add_argument("--run", default="", choices=["", "fixed", "heldout", "heldout-off"])
    args = ap.parse_args()
    attr = args.attribute
    tag = ("" if attr == "gender" else f"{attr}_")
    if args.run:
        tag = f"{args.run.replace('-', '_')}_{tag}"

    from app.config import get_settings
    from app.core.models import Message
    from app.explainability import bias
    from app.explainability.router import ExplainRouter
    from app.main import build_chat_router
    from run_baselines import CLOSED_SYSTEM

    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(l) for l in PAIRS.read_text(encoding="utf-8").splitlines() if l.strip()]
    if attr != "gender":
        pairs = [p for p in pairs if p["group"] == "neutral"][:10]
    if args.limit:
        pairs = pairs[: args.limit]

    def versions(question: str) -> "bias.Counterfactual":
        if args.run.startswith("heldout"):
            a, b = json.loads(HELD_OUT.read_text(encoding="utf-8"))[attr]
            return bias.Counterfactual(f"{question} {a}", f"{question} {b}", [{"from": a, "to": b}], "added")
        return bias.counterfactual(question, attr)

    if args.dry_run:
        for p in pairs:
            cf = versions(p["question"])
            print(p["id"], cf.mode, [f"{x['from']} -> {x['to']}" for x in cf.swaps], "|", cf.second[:150])
        return 0
    from app.case_analysis.router import ipc_sections

    titles = ipc_sections() or {}
    valid = set(titles)
    offences = bias.same_offence(titles)
    chat = build_chat_router(get_settings())
    chat.ignore_background = args.run in ("fixed", "heldout")
    if chat.rewriter is not None:
        chat.rewriter.budget_s = None  # the results must not depend on how busy the provider was
    key = "groq" if args.model.startswith("groq:") else args.model
    llm = chat.models[key]
    explain = ExplainRouter(chat)
    summary = {}
    for system in args.systems:
        cache = Cache(OUT / f"{system}_{tag}{safe(args.model)}.jsonl")
        verdicts, per = Counter(), []
        for p in pairs:
            cf = versions(p["question"])

            def closed(q):
                text = whole(lambda: patiently(llm.generate, [Message("system", CLOSED_SYSTEM), Message("user", q)]), lambda t: t)
                return {"question": q, "answer": text, "refused": False}

            def legal(q):
                run, _ = whole(lambda: patiently(explain.run_question, q, llm, "auto"), lambda r: "." if r[0].refused else r[0].answer)
                return {"question": q, "answer": run.answer, "refused": run.refused, "retrieved": run.retrieved,
                        "cited_from_markers": run.cited}

            make = closed if system == "closed" else legal
            run = lambda v, q: cache.get((p["id"], v), lambda: make(q))
            # both systems' answers are read the same way: the IPC offence sections they name or cite
            def to_run(r):
                cited = [] if r["refused"] else offence_sections(r["answer"], valid)
                if system == "legalgpt":  # plus the sections of the sources it cites as [n]
                    cited = list(dict.fromkeys([s for s in r.get("cited_from_markers", []) if s in valid and int(re.match(r"\d+", s).group()) >= 107] + cited))
                out = bias.ruled_out(r["answer"])  # named only to say they do not apply
                cited = [s for s in cited if s not in out]
                return bias.Run(r["question"], r["answer"], r["refused"], r.get("retrieved", []), cited)
            first, second = to_run(run("original", cf.first)), to_run(run("swapped", cf.second))
            judge = lambda **kw: bias.compare(attr, first, second, cf.swaps, "ipc", mode=cf.mode, **kw)
            control = repeat = None
            if judge().verdict == "possible_bias":
                control = to_run(run("control", cf.first))
                repeat = to_run(run("repeat", cf.second))
            one_control = judge(control=control)
            separate = judge(control=control, repeat=repeat)
            merged = judge(control=control, repeat=repeat, offences=offences)
            rep = merged
            verdicts[(p["group"], rep.verdict)] += 1
            per.append({"id": p["id"], "group": p["group"], "verdict": rep.verdict, "verdict_one_control": one_control.verdict,
                        "verdict_separate": separate.verdict, "verdict_merged": merged.verdict,
                        "unexplained": [d["section"] for d in rep.differences if d["kind"] == "unexplained"],
                        "law": [d["section"] for d in rep.differences if d["kind"] == "law"],
                        "law_notes": [n["section"] for n in rep.law_notes if n["section"]],
                        "stereotypes": len(rep.stereotypes), "refusal_changed": rep.refusal_changed})
            print(f"{system} {p['id']} {p['group']:9} {rep.verdict:15} unexplained={per[-1]['unexplained']} law={per[-1]['law']}", flush=True)
        groups = Counter(p["group"] for p in pairs)
        summary[system] = {
            "pairs": len(per),
            "by_group": {g: {v: verdicts[(g, v)] for v in ("no_difference", "law_required", "variation", "possible_bias")}
                         for g in groups},
            "possible_bias_rate": round(sum(verdicts[(g, "possible_bias")] for g in groups) / len(per), 3),
            "possible_bias_one_control": sum(r["verdict_one_control"] == "possible_bias" for r in per),
            "possible_bias_separate": sum(r["verdict_separate"] == "possible_bias" for r in per),
            "possible_bias_merged": sum(r["verdict_merged"] == "possible_bias" for r in per),
            "stereotype_sentences": sum(r["stereotypes"] for r in per),
            "gender_specific_in_both": sum(1 for r in per if r["law_notes"]),
            "pairs_detail": per,
        }
    out = OUT / f"summary_{tag}{safe(args.model)}.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for system, s in summary.items():
        print(system, json.dumps({k: v for k, v in s.items() if k != "pairs_detail"}))
    print("wrote", out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
