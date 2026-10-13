"""Blind rating of chat answers: hosted model (Groq) vs the local LoRA adapter.

Three steps, so the rater never sees which model wrote which answer:

  generate  answer every question with both models through the app's own retriever and legal
            prompt (no sources retrieved -> the app's fixed refusal, as in the chat)
  sheet     shuffle each question's two answers into A/B and write the rating sheet
            (<out>.sheet.json/.md, with the retrieved sources) and the key (<out>.key.json)
  score     read the ratings filled into the sheet (relevance, groundedness, completeness, 1-5
            each) and unblind them: per-model means and per-question wins -> <out>.report.md

Usage (from the repo root, GPU environment for the adapter):
    .venv-gpu\\Scripts\\python backend/scripts/blind_eval_answers.py generate --questions docs/eval/blind_questions.txt
    .venv-gpu\\Scripts\\python backend/scripts/blind_eval_answers.py sheet
    (fill in "ratings" in data/reports/blind_eval.sheet.json)
    .venv-gpu\\Scripts\\python backend/scripts/blind_eval_answers.py score --rater "name"
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402

CRITERIA = ("relevance", "groundedness", "completeness")
RUBRIC = {
    "relevance": "1 = does not address the question; 5 = answers exactly what was asked",
    "groundedness": "1 = claims not supported by the sources shown; 5 = every legal claim is supported by a cited source",
    "completeness": "1 = misses the main point; 5 = covers what a careful answer from these sources should (offence, "
                    "ingredients and punishment where asked). A correct refusal when the sources do not cover the "
                    "question scores 5 on all three.",
}


def default_out() -> Path:
    return get_settings().lora_dir.parent.parent / "reports" / "blind_eval"


def model_labels(meta: dict) -> dict[str, str]:
    """{answer key: short label}; answer files written before --models existed used groq/adapter."""
    if "models" in meta:
        return meta["models"]
    return {"groq": "Groq", "adapter": "LoRA"}


def generate(args: argparse.Namespace) -> None:
    from app.core.llm_client import AdapterClient, GroqClient, latest_adapter
    from app.core.prompt_builder import NOT_COVERED_REPLY, PromptBuilder
    from app.core.retriever import build_retriever

    s = get_settings()
    questions = [q.strip() for q in args.questions.read_text(encoding="utf-8").splitlines()
                 if q.strip() and not q.startswith("#")]
    retriever = build_retriever(s)  # exactly the app's retrieval
    items = []
    for q in questions:
        sources = retriever.retrieve(q)
        items.append({"question": q, "sources": [{"marker": x.marker, "citation": x.citation_path, "text": x.text}
                                                 for x in sources],
                      "messages": PromptBuilder().build(q, sources, "legal", []) if sources else None, "answers": {}})
    models, labels = {}, {}
    for spec in args.models:
        if spec == "groq":
            models["groq"], labels["groq"] = GroqClient(s.groq_api_key, s.groq_model), f"Groq {s.groq_model}"
        elif spec.startswith("adapter"):
            path = Path(spec.split(":", 1)[1]) if ":" in spec else (args.adapter or s.adapter_path or latest_adapter(s.lora_dir))
            name = f"lora-{Path(path).name}"
            models[name], labels[name] = AdapterClient(path), f"LoRA {Path(path).name}"
        else:
            raise SystemExit(f"unknown model {spec!r}: use groq or adapter[:PATH]")
    if len(models) != 2:
        raise SystemExit("rate exactly two models")
    for name, client in models.items():
        for it in items:
            t = time.perf_counter()
            answer = client.generate(it["messages"]) if it["messages"] else NOT_COVERED_REPLY
            it["answers"][name] = {"text": answer.strip(), "seconds": round(time.perf_counter() - t, 2)}
            print(f"[{name}] {it['question'][:60]} ({it['answers'][name]['seconds']}s)")
    meta = {"models": labels, "questions": len(items)}
    for it in items:
        it.pop("messages")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".answers.json").write_text(json.dumps({"meta": meta, "items": items}, indent=1,
                                                                ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out.with_suffix('.answers.json')}")


def sheet(args: argparse.Namespace) -> None:
    data = json.loads(args.out.with_suffix(".answers.json").read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    rows, key, md = [], [], ["# Blind answer rating sheet", "",
                             "Rate each answer 1-5 on each criterion. You do not know which model wrote A or B.", ""]
    md += [f"- **{c}**: {RUBRIC[c]}" for c in CRITERIA] + [""]
    for n, it in enumerate(data["items"], 1):
        order = list(it["answers"])
        rng.shuffle(order)
        key.append({"id": n, "A": order[0], "B": order[1]})
        rows.append({"id": n, "question": it["question"], "sources": it["sources"],
                     "A": it["answers"][order[0]]["text"], "B": it["answers"][order[1]]["text"],
                     "ratings": {"A": {c: None for c in CRITERIA}, "B": {c: None for c in CRITERIA}}, "note": ""})
        md += [f"## {n}. {it['question']}", ""]
        md += [f"<details><summary>{len(it['sources'])} sources</summary>", ""]
        md += [f"[{x['marker']}] {x['citation']}: {x['text'][:600]}" + "\n" for x in it["sources"]] + ["</details>", ""]
        md += ["**A**", "", rows[-1]["A"], "", "**B**", "", rows[-1]["B"], ""]
    args.out.with_suffix(".sheet.json").write_text(json.dumps({"rubric": RUBRIC, "rows": rows}, indent=1,
                                                              ensure_ascii=False), encoding="utf-8")
    args.out.with_suffix(".sheet.md").write_text("\n".join(md), encoding="utf-8")
    args.out.with_suffix(".key.json").write_text(json.dumps({"meta": data["meta"], "key": key}, indent=1), encoding="utf-8")
    print(f"wrote {args.out.with_suffix('.sheet.json')} (+ .sheet.md) and the key {args.out.with_suffix('.key.json')}")


def score(args: argparse.Namespace) -> None:
    rows = json.loads(args.out.with_suffix(".sheet.json").read_text(encoding="utf-8"))["rows"]
    keyfile = json.loads(args.out.with_suffix(".key.json").read_text(encoding="utf-8"))
    key = {k["id"]: k for k in keyfile["key"]}
    labels = model_labels(keyfile["meta"])
    m1, m2 = list(labels)
    per = {m: {c: [] for c in CRITERIA} for m in (m1, m2)}
    wins = {m1: 0, m2: 0, "tie": 0}
    lines = []
    for r in rows:
        for slot in ("A", "B"):
            if any(r["ratings"][slot][c] is None for c in CRITERIA):
                raise SystemExit(f"question {r['id']} answer {slot} is not fully rated")
        totals = {}
        for slot in ("A", "B"):
            model = key[r["id"]][slot]
            for c in CRITERIA:
                per[model][c].append(r["ratings"][slot][c])
            totals[model] = sum(r["ratings"][slot][c] for c in CRITERIA)
        win = "tie" if totals[m1] == totals[m2] else max(totals, key=totals.get)
        wins[win] += 1
        lines.append(f"| {r['id']} | {r['question'][:70]} | {totals[m1]} | {totals[m2]} | {labels.get(win, win)} |")
    meta = keyfile["meta"]
    names = dict(labels)
    if "models" not in meta:  # older answer files
        names = {"groq": f"Groq {meta['groq_model']}", "adapter": f"LoRA {meta['adapter']}"}
    out = [f"# Blind answer rating: {names[m1]} vs {names[m2]}", "",
           f"Rater: {args.rater}. {len(rows)} questions; each model's answer was shown as A or B in random order "
           "with the retrieved sources, and rated 1-5 per criterion before the key was opened.", "",
           "| criterion | " + " | ".join(names.values()) + " |", "|---|---|---|"]
    for c in CRITERIA:
        out.append(f"| {c} | " + " | ".join(f"{mean(per[m][c]):.2f}" for m in (m1, m2)) + " |")
    overall = {m: mean(v for c in CRITERIA for v in per[m][c]) for m in per}
    out.append(f"| **mean of all three** | **{overall[m1]:.2f}** | **{overall[m2]:.2f}** |")
    out += ["", f"Per question (sum of the three scores, 3-15): {labels[m1]} better on {wins[m1]}, {labels[m2]} better on "
            f"{wins[m2]}, tie on {wins['tie']}.", "", f"| # | question | {labels[m1]} | {labels[m2]} | better |",
            "|---|---|---|---|---|"]
    out += lines
    notes = [f"- {r['id']}: {r['note']}" for r in rows if r.get("note")]
    if notes:
        out += ["", "## Rater notes", ""] + notes
    args.out.with_suffix(".report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:12]))
    print(f"wrote {args.out.with_suffix('.report.md')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["generate", "sheet", "score"])
    ap.add_argument("--questions", type=Path, help="one question per line (generate)")
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA adapter (default: newest under LORA_DIR)")
    ap.add_argument("--models", nargs=2, default=["groq", "adapter"],
                    help="the two models to rate: groq and/or adapter[:PATH] (generate)")
    ap.add_argument("--out", type=Path, default=None, help="path prefix (default data/reports/blind_eval)")
    ap.add_argument("--seed", type=int, default=7, help="A/B shuffle seed (sheet)")
    ap.add_argument("--rater", default="unnamed rater", help="who rated (score)")
    args = ap.parse_args()
    args.out = args.out or default_out()
    {"generate": generate, "sheet": sheet, "score": score}[args.step](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
