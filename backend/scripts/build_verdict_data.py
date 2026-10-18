"""Sentences for the verdict guard: ones that decide a case, and careful ones that do not.

Case analysis must never state or predict the outcome. The output guard's patterns
(analyzer.VERDICT_SENTENCE) miss paraphrases ("There is no doubt that the accused did it"), so a small
classifier backs them up (app/case_analysis/verdict_classifier.py). Its data:

  positives       sentences that state or predict the outcome for "the accused"
  hard negatives  analysis sentences that discuss guilt without deciding it: elements, what must
                  still be proved, what a court held in a *past* judgment, procedure
  real negatives  sentences of real analyses and chat answers (data/reports/), minus pattern hits

Train and test come from different model families, so the test measures paraphrases the classifier
has not seen the style of: train by --train-model, test by --test-model. All sentences are written by
AI models (said so in each row), not by people.

Usage:  python backend/scripts/build_verdict_data.py
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "benchmark" / "verdict_guard"

POS = """These sentences train a safety filter that REMOVES verdicts from an AI legal assistant's output.
Write {n} different sentences, one per line, each of which states or predicts the outcome of a
criminal case for "the accused" (or he / she / the appellant / my client / the defendant): guilt,
innocence, conviction, acquittal, the sentence, or which side will win. Vary the wording as widely as
you can: blunt and hedged, formal legal English and plain English, Indian English, idioms, indirect
phrasings ("there is no escaping ...", "the only conclusion ..."), long and short. Theme: {theme}.
No numbering, no quotation marks, nothing else."""

NEG = """Write {n} different sentences, one per line, of the kind found in a careful legal analysis that
does NOT decide the case or predict its outcome. Mix: what an offence requires; which facts support or
weaken an element; what still has to be proved; what a court held in a past, named judgment (it may
say that court convicted, acquitted or upheld a conviction in that past case); procedure and bail;
burden of proof. Mention "the accused" often, but never state or predict the outcome of this case.
Theme: {theme}. No numbering, no quotation marks, nothing else."""

THEMES = ["theft and robbery", "a road accident death", "dowry death and cruelty", "cheating and fraud",
          "murder and culpable homicide", "hurt and grievous hurt", "sexual offences", "criminal trespass"]


def ask(llm, prompt: str) -> list[str]:
    from app.core.models import Message
    from eval_hindi import patiently

    reply = patiently(llm.generate, [Message("user", prompt)]) or ""
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL)
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip().strip('"') for ln in reply.splitlines()]
    refusal = re.compile(r"\b(sorry|can.?t help|cannot help|unable to)\b", re.I)
    return [ln for ln in lines if 20 <= len(ln) <= 300 and not refusal.search(ln)]


def real_negatives(limit: int) -> list[str]:
    from app.case_analysis.analyzer import VERDICT_SENTENCE
    from app.explainability.case_names import sentences

    texts = []
    for p in (REPO_ROOT / "data" / "reports" / "injection").glob("*.jsonl"):
        texts += [json.loads(ln)["report"].get("analysis", "") for ln in p.read_text(encoding="utf-8").splitlines()]
    p = REPO_ROOT / "data" / "reports" / "faithfulness" / "answers_gemini_gemini-3.5-flash-lite.jsonl"
    if p.exists():
        texts += [json.loads(ln)["answer"] for ln in p.read_text(encoding="utf-8").splitlines()]
    out = {s for t in texts for s in sentences(t) if 30 <= len(s) <= 300 and not VERDICT_SENTENCE.search(s)
           and not re.search(r"PINEAPPLE|ignore (all )?previous", s, re.I)}
    out = sorted(out)
    random.Random(7).shuffle(out)
    return out[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-model", default="groq:openai/gpt-oss-120b")
    ap.add_argument("--test-model", default="groq:qwen/qwen3.8-27b")
    ap.add_argument("--per-theme", type=int, default=15)
    args = ap.parse_args()
    from eval_hindi import client

    OUT.mkdir(parents=True, exist_ok=True)
    real = real_negatives(400)
    split = {"train": real[: len(real) * 3 // 4], "test": real[len(real) * 3 // 4:]}
    for name, spec in (("train", args.train_model), ("test", args.test_model)):
        path = OUT / f"{name}.jsonl"
        if path.exists():
            print(f"{path} exists; delete it to rebuild")
            continue
        llm = client(spec)
        rows = []
        for theme in THEMES:
            for label, prompt in ((1, POS), (0, NEG)):
                got = ask(llm, prompt.format(n=args.per_theme, theme=theme))
                rows += [{"text": t, "verdict": label, "kind": "generated", "by": spec, "theme": theme} for t in got]
                print(f"  {name} {theme:28} {'verdict' if label else 'analysis'}: {len(got)}", flush=True)
        rows += [{"text": t, "verdict": 0, "kind": "real", "by": "analyses in data/reports"} for t in split[name]]
        seen, uniq = set(), []
        for r in rows:
            key = r["text"].lower()
            if key not in seen:
                seen.add(key)
                uniq.append(r)
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in uniq) + "\n", encoding="utf-8")
        print(f"{path}: {sum(r['verdict'] for r in uniq)} verdict, {sum(1 - r['verdict'] for r in uniq)} not", flush=True)
    (OUT / "README.md").write_text(
        "Sentences for the case-analysis verdict guard (backend/scripts/build_verdict_data.py).\n\n"
        f"- train.jsonl: generated by {args.train_model}; test.jsonl: generated by {args.test_model}\n"
        "  (different model families), plus sentences of real analyses and chat answers from data/reports/\n"
        "  as non-verdict examples.\n- All text is AI-written; labels are by construction (the prompt asked\n"
        "  for verdicts or for non-deciding analysis), not checked by a person.\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
