"""Retrieval ablation on the benchmark's held-out scenarios and out-of-scope questions.

Rows add one component at a time; each is scored on the 60 held-out scenarios (an acceptable section
among the sources) and the 30 held-out out-of-scope questions (sources returned = would be answered):

  embedding only        MiniLM search, fixed 0.55 floor
  + lexicon (20 rules)  the original lay-term rules (read from git, commit 8f86355)
  + scope check         the classifier that lowers the floor for crime questions / blocks civil ones
  + lexicon (40 rules)  the expanded rules (current)
  + LLM rewrite         with --rewrite MODEL (cached rewrites, eval_scenario_bench.CachedRewriter)

Usage (from the repo root):
    python backend/scripts/ablation_retrieval.py [--rewrite gemini:gemini-3.5-flash-lite]
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import app.core.retriever as retriever_module  # noqa: E402
from app.config import REPO_ROOT, get_settings  # noqa: E402
from app.core import lay_terms  # noqa: E402
from eval_scenario_bench import CachedRewriter, evaluate, splits  # noqa: E402

ORIGINAL_LEXICON_COMMIT = "8f86355"


def original_rules() -> tuple:
    src = subprocess.run(["git", "show", f"{ORIGINAL_LEXICON_COMMIT}:backend/app/core/lay_terms.py"], cwd=REPO_ROOT,
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    ns: dict = {}
    exec(compile(src, "lay_terms_original", "exec"), ns)  # our own file from git history
    return ns["LAY_TERMS"]


def use_rules(rules) -> None:
    compiled = tuple((re.compile(p, re.IGNORECASE), legal) for p, legal in rules)
    fn = (lambda q: [legal for pattern, legal in compiled if pattern.search(q)]) if rules else (lambda q: [])
    lay_terms.statutory_terms = fn
    retriever_module.statutory_terms = fn  # the retriever imported the name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rewrite", help="also score the LLM rewrite with this model's cached rewrites")
    args = ap.parse_args()
    from app.core.retriever import build_retriever

    data = splits()
    current = lay_terms.LAY_TERMS
    rows = [("embedding only", (), False, None), ("+ lexicon (20 rules)", original_rules(), False, None),
            ("+ scope check", original_rules(), True, None), (f"+ lexicon ({len(current)} rules)", current, True, None)]
    if args.rewrite:
        rows.append((f"+ LLM rewrite ({args.rewrite}, gate 0.5)", current, True, CachedRewriter(args.rewrite, patient=True)))
    print("| retrieval | scenarios: acceptable section | wrong only | refused | out-of-scope answered |")
    print("|---|---|---|---|---|")
    for name, rules, scope, rewriter in rows:
        use_rules(rules)
        r = build_retriever(get_settings(), scope=scope)
        sc = evaluate(r, data["final"], rewriter=rewriter)
        oos = evaluate(r, data["oos-final"], oos=True, rewriter=rewriter)
        print(f"| {name} | {sc['hit']:.0%} | {sc['wrong']:.0%} | {sc['refused']:.0%} | {oos['answered']:.0%} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
