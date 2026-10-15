"""Paired significance tests for the benchmark comparisons (exact McNemar test).

Every system answers the same questions, so a comparison is paired: only the questions where the
two systems disagree carry information. With b = questions only system A gets right and c = only
system B, the exact McNemar p-value is the two-sided binomial test of b against b + c at 1/2.

Usage (from the repo root, after run_baselines.py run/recite):
    python backend/scripts/paired_tests.py            # prints a Markdown table
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.benchmark.metrics import known_sections, load_jsonl, score_oos, score_scenario, score_statute  # noqa: E402
from app.core.corpus import load_chunks  # noqa: E402
from run_baselines import OUT, get_settings, sample  # noqa: E402

# (label, metric, set, higher is better) - metric names as in run_baselines.score
METRICS = {
    "wrong punishment": ("punishment_wrong", "punishment", False),
    "right section (statute)": ("section_correct", "statute", True),
    "acceptable section (scenario)": ("hit", "scenario", True),
    "civil question answered as an offence": ("criminalised", "oos", False),
}
PAIRS = [  # (system A, system B, metrics)
    ("closed_openai_gpt-5.4-mini", "legalgpt_openai_gpt-5.4-mini_rw", list(METRICS)),
    ("closed_gemini_gemini-3.5-flash-lite", "legalgpt_gemini_gemini-3.5-flash-lite", list(METRICS)),
    ("closed_openai_gpt-oss-120b", "legalgpt_openai_gpt-oss-120b", list(METRICS)),
    ("closed_openai_gpt-oss-120b", "legalgpt_openai_gpt-oss-120b_rw", list(METRICS)),
    ("closed_gemini_gemini-3.5-flash-lite", "legalgpt_gemini_gemini-3.5-flash-lite_rw", list(METRICS)),
    ("legalgpt_openai_gpt-oss-120b", "legalgpt_openai_gpt-oss-120b_rw", ["acceptable section (scenario)"]),
    ("legalgpt_gemini_gemini-3.5-flash-lite", "legalgpt_gemini_gemini-3.5-flash-lite_rw", ["acceptable section (scenario)"]),
    ("legalgpt_openai_gpt-5.4-mini", "legalgpt_openai_gpt-5.4-mini_rw", ["acceptable section (scenario)"]),
    ("legalgpt-lora", "legalgpt-lora_v2", ["acceptable section (scenario)"]),
    ("legalgpt-lora_v2", "legalgpt-lora_v3", list(METRICS)),
    ("legalgpt-lora_v3", "legalgpt-lora_v4", list(METRICS)),
    ("legalgpt-lora_v2", "legalgpt-lora_v4", list(METRICS)),
]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def outcomes(name: str, items: dict, ipc: set[str]) -> dict[str, dict[str, bool]]:
    """metric label -> {question id: outcome} for one system's answers."""
    path = OUT / f"{name}.jsonl"
    if not path.exists():
        return {}
    rows = [r for r in load_jsonl(path) if r["id"] in items and not r["answer"].startswith("ERROR")]
    out: dict[str, dict[str, bool]] = {m: {} for m in METRICS}
    for r in rows:
        it = items[r["id"]]
        if r["set"] == "statute":
            s = score_statute(it, r["answer"], r["extra_cited"], ipc)
            out["right section (statute)"][r["id"]] = s["section_correct"]
            if it["kind"] == "punishment":
                out["wrong punishment"][r["id"]] = s["punishment_wrong"]
        elif r["set"] == "scenario":
            out["acceptable section (scenario)"][r["id"]] = score_scenario(it, r["answer"], r["extra_cited"], ipc)["hit"]
        else:
            out["civil question answered as an offence"][r["id"]] = score_oos(r["answer"], r["extra_cited"])["criminalised"]
    return out


def main() -> int:
    ipc = known_sections(load_chunks(get_settings().corpus_chunks_path))  # incl. state-amendment sections
    items = {it["id"]: it for it in sample(7, 40, 20, 60, 30)}
    print("| A | B | measure | n | A | B | only A | only B | p (exact McNemar) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for a, b, metrics in PAIRS:
        oa, ob = outcomes(a, items, ipc), outcomes(b, items, ipc)
        if not oa or not ob:
            continue
        for m in metrics:
            ids = sorted(set(oa[m]) & set(ob[m]))
            if not ids:
                continue
            xa = [oa[m][i] for i in ids]
            xb = [ob[m][i] for i in ids]
            only_a = sum(x and not y for x, y in zip(xa, xb))
            only_b = sum(y and not x for x, y in zip(xa, xb))
            p = mcnemar_exact(only_a, only_b)
            pct = lambda xs: f"{100 * sum(xs) / len(xs):.0f}%"
            ptxt = "<0.001" if p < 0.001 else f"{p:.3f}"
            print(f"| {a} | {b} | {m} | {len(ids)} | {pct(xa)} | {pct(xb)} | {only_a} | {only_b} | {ptxt} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
