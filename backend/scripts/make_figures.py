"""Paper figures from the benchmark runs (docs/figures/*.png and *.pdf).

  fig1_safety     wrong punishments and civil questions answered as offences: each model alone vs
                  inside LegalGPT (95% Wilson intervals)
  fig2_scenarios  held-out everyday scenarios: model alone, LegalGPT's first version, the final system
  fig3_ablation   retrieval on the held-out scenarios as each component is added
  fig4_size       the local models by size (1.5B / 3B / 7B), once their benchmark runs exist
  fig5_rrl        the sentence-role classifier ablation (InLegalBERT, 5-fold CV), from
                  rrl_ablation_report.py --json-out data/models/rrl_cv/v0.3-ablation-report.json

Colours: the first three categorical slots of the reference palette (validated all-pairs for colour
vision deficiency); every bar carries its value, so identity never rests on colour alone.

Usage (from the repo root):
    python backend/scripts/make_figures.py [--skip-ablation]
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from app.config import REPO_ROOT  # noqa: E402
from paired_tests import items_and_ipc, outcomes  # noqa: E402

OUT = REPO_ROOT / "docs" / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (blue, orange, aqua)
MODELS = [("gpt-5.4-mini", "closed_openai_gpt-5.4-mini", "legalgpt_openai_gpt-5.4-mini", "legalgpt_openai_gpt-5.4-mini_final"),
          ("Gemini 3.5 flash-lite", "closed_gemini_gemini-3.5-flash-lite", "legalgpt_gemini_gemini-3.5-flash-lite",
           "legalgpt_gemini_gemini-3.5-flash-lite_final"),
          ("gpt-oss-120b", "closed_openai_gpt-oss-120b", "legalgpt_openai_gpt-oss-120b", "legalgpt_openai_gpt-oss-120b_final")]


def wilson(k: int, n: int) -> tuple[float, float, float]:
    """(rate, low, high) in percent."""
    if n == 0:
        return (math.nan,) * 3
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def style(ax, title: str, ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped(ax, groups: list[str], series: list[tuple[str, list[tuple[float, float, float]]]], ymax: float = 100) -> None:
    """Grouped bars with thin interval whiskers and the value on every bar."""
    n = len(series)
    width = 0.8 / n
    for j, (label, vals) in enumerate(series):
        xs = [i + (j - (n - 1) / 2) * width for i in range(len(groups))]
        rates = [v[0] for v in vals]
        ax.bar(xs, rates, width * 0.92, color=SERIES[j], label=label, edgecolor=SURFACE, linewidth=1)  # 2px-style gap
        for x, r in zip(xs, rates):  # a 0% bar has no height: mark it on the baseline so the series stays visible
            if r == 0:
                ax.plot([x - width * 0.4, x + width * 0.4], [0.4, 0.4], color=SERIES[j], linewidth=3, solid_capstyle="butt")
        lo = [r - v[1] for r, v in zip(rates, vals)]
        hi = [v[2] - r for r, v in zip(rates, vals)]
        ax.errorbar(xs, rates, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
        for x, (r, _, h) in zip(xs, vals):
            ax.text(x, h + 1.5, f"{r:.0f}%", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))


def rate(o: dict, metric: str) -> tuple[float, float, float]:
    vals = list(o.get(metric, {}).values())
    return wilson(sum(vals), len(vals))


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def fig_safety(items, ipc) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, title in ((axes[0], "wrong punishment", "Wrong punishment stated (39 punishment questions)"),
                              (axes[1], "civil question answered as an offence",
                               "Civil question answered as an IPC offence (30 questions)")):
        alone = [rate(outcomes(m[1], items, ipc), metric) for m in MODELS]
        inside = [rate(outcomes(m[3], items, ipc), metric) for m in MODELS]
        style(ax, title)
        grouped(ax, [m[0] for m in MODELS], [("Model alone", alone), ("Inside LegalGPT", inside)], ymax=110)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Safety: the same models, alone and inside LegalGPT (lower is better)", x=0.01, ha="left",
                 fontsize=12, color=INK, y=1.04)
    save(fig, "fig1_safety")


def fig_scenarios(items, ipc) -> None:
    metric = "acceptable section (scenario)"
    fig, ax = plt.subplots(figsize=(8, 4))
    style(ax, "Held-out everyday scenarios: an acceptable IPC section cited (60 questions; higher is better)")
    grouped(ax, [m[0] for m in MODELS], [
        ("Model alone", [rate(outcomes(m[1], items, ipc), metric) for m in MODELS]),
        ("LegalGPT, first version", [rate(outcomes(m[2], items, ipc), metric) for m in MODELS]),
        ("LegalGPT, final system", [rate(outcomes(m[3], items, ipc), metric) for m in MODELS]),
    ], ymax=110)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    save(fig, "fig2_scenarios")


def ablation_rows() -> list[tuple[str, float]]:
    """Held-out retrieval as each component is added (ablation_retrieval.py, plus the rewrite replays)."""
    import ablation_retrieval as ab
    from app.benchmark.metrics import load_jsonl
    from app.config import get_settings
    from app.core.query_rewriter import Rewrite
    from app.core.retriever import build_retriever
    from app.core.scope import ScopeCheck
    from eval_scenario_bench import evaluate, splits
    from run_baselines import OUT as RUNS

    final = splits()["final"]
    rows = []
    for label, rules, scope in (("Embedding search only", (), False), ("+ lay-term lexicon (20 rules)", ab.original_rules(), False),
                                ("+ scope check", ab.original_rules(), True),
                                ("+ lexicon expanded (41 rules)", ab.lay_terms.LAY_TERMS, True)):
        ab.use_rules(rules)
        rows.append((label, 100 * evaluate(build_retriever(get_settings(), scope=scope), final)["hit"]))
    ab.use_rules(ab.lay_terms.LAY_TERMS)
    old = subprocess.run(["git", "show", "8f86355:backend/app/core/scope_model.json"], cwd=REPO_ROOT, capture_output=True,
                         text=True, check=True).stdout
    old_path = OUT / "_scope_model_before_retraining.json"
    OUT.mkdir(parents=True, exist_ok=True)
    old_path.write_text(old, encoding="utf-8")
    rw = {r["id"]: r.get("rewrite") for r in load_jsonl(RUNS / "legalgpt_gemini_gemini-3.5-flash-lite_rw.jsonl")}
    to_rw = lambda d: Rewrite(d["criminal"], [(o["name"], o["section"]) for o in d["offences"]]) if d else None
    for label, scope_path in (("+ LLM query rewrite (Gemini's)", old_path), ("+ retrained scope check", None)):
        r = build_retriever(get_settings())
        if scope_path:
            r.scope = ScopeCheck.load(scope_path)
            r.rewrite_gate = r.scope.threshold
        hit = sum(bool({s.section for s in r.retrieve(q["question"], rewrite=to_rw(rw.get(q["id"])))} & set(q["acceptable"]))
                  for q in final)
        rows.append((label, 100 * hit / len(final)))
    old_path.unlink()
    return rows


def fig_ablation() -> None:
    rows = ablation_rows()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    style(ax, "Retrieval on the 60 held-out scenarios as each component is added")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    labels, vals = [r[0] for r in rows][::-1], [r[1] for r in rows][::-1]
    ax.barh(labels, vals, height=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    for y, v in enumerate(vals):
        ax.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.set_xlabel("acceptable section among the retrieved sources (0 of 30 out-of-scope questions answered at every step)",
                  color=INK2, fontsize=8)
    save(fig, "fig3_ablation")


def fig_size(items, ipc) -> None:
    # the same v0.7 training examples (and split) and retrieval code for all three sizes
    systems = [("1.5B", "legalgpt-lora_v0.7-1.5b"), ("3B", "legalgpt-lora_v0.7-3b"), ("7B", "legalgpt-heavy_v0.7-7b")]
    from app.benchmark.metrics import load_jsonl
    from run_baselines import OUT as RUNS

    def complete(name: str) -> bool:  # a partial run would misstate the model
        path = RUNS / f"{name}.jsonl"
        return path.exists() and len({r["id"] for r in load_jsonl(path) if r["id"] in items}) == len(items)

    present = [(label, outcomes(name, items, ipc)) for label, name in systems if complete(name)]
    if len(present) < 2:
        print("fig4_size: skipped (needs at least two local-model benchmark runs)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1]})
    style(axes[0], "Accuracy (higher is better)")
    grouped(axes[0], ["Right section\n(statute, 59)", "Acceptable section\n(scenario, 60)"], [
        (label, [rate(o, "right section (statute)"), rate(o, "acceptable section (scenario)")]) for label, o in present
    ], ymax=115)
    style(axes[1], "Wrong punishment (lower is better)")
    grouped(axes[1], ["Punishment\nquestions (39)"], [(label, [rate(o, "wrong punishment")]) for label, o in present], ymax=115)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right", title="model size", title_fontsize=9)
    fig.suptitle("Local LoRA models by size: same training data, fully offline", x=0.01, ha="left", fontsize=12,
                 color=INK, y=1.04)
    save(fig, "fig4_size")


def fig_rrl() -> None:
    src = REPO_ROOT / "data" / "models" / "rrl_cv" / "v0.3-ablation-report.json"
    if not src.exists():
        print("fig5_rrl: skipped (run rrl_ablation_report.py --json-out first)")
        return
    rows = json.loads(src.read_text(encoding="utf-8"))["rows"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    style(ax, f"Sentence roles in judgments: macro-F1 on held-out judgments (5-fold CV, 36 judgments)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    labels, vals = [r["model"] for r in rows][::-1], [100 * r["macro_f1"] for r in rows][::-1]
    colours = [SERIES[1] if lab.startswith("Ensemble") else SERIES[0] for lab in labels]
    ax.barh(labels, vals, height=0.6, color=colours, edgecolor=SURFACE, linewidth=1)
    for y, v in enumerate(vals):
        ax.text(v + 0.6, y, f"{v:.1f}", va="center", fontsize=8, color=INK)
    ax.set_xlim(50, 85)
    ax.set_xlabel("macro-F1 x 100 (axis starts at 50)", color=INK2, fontsize=8)
    save(fig, "fig5_rrl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-ablation", action="store_true", help="skip fig3 (it re-runs retrieval, a few minutes)")
    args = ap.parse_args()
    items, ipc = items_and_ipc()
    fig_safety(items, ipc)
    fig_scenarios(items, ipc)
    if not args.skip_ablation:
        fig_ablation()
    fig_size(items, ipc)
    fig_rrl()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
