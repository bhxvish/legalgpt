"""Explaining what LoRA fine-tuning changed (the local models, Module 3).

Follows Section 7 of the LoRA paper (Hu et al., ICLR 2022, "Understanding the low-rank updates"),
which explains an adapter by analysing its weight update dW = (alpha/r) * B @ A, and adds two
behavioural tests that are possible because the model runs on this machine.

  weights    per layer and module (q, k, v, o, gate, up, down): the size of the update relative to the
             base weight ||dW||/||W||, its singular values and effective rank, and the paper's
             amplification factor ||dW||_F / ||U^T W V||_F (U, V: dW's singular directions) - large
             means the update amplifies directions the base model already had but did not emphasise.
  subspace   the paper's normalised subspace similarity between two adapters' A matrices
             (phi = ||U_i^T U_j||_F^2 / min(i, j)): did later training versions keep the same directions?
  behaviour  (GPU) on held-out questions with their real retrieved sources: the tuned model's answer is
             scored token by token under the tuned and the base model; the tokens whose probability the
             adapter raised most show what it taught (citation markers, section numbers, refusals...).
             Then each source is left out of the prompt and the drop in the answer's log-probability
             measures how much the answer relied on it - an inside-the-model attribution - compared with
             the sources the answer actually cites.

Usage (repo root):
    .venv-gpu\\Scripts\\python backend/scripts/analyze_lora.py weights subspace
    .venv-gpu\\Scripts\\python backend/scripts/analyze_lora.py behaviour --n 16
Writes docs/reports/lora_explainability.json (+ .md) and docs/figures/fig6_lora_*.png.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
LORA = ROOT / "data" / "models" / "lora"
OUT_JSON = ROOT / "docs" / "reports" / "lora_explainability.json"
OUT_MD = ROOT / "docs" / "reports" / "lora_explainability.md"
FIG = ROOT / "docs" / "figures"
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
KEY = re.compile(r"layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_([AB])\.weight")


def hf_snapshot(repo: str) -> Path:
    base = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots"
    snaps = sorted(base.iterdir())
    if not snaps:
        raise SystemExit(f"{repo} is not in the Hugging Face cache")
    return snaps[-1]


def load_adapter(name: str) -> tuple[dict, dict]:
    d = LORA / name
    cfg = json.loads((d / "adapter_config.json").read_text())
    mats: dict[tuple[int, str, str], dict[str, torch.Tensor]] = defaultdict(dict)
    with safe_open(str(d / "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            m = KEY.search(k)
            if m:
                mats[(int(m.group(1)), m.group(2), m.group(3))][m.group(4)] = f.get_tensor(k).float()
    return cfg, mats


class BaseWeights:
    def __init__(self, repo: str) -> None:
        snap = hf_snapshot(repo)
        if (snap / "model.safetensors.index.json").exists():  # sharded (3B and up)
            index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
            self.files = {k: snap / v for k, v in index.items()}
        else:  # one file (1.5B)
            with safe_open(str(snap / "model.safetensors"), "pt") as f:
                self.files = {k: snap / "model.safetensors" for k in f.keys()}
        self.handles: dict[Path, object] = {}

    def get(self, layer: int, block: str, module: str) -> torch.Tensor:
        key = f"model.layers.{layer}.{block}.{module}.weight"
        path = self.files[key]
        if path not in self.handles:
            self.handles[path] = safe_open(str(path), "pt")
        return self.handles[path].get_tensor(key).float()


def decompose(A: torch.Tensor, B: torch.Tensor, scale: float):
    """dW = scale * B @ A without forming it: singular values and directions via two QRs."""
    Qb, Rb = torch.linalg.qr(B)  # out x r, r x r
    Qa, Ra = torch.linalg.qr(A.T)  # in x r, r x r
    u, s, vh = torch.linalg.svd(Rb @ Ra.T)
    return Qb @ u, scale * s, Qa @ vh.T  # U (out x r), sigma, V (in x r)


def effective_rank(s: torch.Tensor) -> tuple[float, int]:
    p = s / s.sum()
    entropy = float(-(p * torch.log(p.clamp_min(1e-12))).sum())
    energy = torch.cumsum(s ** 2, 0) / (s ** 2).sum()
    return math.exp(entropy), int((energy < 0.9).sum().item()) + 1


def weights(adapter: str, device: str) -> dict:
    cfg, mats = load_adapter(adapter)
    scale = cfg["lora_alpha"] / cfg["r"]
    base = BaseWeights(cfg["base_model_name_or_path"])
    gen = torch.Generator().manual_seed(0)
    rows = []
    for (layer, block, module), ab in sorted(mats.items()):
        A, B = ab["A"].to(device), ab["B"].to(device)
        U, s, V = decompose(A, B, scale)
        W = base.get(layer, block, module).to(device)
        dW = float(torch.sqrt((s ** 2).sum()))
        Wn = float(torch.linalg.matrix_norm(W))
        proj = float(torch.linalg.matrix_norm(U.T @ W @ V))
        Ur = torch.linalg.qr(torch.randn(W.shape[0], U.shape[1], generator=gen).to(device))[0]
        Vr = torch.linalg.qr(torch.randn(W.shape[1], V.shape[1], generator=gen).to(device))[0]
        rand = float(torch.linalg.matrix_norm(Ur.T @ W @ Vr))
        top = float(torch.sqrt((torch.linalg.svdvals(W)[: U.shape[1]] ** 2).sum()))
        er, r90 = effective_rank(s.cpu())
        rows.append({"layer": layer, "module": module, "dW": round(dW, 5), "W": round(Wn, 3), "rel": round(dW / Wn, 6),
                     "amplification": round(dW / proj, 3) if proj else None, "proj_dW_dirs": round(proj, 4),
                     "proj_random": round(rand, 4), "proj_top_W": round(top, 3), "eff_rank": round(er, 2), "rank90": r90,
                     "sigma": [round(float(x), 5) for x in s.cpu()]})
    by_module = {m: {"rel": round(sum(r["rel"] for r in rows if r["module"] == m) / max(1, sum(r["module"] == m for r in rows)), 6),
                     "amplification": round(sum(r["amplification"] or 0 for r in rows if r["module"] == m) / max(1, sum(r["module"] == m for r in rows)), 2)}
                 for m in MODULES}
    layers = sorted({r["layer"] for r in rows})
    by_layer = [{"layer": L, "rel": round(sum(r["rel"] for r in rows if r["layer"] == L) / 7, 6)} for L in layers]
    return {"adapter": adapter, "base": cfg["base_model_name_or_path"], "r": cfg["r"], "alpha": cfg["lora_alpha"],
            "matrices": len(rows), "rows": rows, "by_module": by_module, "by_layer": by_layer,
            "mean_eff_rank": round(sum(r["eff_rank"] for r in rows) / len(rows), 2),
            "mean_rank90": round(sum(r["rank90"] for r in rows) / len(rows), 2),
            "mean_amplification": round(sum(r["amplification"] or 0 for r in rows) / len(rows), 2),
            "proj_ratio_dW_vs_random": round(sum(r["proj_dW_dirs"] for r in rows) / sum(r["proj_random"] for r in rows), 2),
            "proj_ratio_dW_vs_topW": round(sum(r["proj_dW_dirs"] for r in rows) / sum(r["proj_top_W"] for r in rows), 4)}


def subspace(a: str, b: str) -> dict:
    """Hu et al. eq. 4 on the A matrices' right singular vectors, averaged per module."""
    _, ma = load_adapter(a)
    _, mb = load_adapter(b)
    ks = (1, 2, 4, 8, 16)
    acc: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    rand: dict[int, list[float]] = defaultdict(list)
    gen = torch.Generator().manual_seed(0)
    for key in sorted(set(ma) & set(mb)):
        Va = torch.linalg.svd(ma[key]["A"], full_matrices=False)[2]  # r x in, rows = right singular vectors
        Vb = torch.linalg.svd(mb[key]["A"], full_matrices=False)[2]
        for k in ks:
            k = min(k, Va.shape[0], Vb.shape[0])
            acc[key[2]][k].append(float((Va[:k] @ Vb[:k].T).pow(2).sum()) / k)
        R = torch.linalg.qr(torch.randn(Va.shape[1], Va.shape[0], generator=gen))[0].T
        for k in ks:
            rand[k].append(float((Va[:k] @ R[:k].T).pow(2).sum()) / k)
    return {"a": a, "b": b, "k": list(ks),
            "by_module": {m: [round(sum(v) / len(v), 4) for k, v in sorted(acc[m].items())] for m in MODULES if acc[m]},
            "random_baseline": [round(sum(v) / len(v), 5) for k, v in sorted(rand.items())]}


# --- behaviour ----------------------------------------------------------------------------------------------

def held_out_prompts(n: int):
    from app.config import get_settings
    from app.core.prompt_builder import PromptBuilder
    from app.core.query_rewriter import parse
    from app.core.retriever import build_retriever
    from run_baselines import sample

    settings = get_settings()
    retriever = build_retriever(settings)
    rewrites = json.loads((ROOT / "data/reports/bench/rewrites/groq_openai_gpt-oss-120b.json").read_text(encoding="utf-8"))
    items = [q for q in sample(7, 40, 20, 60, 30) if q["set"] == "scenario"][:n]
    out = []
    for it in items:
        rw = parse(rewrites[it["question"]]) if it["question"] in rewrites else None
        result = retriever.retrieve_with_diagnostics(it["question"], rewrite=rw)
        if result.sources:
            out.append((it, list(result.sources), PromptBuilder()))
    return out


CATEGORIES = [
    ("citation marker", re.compile(r"\[\d+\]")),
    ("section number", re.compile(r"(?i)\b(?:section|s\.)\s*\d+[A-Z]{0,2}\b")),
    ("refusal", re.compile(r"(?i)I don't know|do not cover")),
    ("punishment terms", re.compile(r"(?i)\b(imprisonment|fine|years?|life|death|rigorous|simple)\b")),
]


def behaviour(adapter: str, n: int, max_new: int) -> dict:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cfg = json.loads((LORA / adapter / "adapter_config.json").read_text())
    repo = cfg["base_model_name_or_path"]
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16),
        device_map={"": 0})
    model = PeftModel.from_pretrained(model, str(LORA / adapter))
    model.eval()

    def ids_for(messages):
        chat = [{"role": m.role, "content": m.content} for m in messages]
        return tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].to(0)

    @torch.no_grad()
    def answer_logprobs(prompt_ids, answer_ids) -> torch.Tensor:
        full = torch.cat([prompt_ids, answer_ids], dim=1)
        logits = model(full, logits_to_keep=answer_ids.shape[1] + 1).logits[0, :-1].float()
        return torch.log_softmax(logits, -1).gather(1, answer_ids[0][:, None])[:, 0].cpu()

    rows, token_gain = [], defaultdict(list)
    cat_gain = defaultdict(list)
    for it, sources, pb in held_out_prompts(n):
        messages = pb.build(it["question"], sources, "legal", [])
        prompt_ids = ids_for(messages)
        with torch.no_grad():
            gen = model.generate(prompt_ids, max_new_tokens=max_new, do_sample=False)
        answer_ids = gen[:, prompt_ids.shape[1]:]
        if answer_ids.shape[1] and answer_ids[0, -1].item() in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")):
            answer_ids = answer_ids[:, :-1]
        if answer_ids.shape[1] == 0:
            continue
        answer = tok.decode(answer_ids[0], skip_special_tokens=True)
        tuned = answer_logprobs(prompt_ids, answer_ids)
        with model.disable_adapter():
            base = answer_logprobs(prompt_ids, answer_ids)
        gain = tuned - base
        # characters of each token, to map tokens to categories
        pieces = [tok.decode(answer_ids[0, : i + 1], skip_special_tokens=True) for i in range(answer_ids.shape[1])]
        starts = [0] + [len(p) for p in pieces[:-1]]
        for i, g in enumerate(gain.tolist()):
            token_gain[tok.decode(answer_ids[0, i : i + 1]).strip() or "␣"].append(g)
        for label, pat in CATEGORIES:
            spans = [(m.start(), m.end()) for m in pat.finditer(answer)]
            idx = [i for i, a in enumerate(starts) if any(s <= a < e for s, e in spans)]
            cat_gain[label] += [float(gain[i]) for i in idx]
        cat_gain["all tokens"] += gain.tolist()
        # leave one source out (tuned model): how much does the answer rely on each source?
        cited = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
        full_mean = float(tuned.mean())
        influence = []
        for s in sources:
            kept = [x for x in sources if x.marker != s.marker]
            lp = answer_logprobs(ids_for(pb.build(it["question"], kept, "legal", [])), answer_ids)
            influence.append({"marker": s.marker, "section": s.section, "drop": round(full_mean - float(lp.mean()), 4),
                              "cited": s.marker in cited})
        top = max(influence, key=lambda x: x["drop"])
        rows.append({"id": it["id"], "question": it["question"], "answer": answer[:600], "tokens": int(answer_ids.shape[1]),
                     "mean_logprob_tuned": round(full_mean, 4), "mean_logprob_base": round(float(base.mean()), 4),
                     "influence": influence, "top_source_cited": top["cited"] if cited else None, "cites": sorted(cited)})
        print(f"{it['id']}: {answer_ids.shape[1]} tokens, gain {float(gain.mean()):+.3f}, top source [{top['marker']}] cited={top['cited']}", flush=True)
    common = sorted(((t, sum(v) / len(v), len(v)) for t, v in token_gain.items() if len(v) >= 3), key=lambda x: -x[1])
    with_cites = [r for r in rows if r["cites"]]
    cited_drops = [i["drop"] for r in rows for i in r["influence"] if i["cited"]]
    uncited_drops = [i["drop"] for r in rows for i in r["influence"] if not i["cited"]]
    return {"adapter": adapter, "base": repo, "questions": len(rows), "max_new_tokens": max_new,
            "category_gain": {k: {"mean": round(sum(v) / len(v), 4), "tokens": len(v)} for k, v in cat_gain.items() if v},
            "top_tokens_raised": [{"token": t, "mean_gain": round(g, 3), "count": c} for t, g, c in common[:25]],
            "top_tokens_lowered": [{"token": t, "mean_gain": round(g, 3), "count": c} for t, g, c in common[-15:]],
            "influence": {"answers_with_citations": len(with_cites),
                          "top_source_is_cited": sum(1 for r in with_cites if r["top_source_cited"]),
                          "mean_drop_cited": round(sum(cited_drops) / max(1, len(cited_drops)), 4),
                          "mean_drop_uncited": round(sum(uncited_drops) / max(1, len(uncited_drops)), 4)},
            "rows": rows}


# --- report -------------------------------------------------------------------------------------------------

def figures(report: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = report.get("weights", {}).get("v0.7-3b-l4")
    if w:
        layers = sorted({r["layer"] for r in w["rows"]})
        grid = [[next((r["rel"] for r in w["rows"] if r["layer"] == L and r["module"] == m), 0) for L in layers] for m in MODULES]
        fig, ax = plt.subplots(figsize=(10, 3.2))
        im = ax.imshow(grid, aspect="auto", cmap="Blues")
        ax.set_yticks(range(len(MODULES)), MODULES)
        ax.set_xticks(range(0, len(layers), 2), [str(L) for L in layers[::2]])
        ax.set_xlabel("layer")
        ax.set_title("Where LoRA changed Qwen2.5-3B: relative update ||dW|| / ||W|| per layer and module")
        fig.colorbar(im, ax=ax, fraction=0.02)
        fig.tight_layout()
        fig.savefig(FIG / "fig6_lora_layers.png", dpi=160)
        fig.savefig(FIG / "fig6_lora_layers.pdf")
        plt.close(fig)
    b = report.get("behaviour")
    if b and b.get("category_gain"):
        cats = [c for c in ["citation marker", "section number", "refusal", "punishment terms", "all tokens"]
                if c in b["category_gain"] and b["category_gain"][c]["tokens"] >= 20]  # too few tokens: not shown
        vals = [b["category_gain"][c]["mean"] for c in cats]
        fig, ax = plt.subplots(figsize=(6.5, 2.8))
        ax.barh(cats[::-1], vals[::-1], color=["#4f46e5" if c != "all tokens" else "#94a3b8" for c in cats[::-1]])
        ax.axvline(0, color="#334155", lw=0.8)
        ax.set_xlabel("mean log-probability gain, tuned vs base (nats per token)")
        ax.set_title("What the adapter made more likely in its own answers")
        fig.tight_layout()
        fig.savefig(FIG / "fig7_lora_tokens.png", dpi=160)
        fig.savefig(FIG / "fig7_lora_tokens.pdf")
        plt.close(fig)


def markdown(report: dict) -> str:
    out = ["# What LoRA fine-tuning changed (explainability of the local models)", "",
           "Method: Section 7 of the LoRA paper (Hu et al., ICLR 2022) on our adapters' weight updates, plus "
           "behavioural tests on held-out questions. Reproduce: `backend/scripts/analyze_lora.py`.", ""]
    for name, w in report.get("weights", {}).items():
        out += [f"## Weights: {name} (base {w['base']}, r={w['r']}, alpha={w['alpha']}, {w['matrices']} matrices)", "",
                "| module | mean ||dW||/||W|| | mean amplification |", "|---|---|---|"]
        out += [f"| {m} | {v['rel']:.5f} | {v['amplification']} |" for m, v in w["by_module"].items()]
        top_layers = sorted(w["by_layer"], key=lambda x: -x["rel"])[:5]
        out += ["", f"- Layers changed most: {', '.join(str(x['layer']) for x in top_layers)}.",
                f"- Effective rank of dW (of r={w['r']}): mean {w['mean_eff_rank']}; directions holding 90% of the update: mean {w['mean_rank90']}.",
                f"- Amplification factor ||dW||/||U^T W V||: mean {w['mean_amplification']}.",
                f"- W projected on dW's directions vs random directions: {w['proj_ratio_dW_vs_random']}x; vs W's own top directions: {w['proj_ratio_dW_vs_topW']}.", ""]
    for s in report.get("subspace", []):
        out += [f"## Subspace similarity: {s['a']} vs {s['b']}", "", "| module | " + " | ".join(f"k={k}" for k in s["k"]) + " |",
                "|---|" + "---|" * len(s["k"])]
        out += [f"| {m} | " + " | ".join(str(x) for x in v) + " |" for m, v in s["by_module"].items()]
        out += [f"| random | " + " | ".join(str(x) for x in s["random_baseline"]) + " |", ""]
    b = report.get("behaviour")
    if b:
        out += [f"## Behaviour: {b['adapter']} on {b['questions']} held-out scenarios", "",
                "| tokens | mean log-prob gain (tuned - base) | tokens |", "|---|---|---|"]
        out += [f"| {k} | {v['mean']:+.3f} | {v['tokens']} |" for k, v in b["category_gain"].items()]
        i = b["influence"]
        out += ["", "Most raised tokens: " + ", ".join(f"`{t['token']}` ({t['mean_gain']:+.2f})" for t in b["top_tokens_raised"][:12]) + ".",
                "", f"Inside-the-model source attribution (leave one source out, drop in the answer's mean log-probability): "
                    f"the most influential source is one the answer cites in {i['top_source_is_cited']} of {i['answers_with_citations']} answers "
                    f"with citations; mean drop {i['mean_drop_cited']} for cited sources vs {i['mean_drop_uncited']} for uncited ones.", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="*", default=["weights", "subspace"], choices=["weights", "subspace", "behaviour", "report"])
    ap.add_argument("--adapters", nargs="*", default=["v0.7-3b-l4", "v0.6-scenario-refusals"])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=220)
    args = ap.parse_args()
    report = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "weights" in args.parts:
        report["weights"] = {a: weights(a, device) for a in args.adapters}
        for a, w in report["weights"].items():
            print(a, "rel by module", {m: v["rel"] for m, v in w["by_module"].items()}, "eff rank", w["mean_eff_rank"],
                  "amp", w["mean_amplification"])
    if "subspace" in args.parts:
        report["subspace"] = [subspace("v0.5-scenario", "v0.6-scenario-refusals"), subspace("v0.4-plain", "v0.5-scenario")]
        for s in report["subspace"]:
            print(s["a"], "vs", s["b"], s["by_module"].get("q_proj"), "random", s["random_baseline"])
    if "behaviour" in args.parts:
        report["behaviour"] = behaviour("v0.7-3b-l4", args.n, args.max_new)
        print(json.dumps({k: v for k, v in report["behaviour"].items() if k != "rows"}, indent=1)[:2500])
    OUT_JSON.write_text(json.dumps(report, indent=1), encoding="utf-8")
    OUT_MD.write_text(markdown(report), encoding="utf-8")
    figures(report)
    print("wrote", OUT_JSON.relative_to(ROOT), OUT_MD.relative_to(ROOT))


if __name__ == "__main__":
    main()
