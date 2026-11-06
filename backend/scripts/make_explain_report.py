"""Collect the explainability results into docs/reports/explainability_results.json (served by
GET /api/explain/results for the Explainability page) and docs/reports/explainability.md.

Inputs (each optional; a missing one is left out):
  data/reports/bias/summary_<model>.json           eval_bias.py
  data/reports/bias/summary_<attribute>_<model>.json  eval_bias.py --attribute (caste, income, region, age)
  data/reports/explain/reasoning_summary_<model>.json  eval_reasoning.py
  data/reports/explain/word_importance.json         eval_word_importance.py
  docs/reports/lora_explainability.json             analyze_lora.py
  docs/reports/classifier_explanations.md           eval_explanations.py (InLegalBERT; numbers copied below)
"""

import json
import shutil
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "groq_openai_gpt-oss-120b"
OUT = ROOT / "docs" / "reports" / "explainability_results.json"
MD = ROOT / "docs" / "reports" / "explainability.md"
ATTRIBUTES = ("caste", "income", "region", "age")


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts."""
    n = b + c
    return 1.0 if not n else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def attribute_totals(key: str) -> dict:
    """The four attributes together for one model: model alone vs LegalGPT, the rule with one control,
    and how often LegalGPT's search found different sections in flagged and unflagged pairs."""
    flagged: dict = {"closed": {}, "legalgpt": {}}
    one_control = {"closed": 0, "legalgpt": 0}
    search = {"flagged": [0, 0], "not_flagged": [0, 0]}  # [search changed, pairs]
    stereotypes = refusals = 0
    for attr in ATTRIBUTES:
        s = load(ROOT / "data/reports/bias" / f"summary_{attr}_{key}.json")
        lines = (ROOT / "data/reports/bias" / f"legalgpt_{attr}_{key}.jsonl").read_text(encoding="utf-8").splitlines()
        rows = {(r["id"], r["variant"]): r for r in map(json.loads, lines)}
        for system, d in s.items():
            one_control[system] += d["possible_bias_one_control"]
            for p in d["pairs_detail"]:
                flagged[system][attr, p["id"]] = p["verdict"] == "possible_bias"
                stereotypes += p["stereotypes"]
                refusals += bool(p["refusal_changed"])
        for p in s["legalgpt"]["pairs_detail"]:
            first, second = (set(rows[p["id"], v].get("retrieved") or []) for v in ("original", "swapped"))
            side = search["flagged" if p["verdict"] == "possible_bias" else "not_flagged"]
            side[0] += first != second
            side[1] += 1
    alone, legal = flagged["closed"], flagged["legalgpt"]
    return {"pairs": len(alone), "closed": sum(alone.values()), "legalgpt": sum(legal.values()),
            "one_control": one_control, "search_changed": search, "stereotypes": stereotypes, "refusal_changes": refusals,
            "mcnemar_p": round(mcnemar(sum(alone[k] and not legal[k] for k in alone), sum(legal[k] and not alone[k] for k in alone)), 3)}


def flags(run: str, key: str, verdict: str = "verdict_merged", attributes=ATTRIBUTES) -> dict:
    """{system: {(attribute, scenario): flagged}} for the attributes of one run; {} until all of them exist."""
    out: dict = {}
    for attr in attributes:
        s = load(ROOT / "data/reports/bias" / f"summary_{run}{attr}_{key}.json")
        if not s:
            return {}
        for system, d in s.items():
            out.setdefault(system, {}).update({(attr, p["id"]): p[verdict] == "possible_bias" for p in d["pairs_detail"]})
    return out


def compare(a: dict, b: dict) -> dict:
    """Two flag sets over the same pairs, with the exact McNemar test."""
    return {"pairs": len(a), "first": sum(a.values()), "second": sum(b.values()),
            "mcnemar_p": float(f"{mcnemar(sum(a[k] and not b[k] for k in a), sum(b[k] and not a[k] for k in a)):.2g}")}


def filter_caught() -> list:
    """The attributes whose held-out wordings the filter really removes. For the others the search gets
    the same text with the filter on or off, so those two runs only show run-to-run noise."""
    sys.path.insert(0, str(ROOT / "backend"))
    from app.core.background import strip_background

    wordings = load(ROOT / "benchmark/bias/held_out_wordings.json")
    return [a for a in ATTRIBUTES if all(strip_background("A man broke the lock. " + w)[1] for w in wordings[a])]


def stages(key: str) -> dict:
    """How the caste, income, region and age figures moved: as first reported, with the two parts of one
    offence counted once, with the filter that keeps the added sentence out of the search, then with
    wordings that filter was not built around (the same scenarios, the filter on and off)."""
    out: dict = {}
    first, merged, fixed = flags("", key, "verdict_separate"), flags("", key), flags("fixed_", key)
    held, off = flags("heldout_", key), flags("heldout_off_", key)
    if first:
        out["first_reported"] = compare(first["closed"], first["legalgpt"])
        out["one_offence_counted_once"] = compare(merged["closed"], merged["legalgpt"])
    if merged and fixed:
        out["with_filter"] = compare(merged["closed"], fixed["legalgpt"])
    if held:
        out["held_out"] = compare(held["closed"], held["legalgpt"])
    if held and off:
        out["held_out_filter_off_vs_on"] = compare(off["legalgpt"], held["legalgpt"])
        caught = filter_caught()
        missed = [a for a in ATTRIBUTES if a not in caught]
        out["filter_removed"] = caught
        out["filter_missed"] = missed
        for name, attrs in (("sentence_removed", caught), ("sentence_kept", missed)):
            if attrs:
                out[f"off_vs_on_{name}"] = compare(flags("heldout_off_", key, attributes=attrs)["legalgpt"],
                                                   flags("heldout_", key, attributes=attrs)["legalgpt"])
    return out


def main() -> int:
    res: dict = {"model": "gpt-oss-120b (Groq)"}
    names = {MODEL: "gpt-oss-120b (Groq)", "gemini_gemini-3.5-flash-lite": "Gemini 3.5 flash-lite"}
    complete = {}
    for key, label in names.items():  # only complete runs (every pair, both systems) are reported
        b = load(ROOT / "data/reports/bias" / f"summary_{key}.json")
        if b and all(d["pairs"] == 40 for d in b.values()) and len(b) == 2:
            complete[label] = {s: {k: v for k, v in d.items() if k != "pairs_detail"} | {
                "flagged": [p["id"] for p in d["pairs_detail"] if p["verdict"] == "possible_bias"],
                "misapplied": [p["id"] for p in d["pairs_detail"] if p["law_notes"]]} for s, d in b.items()}
    gender_p = {}
    for key, label in names.items():
        g = load(ROOT / "data/reports/bias" / f"summary_{key}.json")
        if label in complete:
            first, second = ({p["id"]: p["verdict"] == "possible_bias" for p in g[x]["pairs_detail"]} for x in ("closed", "legalgpt"))
            gender_p[label] = compare(first, second)["mcnemar_p"]
    if complete:
        res["bias_gender_p"] = gender_p
        res["bias_models"] = complete
        res["bias_model"] = next(iter(complete))
        res["bias"] = complete[res["bias_model"]]
    attributes: dict = {}
    for key, label in names.items():
        for attr in ATTRIBUTES:
            b = load(ROOT / "data/reports/bias" / f"summary_{attr}_{key}.json")
            if b and len(b) == 2 and all(d["pairs"] == 10 for d in b.values()):
                attributes.setdefault(label, {})[attr] = {s: d["by_group"]["neutral"] | {
                    "pairs": d["pairs"], "stereotypes": d["stereotype_sentences"],
                    "possible_bias_one_control": d.get("possible_bias_one_control"),
                    "flagged": [p["id"] for p in d["pairs_detail"] if p["verdict"] == "possible_bias"]} for s, d in b.items()}
    if attributes:
        res["bias_attributes"] = attributes
        res["bias_attribute_totals"] = {label: attribute_totals(key) for key, label in names.items()
                                        if len(attributes.get(label, {})) == len(ATTRIBUTES)}
    res["bias_stages"] = {label: stages(key) for key, label in names.items() if stages(key)}
    summaries = sorted((ROOT / "data/reports/explain").glob("reasoning_summary_gemini*.json")) or sorted((ROOT / "data/reports/explain").glob("reasoning_summary_*.json"))
    reasoning = load(summaries[-1]) if summaries else None
    if reasoning:
        res["reasoning"] = reasoning
    words = load(ROOT / "data/reports/explain/word_importance.json")
    if words:
        res["word_importance"] = words
    lora = load(ROOT / "docs/reports/lora_explainability.json")
    if lora:
        w = lora.get("weights", {}).get("v0.7-3b-l4")
        b = lora.get("behaviour")
        res["lora"] = {
            "weights": w and {k: w[k] for k in ("adapter", "base", "r", "alpha", "matrices", "by_module", "mean_eff_rank",
                                                  "mean_rank90", "mean_amplification", "proj_ratio_dW_vs_random")}
            | {"top_layers": [x["layer"] for x in sorted(w["by_layer"], key=lambda x: -x["rel"])[:5]]},
            "subspace": [{k: s[k] for k in ("a", "b", "k", "random_baseline")} | {"q_proj": s["by_module"].get("q_proj"),
                         "mean": [round(sum(v[i] for v in s["by_module"].values()) / len(s["by_module"]), 4) for i in range(len(s["k"]))]}
                         for s in lora.get("subspace", [])],
            "behaviour": b and {k: b[k] for k in ("adapter", "questions", "category_gain", "influence")}
            | {"top_tokens_raised": b["top_tokens_raised"][:12]},
        }
        pub = ROOT / "frontend" / "public" / "explain"
        pub.mkdir(parents=True, exist_ok=True)
        for f in ("fig6_lora_layers.png", "fig7_lora_tokens.png"):
            if (ROOT / "docs/figures" / f).exists():
                shutil.copy(ROOT / "docs/figures" / f, pub / f)
    # InLegalBERT word explanations (docs/reports/classifier_explanations.md, 100 held-out sentences)
    res["classifier"] = {"sentences": 100, "comprehensiveness_at_10": {"occlusion": 0.163, "integrated_gradients": 0.091, "random": 0.022},
                         "sufficiency_at_20": {"occlusion": 0.025, "integrated_gradients": 0.071, "random": 0.118},
                         "context_share": 0.65, "source": "docs/reports/classifier_explanations.md"}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    MD.write_text(markdown(res), encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT), "and", MD.relative_to(ROOT), "with", ", ".join(k for k in res if k != "model"))
    return 0


def markdown(r: dict) -> str:
    out = ["# Explainability results", "",
           "How LegalGPT explains its decisions, and how faithful each explanation is. Reproduce: backend/scripts/"
           "eval_bias.py, eval_reasoning.py, eval_word_importance.py, analyze_lora.py, then make_explain_report.py.", ""]
    for model, bias in r.get("bias_models", {}).items():
        out += [f"## Bias check: gender swapped (40 scenario pairs), {model}", "",
                "Both versions run twice; a section counts only if it holds in both runs of one version and neither run of the other.", "",
                "| system | group | no difference | required by law | run-to-run variation | possible bias |", "|---|---|---|---|---|---|"]
        for s, d in bias.items():
            for g, v in d["by_group"].items():
                out.append(f"| {'model alone' if s == 'closed' else 'LegalGPT'} | {g} | {v['no_difference']} | {v['law_required']} | {v['variation']} | {v['possible_bias']} |")
        out += [""] + [f"- {'Model alone' if s == 'closed' else 'LegalGPT'}: possible bias in {round(d['possible_bias_rate'] * 40)} of 40 pairs; "
                       f"a gender-specific section relied on in both versions (likely misapplied in one) in {d['gender_specific_in_both']} pairs; "
                       f"{d['stereotype_sentences']} stereotyped sentences." for s, d in bias.items()]
        out += [f"- Model alone vs LegalGPT: exact McNemar p = {r['bias_gender_p'][model]}.", ""]
    if "bias_attributes" in r:
        out += ["## Bias check: caste, income level, region and age (10 scenarios each, where none of these should matter)", "",
                "One sentence about the accused differs between two otherwise identical versions (a stated age or age word is "
                "swapped where the scenario has one). Both versions run twice: possible bias means a section held in both runs "
                "of one version and neither run of the other. The last column is the looser rule with one control.", "",
                "| model | attribute | system | no difference | run-to-run variation | possible bias | with one control |", "|---|---|---|---|---|---|---|"]
        for model, attrs in r["bias_attributes"].items():
            for attr, systems in attrs.items():
                for s, v in systems.items():
                    out.append(f"| {model} | {attr} | {'model alone' if s == 'closed' else 'LegalGPT'} | {v['no_difference']} | {v['variation']} | {v['possible_bias']} | {v['possible_bias_one_control']} |")
        out.append("")
        for model, t in r.get("bias_attribute_totals", {}).items():
            s = t["search_changed"]
            out.append(f"- {model}, four attributes together: possible bias in {t['closed']} of {t['pairs']} pairs for the model alone and "
                       f"{t['legalgpt']} inside LegalGPT (exact McNemar p = {t['mcnemar_p']}); {t['stereotypes']} stereotyped sentences, "
                       f"{t['refusal_changes']} changed refusals. LegalGPT's search found different sections in {s['flagged'][0]} of "
                       f"{s['flagged'][1]} flagged pairs and {s['not_flagged'][0]} of {s['not_flagged'][1]} others.")
        out += ["", "LegalGPT cites only what it retrieves, so this shows where the added sentence enters, not why the result changed. "
                    "Some differences are the definition and the punishment of one offence (s.441 and s.447). "
                    "10 scenarios per attribute is too few to conclude that a system is or is not biased.", ""]
    if r.get("bias_stages"):
        names = {"first_reported": "As first reported", "one_offence_counted_once": "One offence counted once",
                 "with_filter": "With the filter", "held_out": "New wordings, filter on"}
        out += ["## Bias check: how the figures moved (caste, income, region and age together)", "",
                "One offence counted once: a definition and its punishment section are one difference, not two. The filter "
                "removes the added sentence about the person from the search, so it cannot change which sections are found; "
                "it was built around one wording per attribute, then frozen and tested on a new wording in the same scenarios. "
                "The model alone has no search, so the filter does not touch it.", "",
                "| model | stage | model alone | LegalGPT | pairs | exact McNemar p |", "|---|---|---|---|---|---|"]
        for model, st in r["bias_stages"].items():
            out += [f"| {model} | {label} | {st[k]['first']} | {st[k]['second']} | {st[k]['pairs']} | {st[k]['mcnemar_p']} |"
                    for k, label in names.items() if k in st]
        out.append("")
        for model, st in r["bias_stages"].items():
            if "held_out_filter_off_vs_on" in st:
                t = st["held_out_filter_off_vs_on"]
                out.append(f"- {model}, new wordings: LegalGPT flagged {t['first']} of {t['pairs']} with the filter off and "
                           f"{t['second']} with it on (exact McNemar p = {t['mcnemar_p']}).")
                a, b = st.get("off_vs_on_sentence_removed"), st.get("off_vs_on_sentence_kept")
                if a and b:
                    out.append(f"  - It removed the new {' and '.join(st['filter_removed'])} wordings: {a['first']} off, {a['second']} on, of "
                               f"{a['pairs']} (p = {a['mcnemar_p']}). It missed the new {' and '.join(st['filter_missed'])} wordings, so the "
                               f"search got the same text both times: {b['first']} and {b['second']} of {b['pairs']}, which is run-to-run noise.")
        out.append("")
    if "reasoning" in r:
        x = r["reasoning"]
        out += [f"## Reasoning trace faithfulness ({x['questions']} held-out scenarios)", "",
                f"- Key facts named by the trace that really change the conclusion when removed: {x['facts']['key_facts_confirmed']} of {x['facts']['key_facts']}.",
                *([f"- Facts it did not name that changed the conclusion anyway: {x['facts']['unnamed_influential']} of {x['facts']['unnamed']}."]
                  if x['facts']['unnamed'] else []),
                f"- Mean agreement (named == influential): {x['facts']['agreement_mean']}.",
                f"- Sections the trace says apply that are decisive when left out: precision {x['sections']['precision']}, recall {x['sections']['recall']}.",
                f"- Key facts quoted word for word: {x['trace']['key_facts_verbatim']:.0%}; invalid sections/sources removed per trace: {x['trace']['invalid_items_removed_per_trace']}.",
                f"- Controls (unchanged input, asked again) that changed: facts {x['facts']['control_changed']:.0%}, sources {x['sections']['control_changed']:.0%}.", ""]
    if "word_importance" in r:
        w = r["word_importance"]
        out += [f"## Key-word explanations: deletion test ({w['questions']} held-out scenarios, no model calls)", "",
                "| words removed | match drop: top words | match drop: random words | crime-probability drop: top | random |", "|---|---|---|---|---|"]
        out += [f"| {k} | {v['match_drop_top']} | {v['match_drop_random']} | {v['scope_drop_top']} | {v['scope_drop_random']} |" for k, v in w["by_k"].items()] + [""]
    if "lora" in r:
        lw, lb = r["lora"]["weights"], r["lora"]["behaviour"]
        if lw:
            out += [f"## LoRA: what fine-tuning changed ({lw['adapter']}, base {lw['base']})", "",
                    f"- Largest relative updates: gate_proj {lw['by_module']['gate_proj']['rel']}, up_proj {lw['by_module']['up_proj']['rel']} "
                    f"(attention q/k/v/o about {lw['by_module']['q_proj']['rel']}); layers changed most: {lw['top_layers']}.",
                    f"- Effective rank {lw['mean_eff_rank']} of r={lw['r']}; amplification factor {lw['mean_amplification']}.", ""]
        if lb:
            i = lb["influence"]
            out += [f"- Tuned vs base on its own answers ({lb['questions']} held-out scenarios): "
                    f"answers {lb['category_gain'].get('all tokens', {}).get('mean')} nats per token more likely (citation markers " f"{lb['category_gain'].get('citation marker', {}).get('mean')}): a change of style across the whole answer.",
                    f"- Leave-one-source-out inside the model: the most influential source is cited in {i['top_source_is_cited']} of "
                    f"{i['answers_with_citations']} answers; mean drop {i['mean_drop_cited']} (cited) vs {i['mean_drop_uncited']} (uncited).", ""]
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
