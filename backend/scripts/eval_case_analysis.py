"""Evaluate Case analysis on real Supreme Court judgments, without ever asking for a verdict.

For each judgment in the library (data/judgment_library/sc, roles from build_judgment_index.py):
  input   its Facts sentences (the facts and procedural history as the court records them), with
          every section number and Act name masked, so the model cannot copy the answer;
  gold    the IPC offences cited in the judgment's own Facts and Ruling sentences (what this case is
          about; precedent discussions, which quote unrelated sections, are left out). General
          provisions that are not offences (Chapter II, e.g. s.34; s.149) are excluded on both sides;
  run     the case-analysis fact step (extract_facts) and the similar-judgment search.
Reported:
  - offence identification: precision / recall / F1 of the sections the system names vs gold, for
    the model's raw proposal and after the IPC validation filter (exists + title matches);
  - the filter: how many proposed sections it dropped, and how many of those were in fact gold;
  - similar judgments: share of the top-3 (excluding the judgment itself) sharing a gold section.
Case-level bootstrap 95% intervals. Results are cached per judgment (re-runs resume).

Usage (from the repo root):
    python backend/scripts/eval_case_analysis.py --n 120 [--model groq]
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation.collector import JudgmentCollector  # noqa: E402
from app.case_analysis.analyzer import extract_facts  # noqa: E402
from app.case_analysis.router import StickyFallback, ipc_sections  # noqa: E402
from app.config import REPO_ROOT, get_settings  # noqa: E402

LIB = REPO_ROOT / "data" / "judgment_library"
OUT = REPO_ROOT / "data" / "reports" / "case_eval"

_MASKS = [
    # "Section 302/34 IPC", "Sections 304-B and 498-A of the Indian Penal Code", "u/s 302", "S. 376(2)"
    (re.compile(r"\b(?:sections?|secs?\.?|u/s\.?|s\.)\s*[\d][\dA-Z/,()\-\s]*(?:and\s+[\dA-Z/()\-]+\s*)*", re.I), "[provision] "),
    (re.compile(r"\b\d{2,3}[A-Z]?(?:\(\d+\))?(?:\s*/\s*\d{1,3}[A-Z]?(?:\(\d+\))?)+\b"), "[provisions]"),  # 302/34
    (re.compile(r"\b(?:Indian Penal Code|I\.\s?P\.\s?C\.?|IPC|Penal Code|Code of Criminal Procedure|Cr\.?\s?P\.?\s?C\.?)\b", re.I), "[code]"),
]


def mask(text: str, sections: list[str] = ()) -> str:
    """Section references and Act names out; then every remaining mention of the case's own
    sections in any spelling (304B, 304-B, 302/34, 376(2)), so no section number leaks."""
    for pattern, repl in _MASKS:
        text = pattern.sub(repl, text)
    for sec in sorted(sections, key=len, reverse=True):
        num, letters = re.match(r"(\d+)([A-Z]*)", sec).groups()
        spelled = num + (r"\s*-?\s*" + letters if letters else "")
        text = re.sub(rf"(?<![\d.])(?<!\d,){spelled}(?:\(\d+\))?(?!\d|,\d|\.\d)", "[s]", text, flags=re.I)
    return re.sub(r"\[provision\]\s*(?:of\s+the\s+)?\[code\]", "[provision]", text)


def general_sections(ipc_paths: dict[str, str]) -> set[str]:
    """Provisions that are not offences: Chapter II (general explanations, e.g. s.34 common
    intention), Chapter I (title, extent) and s.149 (liability of members of an unlawful assembly)."""
    return {sec for sec, path in ipc_paths.items() if "> Chapter II >" in path or "> Chapter I >" in path} | {"149"}


GENERAL: set[str] = set()


def judgments(ipc: dict[str, str]) -> list[dict]:
    out = []
    for meta_path in sorted((LIB / "sc").glob("*.json")):
        case_id = meta_path.stem
        roles_file = next(iter((LIB / "roles").glob(f"{case_id}.*.json")), None)
        if roles_file is None:
            continue
        r = json.loads(roles_file.read_text(encoding="utf-8"))
        own = " ".join(s for s, role in zip(r["sentences"], r["roles"]) if role in ("Facts", "Ruling"))
        gold = [s for s in JudgmentCollector.detect_sections(own) if s in ipc and s not in GENERAL]
        facts = " ".join(s for s, role in zip(r["sentences"], r["roles"]) if role == "Facts")
        if gold and len(facts) >= 600:
            out.append({"case_id": case_id, "title": json.loads(meta_path.read_text(encoding="utf-8"))["title"],
                        "gold": gold, "input": mask(facts, JudgmentCollector.detect_sections(" ".join(r["sentences"])))[:6000]})
    return out


def prf(pred: set, gold: set) -> tuple[float, float, float]:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default="", help="chat model id (default: the app default, falling back to Gemini)")
    args = ap.parse_args()

    from app.main import build_models
    from app.core.embeddings import EmbeddingService
    from app.core.vector_store import VectorStore
    from app.judgments.library import COLLECTION, JudgmentSearch

    s = get_settings()
    ipc = ipc_sections()
    from app.core.corpus import load_chunks

    GENERAL.update(general_sections({c.section: c.citation_path for c in load_chunks(s.corpus_chunks_path) if c.section}))
    models, _, default = build_models(s)
    first = models[args.model or default]
    backups = [m for k, m in models.items() if m is not first and not getattr(m, "is_local", False)
               and (k == "groq" or str(getattr(m, "model_id", "")).startswith("gemini:"))]
    search = JudgmentSearch(VectorStore(s.chroma_persist_dir, COLLECTION), EmbeddingService(s.embedding_model))
    pool = judgments(ipc)
    random.Random(args.seed).shuffle(pool)
    sample = pool[: args.n]
    print(f"{len(pool)} judgments with gold sections and enough facts; evaluating {len(sample)}")
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUT / "results.jsonl"
    cached = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cached[r["case_id"]] = r
    gold_of = {j["case_id"]: set(j["gold"]) for j in pool}
    title_key = lambda t: re.sub(r"[^a-z0-9]+", "", t.lower())[:40]  # noqa: E731
    t0 = time.time()
    with open(cache_path, "a", encoding="utf-8") as f:
        for n, j in enumerate(sample, 1):
            if j["case_id"] in cached:
                continue
            llm = StickyFallback([first] + backups)
            try:
                facts = extract_facts(j["input"], llm, ipc)
                raw = facts.offences + [(d["name"], d["section"]) for d in facts.dropped]
            except Exception as e:
                print(f"  {j['case_id']}: {type(e).__name__}: {str(e)[:120]}")
                continue
            query = facts.summary + " " + "; ".join(f"{a} (section {b})" for a, b in facts.offences)
            own = title_key(j["title"])
            hits = [c for c, _ in search.search(query, roles=("Facts",), k=6, min_similarity=0.0, exclude_cases=[j["case_id"]])
                    if title_key(c.title) != own][:3]
            rec = {"case_id": j["case_id"], "gold": j["gold"], "model": llm.model_id, "note": llm.note,
                   "pred_raw": sorted({b for _, b in raw} - GENERAL), "pred": sorted({b for _, b in facts.offences} - GENERAL),
                   "dropped": facts.dropped, "facts": len(facts.facts), "facts_verified": len(facts.verified),
                   "similar": [{"case_id": c.doc_id, "gold": sorted(gold_of.get(c.doc_id, set()))} for c in hits]}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            cached[j["case_id"]] = rec
            if n % 10 == 0:
                print(f"  {n}/{len(sample)} ({time.time() - t0:.0f}s)", flush=True)

    rows = [cached[j["case_id"]] for j in sample if j["case_id"] in cached]
    report(rows)
    return 0


def report(rows: list[dict]) -> None:
    rng = np.random.default_rng(13)

    def ci(values: list[float]) -> str:
        v = np.array(values)
        boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
        return f"{v.mean():.3f} (95% CI {np.percentile(boots, 2.5):.3f}-{np.percentile(boots, 97.5):.3f})"

    out = {"n": len(rows)}
    print(f"\n=== {len(rows)} judgments ===")
    for key, label in (("pred_raw", "model's raw proposal"), ("pred", "after the IPC validation filter")):
        p, r, f = zip(*(prf(set(x[key]), set(x["gold"])) for x in rows))
        print(f"{label:34} precision {ci(list(p))}  recall {ci(list(r))}  F1 {ci(list(f))}")
        hit = [bool(set(x[key]) & set(x["gold"])) for x in rows]
        print(f"{'':34} at least one correct offence: {ci([float(h) for h in hit])}")
        out[key] = {"precision": float(np.mean(p)), "recall": float(np.mean(r)), "f1": float(np.mean(f)), "hit": float(np.mean(hit))}
    dropped = [d for x in rows for d in x["dropped"]]
    dropped_gold = sum(d["section"] in x["gold"] for x in rows for d in x["dropped"])
    reasons = {}
    for d in dropped:
        k = "not an IPC section" if d["reason"].startswith("not") else "name does not match the IPC title"
        reasons[k] = reasons.get(k, 0) + 1
    print(f"filter dropped {len(dropped)} proposed sections {reasons}; of those, {dropped_gold} were gold")
    any_hit = [any(set(s["gold"]) & set(x["gold"]) for s in x["similar"][:1]) for x in rows if x["similar"]]
    p3 = [np.mean([bool(set(s["gold"]) & set(x["gold"])) for s in x["similar"]]) for x in rows if x["similar"]]
    print(f"similar judgments: top-1 shares a gold section {ci([float(v) for v in any_hit])}; "
          f"precision@3 {ci([float(v) for v in p3])}")
    exact = np.mean([set(x["pred"]) == set(x["gold"]) for x in rows])
    print(f"exactly the gold set: {exact:.1%}; facts verified against the text: "
          f"{sum(x['facts_verified'] for x in rows)}/{sum(x['facts'] for x in rows)}")
    out.update({"dropped": len(dropped), "dropped_reasons": reasons, "dropped_gold": dropped_gold,
                "similar_top1": float(np.mean(any_hit)), "similar_p3": float(np.mean(p3)), "exact": float(exact),
                "models": sorted({x["model"] for x in rows})})
    (OUT / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
