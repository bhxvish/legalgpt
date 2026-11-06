"""How faithful are the key-word explanations? A deletion test (ERASER-style comprehensiveness).

For each held-out scenario (the benchmark's seed-7 sample, retrieved with the cached gpt-oss-120b
rewrites; no model calls), word_importance ranks the question's words by how much removing each one
lowers the match with the top retrieved section and the scope check's crime probability. Then the top
k words are removed together, and the drop is compared with removing k random content words (20
draws). If the ranking reflects what the retrieval actually uses, removing the top words must hurt far
more than removing random ones.

Usage (repo root): python backend/scripts/eval_word_importance.py
"""

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
OUT = ROOT / "data" / "reports" / "explain" / "word_importance.json"


def main() -> int:
    from app.config import get_settings
    from app.core.query_rewriter import parse
    from app.core.retriever import build_retriever
    from app.explainability.perturbation import by_section, word_importance
    from run_baselines import sample

    retriever = build_retriever(get_settings())
    rewrites = json.loads((ROOT / "data/reports/bench/rewrites/groq_openai_gpt-oss-120b.json").read_text(encoding="utf-8"))
    items = [q for q in sample(7, 40, 20, 60, 30) if q["set"] == "scenario"]
    rng = random.Random(0)
    ks = (1, 2, 3)
    res = {k: {"match_top": [], "match_rand": [], "scope_top": [], "scope_rand": []} for k in ks}
    used = 0
    emb, exp, scope = retriever.embedder, retriever.expander, retriever.scope

    def measure(q: str, chunk_vec):
        v = emb.embed([exp.search_query(q)])[0]
        p = scope.probability(emb.embed([exp.expand(q)])[0]) if scope is not None else 0.0
        return float(np.dot(v, chunk_vec) / (np.linalg.norm(v) * np.linalg.norm(chunk_vec))), p

    def without(q: str, words):
        for a, b in sorted(((w["start"], w["end"]) for w in words), reverse=True):
            q = q[:a] + q[b:]
        return " ".join(q.split())

    for it in items:
        rw = parse(rewrites[it["question"]]) if it["question"] in rewrites else None
        result = retriever.retrieve_with_diagnostics(it["question"], rewrite=rw)
        statute = by_section(result.sources)
        if not statute:
            continue
        top = statute[0]
        wi = word_importance(it["question"], [top], retriever)
        words = wi["words"]
        if len(words) < 4:
            continue
        vec = retriever.store.get_embeddings([top.chunk_id])[top.chunk_id]
        base_m, base_p = measure(it["question"], vec)
        by_match = sorted(words, key=lambda w: -w["match_drop"][str(top.marker)])
        by_scope = sorted(words, key=lambda w: -(w["scope_drop"] or 0))
        for k in ks:
            m, _ = measure(without(it["question"], by_match[:k]), vec)
            _, p = measure(without(it["question"], by_scope[:k]), vec)
            res[k]["match_top"].append(base_m - m)
            res[k]["scope_top"].append(base_p - p)
            rm, rp = [], []
            for _ in range(20):
                pick = rng.sample(words, k)
                m2, p2 = measure(without(it["question"], pick), vec)
                rm.append(base_m - m2)
                rp.append(base_p - p2)
            res[k]["match_rand"].append(sum(rm) / len(rm))
            res[k]["scope_rand"].append(sum(rp) / len(rp))
        used += 1
    summary = {"questions": used, "k": list(ks), "by_k": {}}
    for k in ks:
        r = res[k]
        mean = lambda xs: round(sum(xs) / len(xs), 4)  # noqa: E731
        summary["by_k"][str(k)] = {
            "match_drop_top": mean(r["match_top"]), "match_drop_random": mean(r["match_rand"]),
            "scope_drop_top": mean(r["scope_top"]), "scope_drop_random": mean(r["scope_rand"]),
            "match_top_beats_random": round(sum(a > b for a, b in zip(r["match_top"], r["match_rand"])) / used, 3),
        }
        print(k, summary["by_k"][str(k)])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
