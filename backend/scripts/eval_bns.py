"""Answering under the BNS: retrieval from the Bharatiya Nyaya Sanhita's own index.

The 60 held-out benchmark scenarios and 30 out-of-scope questions (run_baselines.py, seed 7), with
each scenario's acceptable IPC sections converted to BNS sections by the official BPRD table
(app/core/bns.py). Compared:
  ipc          the IPC index with the LLM rewrite (the app on the IPC)
  bns          the BNS index, the rewrite's IPC numbers converted to BNS numbers (the app on the BNS)
  bns-norw     the BNS index without the rewrite (embedding search only)
hit = an acceptable section among the sources; for out-of-scope questions, "answered" = any sources.
Rewrites are the cached ones (data/reports/bench/rewrites/), so nothing is sent to a model.

Usage:  python backend/scripts/eval_bns.py [--rewrite groq:openai/gpt-oss-120b]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import REPO_ROOT, get_settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rewrite", default="groq:openai/gpt-oss-120b")
    args = ap.parse_args()
    from app.core.bns import base, bns_for
    from app.core.chat_router import ChatRouter
    from app.core.embeddings import EmbeddingService
    from app.core.retriever import build_retriever
    from eval_scenario_bench import CachedRewriter
    from run_baselines import sample

    s = get_settings()
    emb = EmbeddingService(s.embedding_model)
    ipc, bns = build_retriever(s, emb), build_retriever(s, emb, law="bns")
    rewriter = CachedRewriter(args.rewrite)
    items = [it for it in sample(7, 40, 20, 60, 30) if it["id"].startswith(("sc-", "oos-"))]
    conditions = {
        "ipc": lambda q: ipc.retrieve_with_diagnostics(q, rewrite=rewriter(q)),
        "bns": lambda q: bns.retrieve_with_diagnostics(q, rewrite=(lambda r: ChatRouter._to_bns(r) if r else None)(rewriter(q))),
        "bns-norw": lambda q: bns.retrieve_with_diagnostics(q),
    }
    report, rows = {}, []
    for name, run in conditions.items():
        hit = refused = oos_answered = n_sc = n_oos = 0
        for it in items:
            secs = [x.section for x in run(it["question"]).sources]
            if it["id"].startswith("oos-"):
                n_oos += 1
                oos_answered += bool(secs)
                continue
            n_sc += 1
            want = set(it["acceptable"]) if name == "ipc" else {base(b) for sec in it["acceptable"] for b in bns_for(sec)}
            ok = bool(want & set(secs))
            hit += ok
            refused += not secs
            rows.append({"condition": name, "id": it["id"], "want": sorted(want), "got": secs, "hit": ok})
        report[name] = {"scenarios": n_sc, "hit": hit, "refused": refused, "oos": n_oos, "oos_answered": oos_answered}
        print(f"  {name:9} hit {hit}/{n_sc}  refused {refused}/{n_sc}  out-of-scope answered {oos_answered}/{n_oos}", flush=True)
    no_bns = sorted({sec for it in items if it["id"].startswith("sc-") for sec in it["acceptable"] if not bns_for(sec)})
    report["ipc_sections_without_bns_equivalent"] = no_bns
    out = REPO_ROOT / "data" / "reports" / "bns_eval.json"
    out.write_text(json.dumps({"summary": report, "rows": rows}, indent=1), encoding="utf-8")
    print(f"acceptable IPC sections with no BNS equivalent: {no_bns or 'none'}\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
