"""Check a downloaded proof or signed answer offline, from the published files only.

Uses docs/zk/ (manifest, verification keys, verdict layout) and data/corpus/ipc_chunks.jsonl -
no server, no proving keys, nothing the prover controls. Needs EZKL for proofs
(pip install -r backend/requirements-zk.txt); signed answers need only `cryptography`.

Usage (from the repo root):
    python backend/scripts/zk_verify.py verdict-proof-s304A.json
    python backend/scripts/zk_verify.py verdict-proof-s304A.json --facts facts.json   # disclosed facts
    python backend/scripts/zk_verify.py retrieval-proof.json [--no-embed]
    python backend/scripts/zk_verify.py signed-answer.json --question "..." --answer-file answer.txt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import REPO_ROOT, get_settings  # noqa: E402
from app.core.corpus import load_chunks  # noqa: E402
from app.zk.attest import verify_attestation  # noqa: E402
from app.zk.commit import corpus_order  # noqa: E402
from app.zk.retrieval_circuit import retriever_from_rules  # noqa: E402
from app.zk.service import ZkService  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--facts", type=Path, help="JSON of disclosed facts to check against a verdict proof's commitment")
    ap.add_argument("--question", help="for a signed answer: the question text")
    ap.add_argument("--answer-file", type=Path, help="for a signed answer: the answer text")
    ap.add_argument("--no-embed", action="store_true", help="retrieval: skip recomputing the query embedding (no model download)")
    args = ap.parse_args()

    s = get_settings()
    published = s.zk_published_dir
    manifest = json.loads((published / "manifest.json").read_text(encoding="utf-8"))
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))

    if "signature" in bundle and "record" in bundle:
        answer = args.answer_file.read_text(encoding="utf-8") if args.answer_file else None
        res = {"kind": "attestation", **verify_attestation(bundle, manifest.get("signing_public_key"), args.question, answer)}
    else:
        embedder = None
        if bundle.get("kind") == "retrieval" and not args.no_embed:
            from app.core.embeddings import EmbeddingService

            embedder = EmbeddingService(manifest["corpus"]["embedding_model"])
        rules = manifest.get("retrieval", {}).get("rules")
        retriever = retriever_from_rules(rules, embedder) if rules else None
        # a directory without proving keys: verification uses the published keys only
        svc = ZkService(REPO_ROOT / "docs" / "zk" / ".no-proving-keys", published, retriever=retriever,
                        chunks=corpus_order(load_chunks(s.corpus_chunks_path)))
        if bundle.get("kind") == "verdict":
            facts = json.loads(args.facts.read_text(encoding="utf-8")) if args.facts else None
            res = svc.verify_verdict(bundle, facts)
        elif bundle.get("kind") == "retrieval":
            res = svc.verify_retrieval(bundle)
        else:
            raise SystemExit("not a verdict / retrieval proof or a signed answer")

    print(f"{res['kind']}: {'VALID' if res['valid'] else 'INVALID'}")
    for c in res["checks"]:
        print(f"  {'ok ' if c['ok'] else 'BAD'} {c['name']}" + (f" ({c['detail']})" if c.get("detail") else ""))
    if res.get("cited"):
        print(f"  proven: s.{res['cited']['section']} {res['cited']['status']}")
    if res.get("sources"):
        print("  proven sources: " + ", ".join(x["citation_path"].split(" > ")[-1] for x in res["sources"]))
    return 0 if res["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
