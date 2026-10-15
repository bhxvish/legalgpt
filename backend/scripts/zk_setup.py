"""Build the ZKML circuits and publish the commitments they refer to.

  1. commitments  Merkle root of the IPC chunks, hash of their embeddings, hashes of the Prolog
                  rules and of the newest LoRA adapter
  2. verdict      circuit for the fact check (element matrix from the rules), scale 0
  3. retrieval    circuit for question-to-chunk similarity over all chunks, scale 12
  4. publish      docs/zk/manifest.json, plus settings.json + vk.key per circuit (and the verdict
                  layout) - everything a third party needs to verify a proof
  5. self-test    one proof of each kind, proved and verified; timings go into the manifest

Needs: pip install -r backend/requirements-zk.txt, SWI-Prolog, and the vector index (the backend
builds it on first start). Takes a few minutes; writes ~1.3 GB of proving keys to data/zk/.

Usage (from the repo root):
    python backend/scripts/zk_setup.py
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import REPO_ROOT, get_settings  # noqa: E402
from app.core.corpus import load_chunks  # noqa: E402
from app.core.embeddings import EmbeddingService  # noqa: E402
from app.core.llm_client import latest_adapter  # noqa: E402
from app.core.retriever import build_retriever  # noqa: E402
from app.core.vector_store import VectorStore  # noqa: E402
from app.verification.prolog_engine import PrologEngine  # noqa: E402
from app.zk import retrieval_circuit, verdict_circuit  # noqa: E402
from app.zk.retrieval_circuit import retrieval_rules  # noqa: E402
from app.zk.attest import Attestor  # noqa: E402
from app.zk.commit import adapter_hash, corpus_tree, embeddings_hash, rules_hash  # noqa: E402
from app.zk.prover import EzklCircuit  # noqa: E402
from app.zk.service import ZkService  # noqa: E402

# the verdict circuit only handles small integers: a small lookup table keeps it at 2^13 rows
VERDICT_LOGROWS, VERDICT_LOOKUP = 13, (-2048, 2048)
SELF_TEST_QUESTION = "What is the punishment for causing death by negligence?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-retrieval", action="store_true", help="only build the verdict circuit")
    ap.add_argument("--rules-only", action="store_true",
                    help="only refresh the published retrieval rules (e.g. a retrained scope check); the circuits, "
                         "keys and corpus commitments are unchanged, so nothing is rebuilt")
    ap.add_argument("--adapter-only", action="store_true",
                    help="only refresh the published LoRA adapter fingerprint (after changing ADAPTER_PATH); signed "
                         "answers from that adapter then carry it. Circuits and keys are unchanged")
    args = ap.parse_args()
    s = get_settings()
    if args.adapter_only:
        path = s.zk_published_dir / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        adapter = s.adapter_path or latest_adapter(s.lora_dir)
        if adapter is None:
            raise SystemExit("no adapter: set ADAPTER_PATH in .env")
        old = (manifest.get("adapter") or {}).get("name")
        manifest["adapter"] = adapter_hash(adapter)
        manifest["adapter_updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        print(f"published adapter fingerprint: {old} -> {manifest['adapter']['name']} ({manifest['adapter']['sha256'][:16]}...)")
        return 0
    if args.rules_only:
        path = s.zk_published_dir / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        old = manifest["retrieval"]["rules"]
        manifest["retrieval"]["rules"] = retrieval_rules(build_retriever(s))
        manifest["retrieval"]["rules_updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        changed = [k for k in manifest["retrieval"]["rules"] if manifest["retrieval"]["rules"][k] != old.get(k)]
        print(f"updated the retrieval rules in {path} (changed: {', '.join(changed) or 'nothing'})")
        return 0
    import ezkl

    zk, pub = s.zk_dir, s.zk_published_dir
    manifest: dict = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "ezkl_version": getattr(ezkl, "__version__", "unknown"), "proof_system": "Halo2 (KZG), via EZKL"}

    # 1. commitments ---------------------------------------------------------
    store = VectorStore(s.chroma_persist_dir, s.chroma_collection)
    chunks, E = retrieval_circuit.chunk_matrix(store)
    corpus_ids = {c.chunk_id for c in load_chunks(s.corpus_chunks_path)}
    if {c.chunk_id for c in chunks} != corpus_ids:
        raise SystemExit("the vector index does not match data/corpus/ipc_chunks.jsonl; rebuild it first")
    tree = corpus_tree(chunks)
    manifest["corpus"] = {"chunks": len(chunks), "merkle_root": tree.root, "leaf": "sha256(0x00 || canonical JSON of the chunk)",
                          "embedding_model": s.embedding_model, "embeddings_sha256": embeddings_hash(E)}
    manifest["rules"] = rules_hash(s.prolog_rules_dir)
    adapter = s.adapter_path or latest_adapter(s.lora_dir)
    if adapter:
        manifest["adapter"] = adapter_hash(adapter)
    print(f"corpus root {tree.root[:16]}…  ({len(chunks)} chunks)")

    # 2. verdict circuit -----------------------------------------------------
    layout = verdict_circuit.VerdictLayout.from_sections(PrologEngine(s.prolog_rules_dir).sections())
    vc = EzklCircuit(zk / "verdict")
    t = vc.build(verdict_circuit.torch_module(layout), [[0] * len(layout.slots)], "hashed", scale=0,
                 logrows=VERDICT_LOGROWS, lookup_range=VERDICT_LOOKUP)
    manifest["verdict"] = {"elements": len(layout.rows), "slots": len(layout.slots), "sections": layout.sections(),
                           "layout_sha256": layout.sha256, **vc.describe(), "build": t}
    print(f"verdict circuit: {len(layout.rows)} elements x {len(layout.slots)} slots, logrows {manifest['verdict']['logrows']}, {t}")

    # 3. retrieval circuit ---------------------------------------------------
    if not args.skip_retrieval:
        rc = EzklCircuit(zk / "retrieval")
        t = rc.build(retrieval_circuit.torch_module(E), [[0.0] * E.shape[1]], "public", scale=retrieval_circuit.SCALE)
        manifest["retrieval"] = {"chunks": len(chunks), "dims": int(E.shape[1]), **rc.describe(), "build": t,
                                 "scope": "single questions; the query embedding is public, not proven",
                                 "rules": retrieval_rules(build_retriever(s))}
        print(f"retrieval circuit: {len(chunks)} x {E.shape[1]}, logrows {manifest['retrieval']['logrows']}, {t}")

    # 4. publish -------------------------------------------------------------
    for name in ("verdict", "retrieval"):
        src = zk / name
        if not (src / "vk.key").exists():
            continue
        (pub / name).mkdir(parents=True, exist_ok=True)
        for f in ("settings.json", "vk.key"):
            shutil.copy2(src / f, pub / name / f)
    (pub / "verdict" / "layout.json").write_text(json.dumps(layout.to_dict(), indent=1), encoding="utf-8")
    attestor = Attestor(zk / "signing_key.pem")
    manifest["signing_public_key"] = attestor.public_key
    (pub / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    # 5. self-test -----------------------------------------------------------
    retriever = build_retriever(s)
    svc = ZkService(zk, pub, retriever=retriever, chunks=chunks, attestor=attestor)
    facts = {"caused_death": "true", "act_type": "rash", "intent_to_kill": "false", "knowledge_likely_death": "false"}
    b = svc.prove_verdict(facts, "304A")
    t0 = time.perf_counter()
    v = svc.verify_verdict(b, facts)
    bench = {"verdict_prove_s": b["prove_seconds"], "verdict_verify_s": round(time.perf_counter() - t0, 2),
             "verdict_proof_kb": round(len(json.dumps(b["proof"])) / 1024)}
    assert v["valid"] and v["cited"]["status"] == "CONSISTENT", v
    print(f"self-test verdict: {v['cited']['status']}, valid={v['valid']}")
    if not args.skip_retrieval:
        b = svc.prove_retrieval(SELF_TEST_QUESTION)
        t0 = time.perf_counter()
        v = svc.verify_retrieval(b)
        bench |= {"retrieval_prove_s": b["prove_seconds"], "retrieval_verify_s": round(time.perf_counter() - t0, 2),
                  "retrieval_proof_kb": round(len(json.dumps(b["proof"])) / 1024),
                  "retrieval_matches_live": b["live_comparison"]["match"]}
        assert v["valid"], v
        print(f"self-test retrieval: valid={v['valid']}, matches live retrieval={b['live_comparison']['match']}")
    manifest["benchmarks"] = bench | {"machine": "measured by zk_setup.py on the machine that built the circuits"}
    (pub / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {(pub / 'manifest.json').relative_to(REPO_ROOT)}: {json.dumps(bench)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
