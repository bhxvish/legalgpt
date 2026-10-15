"""ZkService: proofs for fact-check verdicts and retrieval, signed answer records, and the manifest
of commitments they refer to.

Layout:
  <ZK_DIR>/verdict/, <ZK_DIR>/retrieval/   EzklCircuit directories (proving keys: large, git-ignored)
  <ZK_DIR>/signing_key.pem                 the server's Ed25519 key (git-ignored)
  docs/zk/manifest.json                    published commitments (written by scripts/zk_setup.py)
  docs/zk/verdict/, docs/zk/retrieval/     published settings.json + vk.key (+ the verdict layout)

A proof bundle is {"kind", "version", "proof", ...}; verify() re-checks it from the bundle alone
plus the published verification keys - nothing the prover says about the result is trusted.
"""

import json
import logging
from pathlib import Path
from typing import Any

from app.zk.commit import chunk_leaf, corpus_tree, verify_inclusion
from app.zk.prover import EzklCircuit, ZkUnavailable
from app.zk.verdict_circuit import FactEncoder, VerdictLayout

logger = logging.getLogger("app.zk")
BUNDLE_VERSION = 1


class ZkService:
    def __init__(self, zk_dir: Path, published_dir: Path, retriever=None, chunks=None, attestor=None) -> None:
        self.zk_dir, self.published_dir = Path(zk_dir), Path(published_dir)
        self.retriever = retriever
        self.attestor = attestor
        self.encoder = FactEncoder()
        self.verdict = EzklCircuit(self.zk_dir / "verdict")
        self.retrieval = EzklCircuit(self.zk_dir / "retrieval")
        # verification needs only the published settings + vk; fall back to them if the local
        # circuit directory (with the proving key) does not exist on this machine
        self.verdict_v = self.verdict if self.verdict.ready_to_verify() else EzklCircuit(self.published_dir / "verdict")
        self.retrieval_v = self.retrieval if self.retrieval.ready_to_verify() else EzklCircuit(self.published_dir / "retrieval")
        layout_path = self.published_dir / "verdict" / "layout.json"
        self.layout = VerdictLayout.from_dict(json.loads(layout_path.read_text(encoding="utf-8"))) if layout_path.exists() else None
        self._chunks = list(chunks) if chunks is not None else None  # committed order
        self._tree = None

    # ---------------------------------------------------------------- manifest
    @property
    def manifest(self) -> dict[str, Any]:
        p = self.published_dir / "manifest.json"
        m = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        m["available"] = {"verdict_prove": self.verdict.ready_to_prove(), "verdict_verify": self.verdict_v.ready_to_verify(),
                          "retrieval_prove": self.retrieval.ready_to_prove() and self.retriever is not None,
                          "retrieval_verify": self.retrieval_v.ready_to_verify(), "attestation": self.attestor is not None}
        if self.attestor is not None:
            m["signing_public_key"] = self.attestor.public_key
        return m

    # ------------------------------------------------------------- corpus tree
    @property
    def chunks(self):
        if self._chunks is None:
            from app.zk.retrieval_circuit import chunk_matrix

            self._chunks, _ = chunk_matrix(self.retriever.store)
        return self._chunks

    @property
    def tree(self):
        if self._tree is None:
            self._tree = corpus_tree(self.chunks)
        return self._tree

    def leaf_of(self, chunk_id: str) -> str:
        c = next((c for c in self.chunks if c.chunk_id == chunk_id), None)
        return chunk_leaf(c) if c else ""

    def inclusion(self, chunk_id: str) -> dict[str, Any]:
        ids = [c.chunk_id for c in self.chunks]
        if chunk_id not in ids:
            raise KeyError(chunk_id)
        i = ids.index(chunk_id)
        leaf = chunk_leaf(self.chunks[i])
        path = self.tree.proof(i)
        return {"chunk_id": chunk_id, "leaf": leaf, "path": path, "root": self.tree.root,
                "valid": verify_inclusion(leaf, path, self.tree.root)}

    # -------------------------------------------------------------- attestation
    def attest_answer(self, *, question, prompt, answer, sources, model_id, refused, follow_up, rewrite=None) -> dict[str, Any]:
        m = self.manifest
        record_context = {"corpus_merkle_root": m.get("corpus", {}).get("merkle_root") or self.tree.root}
        adapter = m.get("adapter") or {}
        if "+lora" in model_id and adapter and model_id.endswith(f":{adapter.get('name')}"):
            record_context["adapter_sha256"] = adapter["sha256"]  # only for the adapter the manifest publishes
        elif "+gguf-adapter:" in model_id:  # a llama.cpp model: its GGUF adapter, if published
            gguf = (m.get("adapter_gguf") or {}).get(model_id.rsplit(":", 1)[-1])
            if gguf:
                record_context["adapter_sha256"] = gguf
        self.attestor.context = record_context
        return self.attestor.answer_record(question=question, prompt=prompt, answer=answer, sources=sources,
                                           model_id=model_id, refused=refused, follow_up=follow_up, leaf_of=self.leaf_of,
                                           rewrite=rewrite)

    # ----------------------------------------------------------------- verdict
    def prove_verdict(self, facts: dict[str, Any], section: str | None = None) -> dict[str, Any]:
        if self.layout is None:
            raise ZkUnavailable("the verdict circuit is not set up; run backend/scripts/zk_setup.py")
        x = self.encoder.encode(facts)
        out = self.verdict.prove(x)
        bundle = {"kind": "verdict", "version": BUNDLE_VERSION, "section": (section or "").upper() or None,
                  "layout_sha256": self.layout.sha256, "vk_sha256": self.verdict.describe()["vk_sha256"],
                  "prove_seconds": out["prove_seconds"], "proof": out["proof"]}
        return bundle

    def verify_verdict(self, bundle: dict[str, Any], facts: dict[str, Any] | None = None) -> dict[str, Any]:
        checks = []
        ok = self.verdict_v.verify(bundle["proof"])
        checks.append({"name": "proof verifies against the published verification key", "ok": ok})
        result: dict[str, Any] = {"kind": "verdict", "checks": checks}
        if not ok:
            result["valid"] = False
            return result
        pub = self.verdict_v.public_values(bundle["proof"])
        statuses = [int(round(v)) for v in pub["outputs"]]
        try:
            sections = self.layout.decode(statuses)
            checks.append({"name": "every element status is satisfied / violated / missing", "ok": True})
        except ValueError as e:
            checks.append({"name": "every element status is satisfied / violated / missing", "ok": False, "detail": str(e)})
            sections = {}
        result["facts_commitment"] = pub["input_hash"]
        result["sections"] = sections
        sec = bundle.get("section")
        if sec:
            result["cited"] = sections.get(sec)
            checks.append({"name": f"s.{sec} is an encoded section", "ok": sec in sections})
        if facts is not None:  # selective disclosure: these are the facts that were proven
            same = self.verdict_v.input_hash(self.encoder.encode(facts)) == pub["input_hash"]
            checks.append({"name": "the disclosed facts match the proof's commitment", "ok": same})
        result["valid"] = all(c["ok"] for c in checks)
        return result

    # --------------------------------------------------------------- retrieval
    def prove_retrieval(self, question: str, live_chunk_ids: list[str] | None = None,
                        rewrite: str | None = None) -> dict[str, Any]:
        """`rewrite`: the LLM rewrite the chat used ("cheating (section 420); ..."), if any. The proof
        then covers the search for question + rewrite; the rewrite itself is an unproven, disclosed
        input (an LLM produced it)."""
        from app.zk.retrieval_circuit import compare, query_vector, scope_vector, select

        if self.retriever is None:
            raise ZkUnavailable("retrieval proofs need the retriever")
        from app.core.query_rewriter import Rewrite

        query = f"{question}\n{rewrite}" if rewrite else question
        q = query_vector(self.retriever, query)
        scope_q = scope_vector(self.retriever, question)
        out = self.retrieval.prove(q)
        sims = self.retrieval.public_values(out["proof"])["outputs"]
        selected = select(self.retriever, query, self.chunks, sims, scope_q)
        if live_chunk_ids is None:
            rw = Rewrite.from_text(rewrite) if rewrite else None
            live_chunk_ids = [s.chunk_id for s in self.retriever.retrieve(question, rewrite=rw)]
        live = live_chunk_ids
        return {"kind": "retrieval", "version": BUNDLE_VERSION, "question": question,
                "rewrite": rewrite,  # unproven LLM input, disclosed; the proof covers question + rewrite
                "sources": [s.chunk_id for s in selected], "live_comparison": compare(selected, live),
                "scope_embedding": [round(v, 6) for v in scope_q],  # unproven; verifiers recompute it
                "corpus_root": self.tree.root, "vk_sha256": self.retrieval.describe()["vk_sha256"],
                "prove_seconds": out["prove_seconds"], "proof": out["proof"]}

    def verify_retrieval(self, bundle: dict[str, Any]) -> dict[str, Any]:
        from app.zk.retrieval_circuit import select

        checks = []
        ok = self.retrieval_v.verify(bundle["proof"])
        checks.append({"name": "proof verifies against the published verification key", "ok": ok})
        result: dict[str, Any] = {"kind": "retrieval", "checks": checks}
        if not ok:
            result["valid"] = False
            return result
        pub = self.retrieval_v.public_values(bundle["proof"])
        sims = pub["outputs"]
        checks.append({"name": "one proven score per committed chunk", "ok": len(sims) == len(self.chunks)})
        scope_q = bundle.get("scope_embedding")
        if getattr(self.retriever, "embedder", None) is not None:
            from app.zk.retrieval_circuit import scope_vector

            scope_q = scope_vector(self.retriever, bundle["question"])
        query = f"{bundle['question']}\n{bundle['rewrite']}" if bundle.get("rewrite") else bundle["question"]
        selected = select(self.retriever, query, self.chunks, sims, scope_q)
        ids = [s.chunk_id for s in selected]
        checks.append({"name": "the retrieval rules applied to the proven scores give exactly these sources",
                       "ok": ids == bundle.get("sources")})
        if getattr(self.retriever, "embedder", None) is not None:  # recompute the (unproven) query embedding
            from app.zk.retrieval_circuit import query_vector

            q = query_vector(self.retriever, query)
            diff = max(abs(a - b) for a, b in zip(q, pub["inputs"]))
            checks.append({"name": "the proven query embedding is this question's"
                           + (" (with the disclosed rewrite)" if bundle.get("rewrite") else ""), "ok": diff < 2e-3,
                           "detail": f"max difference {diff:.5f}"})
        published_root = self.manifest.get("corpus", {}).get("merkle_root")
        checks.append({"name": "corpus root matches the published manifest", "ok": bundle.get("corpus_root") == published_root})
        checks.append({"name": "the corpus used to re-run the rules is the committed one", "ok": self.tree.root == published_root})
        result["sources"] = [{"chunk_id": s.chunk_id, "citation_path": s.citation_path, "similarity": s.similarity} for s in selected]
        result["valid"] = all(c["ok"] for c in checks)
        return result
