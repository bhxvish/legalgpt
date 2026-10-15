"""ZKML (app/zk): commitments, the verdict circuit's agreement with Prolog, retrieval re-selection,
signed answers, and - when EZKL and the built circuits are present - real proofs."""

import json
import random

import numpy as np
import pytest

from app.config import BACKEND_DIR, REPO_ROOT, get_settings
from app.core.models import Chunk
from app.core.retriever import Retriever
from app.core.vector_store import QueryResult
from app.verification.fact_schema import PREDICATES
from app.zk.attest import Attestor, verify_attestation
from app.zk.commit import MerkleTree, chunk_leaf, corpus_tree, rules_hash, verify_inclusion
from app.zk.retrieval_circuit import select
from app.zk.verdict_circuit import FactEncoder, VerdictLayout, validity_penalty

RULES = BACKEND_DIR / "app" / "verification" / "rules"
EVAL_FACTS = json.loads((BACKEND_DIR / "tests" / "fixtures" / "verification_eval_facts.json").read_text(encoding="utf-8"))["cases"]


def _prolog():
    try:
        from app.verification.prolog_engine import PrologEngine

        return PrologEngine(RULES)
    except Exception:
        return None


@pytest.fixture(scope="module")
def engine():
    e = _prolog()
    if e is None:
        pytest.skip("SWI-Prolog / pyswip not available")
    return e


@pytest.fixture(scope="module")
def layout(engine):
    return VerdictLayout.from_sections(engine.sections())


# ------------------------------------------------------------------ commitments
def test_merkle_inclusion_for_every_leaf_and_tamper_detection() -> None:
    for n in (1, 2, 5, 8, 13):
        leaves = [f"{i:064x}" for i in range(n)]
        tree = MerkleTree(leaves)
        for i, leaf in enumerate(leaves):
            assert verify_inclusion(leaf, tree.proof(i), tree.root)
        if n > 1:
            assert not verify_inclusion(leaves[0], tree.proof(1), tree.root)


def test_corpus_root_is_order_independent_and_text_sensitive() -> None:
    a = [Chunk("IPC:1:a", "IPC", "one", "IPC > Section 1", section="1"), Chunk("IPC:2:b", "IPC", "two", "IPC > Section 2", section="2")]
    assert corpus_tree(a).root == corpus_tree(list(reversed(a))).root  # committed order is by chunk_id
    b = [a[0], Chunk("IPC:2:b", "IPC", "two!", "IPC > Section 2", section="2")]
    assert corpus_tree(a).root != corpus_tree(b).root
    assert chunk_leaf(a[0]) != chunk_leaf(b[1])


def test_rules_hash_ignores_line_endings(tmp_path) -> None:
    (tmp_path / "x.pl").write_bytes(b"a.\nb.\n")
    lf = rules_hash(tmp_path)["sha256"]
    (tmp_path / "x.pl").write_bytes(b"a.\r\nb.\r\n")
    assert rules_hash(tmp_path)["sha256"] == lf


def test_published_manifest_matches_the_rules_in_the_repo() -> None:
    m = REPO_ROOT / "docs" / "zk" / "manifest.json"
    if not m.exists():
        pytest.skip("circuits not set up (scripts/zk_setup.py)")
    assert json.loads(m.read_text(encoding="utf-8"))["rules"]["sha256"] == rules_hash(RULES)["sha256"], \
        "rules changed since the circuits were built: re-run backend/scripts/zk_setup.py"


# ------------------------------------------------------------------ verdict circuit == Prolog
def _prolog_result(engine, facts: dict[str, str], case: str) -> dict[str, tuple[str, dict[str, str]]]:
    clauses = [f"fact('{case}', {k}, {v})" for k, v in facts.items()]
    out = {}
    with engine.session(case, clauses):
        for s in engine.sections():
            out[s.section] = (engine.verdict(case, s.section), engine.element_statuses(case, s.section))
    return out


def _random_facts(rng: random.Random) -> dict[str, str]:
    return {p.name: rng.choice(p.values) for p in PREDICATES}


def test_verdict_circuit_agrees_with_prolog(engine, layout) -> None:
    """The 15 extracted fact sets of the verification test set, plus 300 random ones: same verdict
    and same element statuses for every encoded section."""
    rng = random.Random(7)
    cases = [c["facts"] for c in EVAL_FACTS] + [_random_facts(rng) for _ in range(300)]
    enc = FactEncoder()
    for i, facts in enumerate(cases):
        circuit = layout.decode(layout.evaluate(enc.encode(facts)))
        prolog = _prolog_result(engine, facts, f"zk{i}")
        assert set(circuit) == set(prolog)
        for sec, (verdict, statuses) in prolog.items():
            assert circuit[sec]["status"] == verdict, (i, sec, facts)
            assert {e["predicate"]: e["status"] for e in circuit[sec]["elements"]} == statuses, (i, sec)


def test_eval_cases_get_the_verdict_prolog_gave(layout) -> None:
    enc = FactEncoder()
    for c in EVAL_FACTS:
        assert layout.decode(layout.evaluate(enc.encode(c["facts"])))[c["section"]]["status"] == c["prolog_verdict"], c["id"]


def test_invalid_encodings_are_rejected(layout) -> None:
    enc = FactEncoder()
    x = enc.encode({"act_type": "rash"})
    assert validity_penalty(layout.slots, x) == 0
    two_acts = list(x)
    two_acts[layout.slots.index("act_type=intentional")] = 1  # rash and intentional at once
    big = enc.encode({})
    big[layout.slots.index("caused_death")] = 5
    no_known = list(x)
    no_known[layout.slots.index("act_type:known")] = 0
    for bad in (two_acts, big, no_known):
        assert validity_penalty(layout.slots, bad) > 0
        with pytest.raises(ValueError, match="not a valid encoding"):
            layout.decode(layout.evaluate(bad))


def test_layout_round_trips_and_hash_is_stable(layout) -> None:
    again = VerdictLayout.from_dict(json.loads(json.dumps(layout.to_dict())))
    assert again.sha256 == layout.sha256 and again.W == layout.W


# ------------------------------------------------------------------ retrieval re-selection
class _ExactStore:
    """Exact cosine search over a small fixed corpus (stands in for Chroma)."""

    def __init__(self, chunks, E) -> None:
        self.chunks, self.E = chunks, E

    def query(self, embedding, k, where=None):
        sims = self.E @ np.asarray(embedding)
        idx = [i for i in np.argsort(-sims) if not where or self.chunks[i].section == where["section"]][:k]
        return [QueryResult(self.chunks[i], float(1 - sims[i])) for i in idx]


class _Embedder:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed(self, texts):
        out = []
        for t in texts:
            v = np.random.default_rng(abs(hash(t)) % 2**32).normal(size=self.dim)
            out.append(v / np.linalg.norm(v))
        return np.asarray(out)


def test_selection_from_exact_scores_reproduces_the_retriever() -> None:
    rng = np.random.default_rng(3)
    dim, n = 16, 40
    chunks = [Chunk(f"IPC:{i}:x", "IPC", f"text {i} " * (20 + i), f"IPC > Section {i}", section=str(i % 12),
                    role="Law" if i % 3 else "Explanation") for i in range(n)]
    E = rng.normal(size=(n, dim))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    emb = _Embedder(dim)
    retr = Retriever(emb, _ExactStore(chunks, E), min_similarity=0.0, top_k=8, max_chunks=6, max_context_chars=900)
    for q in ["What does Section 7 say?", "theft", "u/s 3 and section 11", "punishment for hurt"]:
        live = retr.retrieve(q)
        sims = E @ emb.embed([retr.expander.expand(q)])[0]
        assert [s.chunk_id for s in select(retr, q, chunks, sims)] == [s.chunk_id for s in live], q
    # an LLM rewrite: the proof covers question + disclosed rewrite, and must select what the chat did
    from app.core.query_rewriter import Rewrite

    rw = Rewrite(True, [("cheating", "5"), ("hurt", "9")])
    q = "someone tricked me out of my savings"
    live = retr.retrieve(q, rewrite=Rewrite.from_text(rw.text))
    query = f"{q}\n{rw.text}"
    sims = E @ emb.embed([retr.expander.search_query(query)])[0]
    assert [s.chunk_id for s in select(retr, query, chunks, sims)] == [s.chunk_id for s in live]
    assert any(s.section in {"5", "9"} for s in live)  # the rewrite's sections were fetched


# ------------------------------------------------------------------ attestation
def test_signed_answer_verifies_and_detects_tampering(tmp_path) -> None:
    att = Attestor(tmp_path / "k.pem", {"corpus_merkle_root": "ab" * 32})

    class S:
        marker, chunk_id = 1, "IPC:304A:x"

    signed = att.answer_record(question="q?", prompt=[{"role": "user", "content": "q?"}], answer="A [1]", sources=[S()],
                               model_id="m", refused=False, follow_up=False, leaf_of=lambda cid: "cd" * 32)
    ok = verify_attestation(signed, att.public_key, question="q?", answer="A [1]")
    assert ok["valid"], ok
    assert not verify_attestation(signed, att.public_key, answer="A different answer [1]")["valid"]
    tampered = json.loads(json.dumps(signed))
    tampered["record"]["model"] = "other"
    assert not verify_attestation(tampered, att.public_key)["valid"]
    assert not verify_attestation(signed, Attestor(tmp_path / "other.pem").public_key)["valid"]
    assert Attestor(tmp_path / "k.pem").public_key == att.public_key  # the key persists


# ------------------------------------------------------------------ real proofs (EZKL)
@pytest.fixture(scope="module")
def zk_service():
    pytest.importorskip("ezkl")
    from app.zk.service import ZkService

    s = get_settings()
    svc = ZkService(s.zk_dir, s.zk_published_dir)
    if not svc.verdict.ready_to_prove() or svc.layout is None:
        pytest.skip("verdict circuit not built (scripts/zk_setup.py)")
    return svc


def test_verdict_proof_end_to_end(zk_service) -> None:
    case = EVAL_FACTS[0]
    bundle = zk_service.prove_verdict(case["facts"], case["section"])
    res = zk_service.verify_verdict(bundle, case["facts"])
    assert res["valid"], res
    assert res["cited"]["status"] == case["prolog_verdict"]

    # disclosing different facts than the ones proven fails the commitment check
    other = dict(case["facts"], caused_death="false")
    assert not zk_service.verify_verdict(bundle, other)["valid"]

    # changing a public output (an element status) breaks the proof
    import ezkl

    forged = json.loads(json.dumps(bundle))
    first = int(round(ezkl.felt_to_float(forged["proof"]["instances"][0][1], 0)))
    forged["proof"]["instances"][0][1] = ezkl.float_to_felt(float(1 if first != 1 else -1), 0)
    assert not zk_service.verify_verdict(forged)["valid"]


def test_invalid_encoding_cannot_pass_verification(zk_service) -> None:
    """A dishonest prover feeding the circuit two act types gets a proof, but the verifier rejects it."""
    x = FactEncoder().encode({"caused_death": "true", "act_type": "rash", "intent_to_kill": "false",
                              "knowledge_likely_death": "false"})
    x[zk_service.layout.slots.index("act_type=intentional")] = 1
    try:
        out = zk_service.verdict.prove(x)
    except Exception:
        return  # also acceptable: the circuit refuses to prove it at all
    res = zk_service.verify_verdict({"kind": "verdict", "section": "304A", "proof": out["proof"]})
    assert not res["valid"]


# ------------------------------------------------------------------ API
def test_zk_api_signs_and_checks_answers(tmp_path) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.zk.router import ZkRouter
    from app.zk.service import ZkService

    svc = ZkService(tmp_path / "zk", tmp_path / "pub", attestor=Attestor(tmp_path / "k.pem"))
    app = FastAPI()
    app.include_router(ZkRouter(svc).router)
    client = TestClient(app)
    m = client.get("/api/zk/manifest").json()
    assert m["available"]["attestation"] and not m["available"]["verdict_prove"]
    assert m["signing_public_key"] == svc.attestor.public_key

    signed = svc.attestor.sign({"answer_sha256": "x"})
    assert client.post("/api/zk/verify", json={"bundle": signed}).json()["valid"]
    signed["record"]["answer_sha256"] = "y"
    assert not client.post("/api/zk/verify", json={"bundle": signed}).json()["valid"]
    # no circuits set up here: proving is a clear 503, not a crash
    r = client.post("/api/zk/prove/verdict", json={"facts": {}, "section": "304A"})
    assert r.status_code == 503 and "zk_setup" in r.json()["detail"]
    assert client.post("/api/zk/verify", json={"bundle": {"kind": "nonsense"}}).status_code == 422


def test_selection_handles_definition_questions_like_the_retriever() -> None:
    """The definition lookup (section titled with the asked term) is rebuilt from the committed chunks."""
    from app.core.retriever import title_index

    rng = np.random.default_rng(5)
    dim, n = 16, 30
    chunks = [Chunk(f"IPC:{i}:x", "IPC", f"text {i} " * 20, f"IPC > Section {i}", section=str(i), title=f"Offence {i}")
              for i in range(n)]
    E = rng.normal(size=(n, dim)); E /= np.linalg.norm(E, axis=1, keepdims=True)
    emb = _Embedder(dim)
    retr = Retriever(emb, _ExactStore(chunks, E), min_similarity=0.0, top_k=4, max_chunks=6, titles=title_index(chunks))
    q = "What is offence 17 under the IPC?"
    live = retr.retrieve(q)
    sims = E @ emb.embed([retr.expander.search_query(q)])[0]
    assert [s.chunk_id for s in select(retr, q, chunks, sims)] == [s.chunk_id for s in live]
    assert live[0].section == "17"
