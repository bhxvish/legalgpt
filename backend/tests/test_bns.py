from types import SimpleNamespace

from app.core.bns import annotate, bns_for, source


def test_known_ipc_to_bns_mappings_from_the_official_table():
    assert bns_for("302") == ["103"]  # murder
    assert bns_for("379") == ["303"]  # theft
    assert bns_for("420") == ["318"]  # cheating
    assert bns_for("304b") == ["80"]  # dowry death (case-insensitive)
    assert bns_for("34") == ["3(5)"]  # common intention
    assert bns_for("124A") == []  # sedition: no direct equivalent
    assert "Bureau of Police Research and Development" in source()


def test_annotate_marks_statute_sources_only():
    statute = SimpleNamespace(kind="statute", section="302", bns=[])
    repealed = SimpleNamespace(kind="statute", section="497", bns=[])
    judgment = SimpleNamespace(kind="judgment", section="", bns=[])
    annotate([statute, repealed, judgment])
    assert statute.bns == ["103"] and repealed.bns == ["none"] and judgment.bns == []


def test_choose_law_by_name_and_date():
    from app.core.bns import choose_law

    assert choose_law("What is the punishment for murder?")[0] == "ipc"
    assert choose_law("Punishment for murder under the BNS?")[0] == "bns"
    assert choose_law("On 3 March 2025 my bike was stolen.")[0] == "bns"
    assert choose_law("On 12 August 2024 he hit me.")[0] == "bns"
    assert choose_law("In June 2024 he cheated me.")[0] == "ipc"
    assert choose_law("What does section 420 IPC say about 2025?")[0] == "ipc"  # naming the code wins
    assert choose_law("anything", "bns") == ("bns", "chosen")


def test_reverse_map():
    from app.core.bns import base, ipc_for

    assert ipc_for("103") == ["302"] and "420" in ipc_for("318") and ipc_for("111") == []  # 318: 415, 417, 418, 420
    assert base("3(5)") == "3"


def test_a_bns_question_is_answered_from_the_bns_index(fake_embedder, memory_store):
    import json

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.core.chat_router import ChatRouter
    from app.core.models import Chunk, Message
    from app.core.query_rewriter import QueryRewriter
    from app.core.retriever import Retriever
    import uuid

    import chromadb

    from app.core.vector_store import VectorStore
    from tests.conftest import StubLLM

    class Scope:
        threshold = 0.5

        def probability(self, _):
            return 0.9

    ipc = Chunk(chunk_id="IPC:420:a", doc_id="IPC", section="420", title="Cheating", citation_path="IPC > Section 420",
                text="Section 420. Cheating and dishonestly inducing delivery of property.")
    bns = Chunk(chunk_id="BNS:318:a", doc_id="BNS", section="318", title="Cheating", citation_path="BNS > Chapter XVII > Section 318",
                text="Section 318. Cheating. Whoever cheats shall be punished.")
    memory_store.upsert([ipc], fake_embedder.embed([ipc.text]))
    bns_store = VectorStore(client=chromadb.EphemeralClient(), collection_name=f"test-bns-{uuid.uuid4().hex[:8]}")
    bns_store.upsert([bns], fake_embedder.embed([bns.text]))
    mk = lambda store: Retriever(fake_embedder, store, min_similarity=0.9, scope=Scope(), scope_floor=0.9)  # noqa: E731

    class Model(StubLLM):
        def generate(self, messages: list[Message]) -> str:
            self.calls.append(messages)
            return json.dumps({"criminal": True, "offences": [{"name": "Cheating", "section": "420"}]})

    llm = Model(reply="This is cheating under section 318 of the BNS [1].")
    app = FastAPI()
    app.include_router(ChatRouter(mk(memory_store), llm, rewriter=QueryRewriter(), bns_retriever=mk(bns_store)).router)
    body = TestClient(app).post("/api/chat", json={"question": "On 5 January 2025 a man sold me a fake phone online."}).text
    events = {}
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events[lines["event"]] = json.loads(lines["data"])
    assert events["retrieval"]["law"] == "bns"
    assert [s["section"] for s in events["sources"]] == ["318"] and "420" in events["sources"][0]["ipc"]
    assert "Bharatiya Nyaya Sanhita" in llm.calls[-1][0].content  # the answer prompt names the BNS
