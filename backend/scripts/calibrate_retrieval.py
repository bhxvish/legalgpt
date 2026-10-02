"""Measure retrieval on in-scope vs out-of-scope questions to choose RETRIEVAL_MIN_SIMILARITY.

Prints top-1 similarity, whether the expected section is retrieved, and the range of floors
that separates the two sets (best out-of-scope vs worst in-scope top-1 score). Needs an ingested index.

Usage (from the repo root):  python backend/scripts/calibrate_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.core.embeddings import EmbeddingService  # noqa: E402
from app.core.retriever import QueryExpander  # noqa: E402
from app.core.vector_store import VectorStore  # noqa: E402

IN_SCOPE: list[tuple[str, str]] = [  # (question, section that should be retrieved)
    ("What is the punishment for causing death by negligence?", "304A"),
    ("What is the punishment for murder?", "302"),
    ("When is culpable homicide not murder?", "300"),
    ("What is dowry death?", "304B"),
    ("Define theft", "378"),
    ("What is the punishment for theft?", "379"),
    ("What is cheating?", "415"),
    ("What is the punishment for rash driving on a public road?", "279"),
    ("Is attempt to commit suicide an offence?", "309"),
    ("What is criminal breach of trust?", "405"),
    ("What amounts to defamation?", "499"),
    ("What is the right of private defence of the body?", "97"),
    ("What is criminal conspiracy?", "120A"),
    ("Cruelty by husband or his relatives towards a wife", "498A"),
    ("What is extortion?", "383"),
    ("Punishment for kidnapping a child", "363"),
]

OUT_OF_SCOPE: list[str] = [
    "What is the best recipe for chocolate cake?",
    "Who won the cricket world cup in 2011?",
    "How do I file for divorce in India?",
    "What is the capital gains tax rate on mutual funds?",
    "How do I register a property sale deed?",
    "How can I get anticipatory bail?",
    "What is the procedure to file a consumer complaint?",
    "How do I apply for a passport?",
    "What are my rights as a tenant if the landlord refuses to return my deposit?",
    "Explain how photosynthesis works",
    "What is the GST rate on restaurant bills?",
    "How do I write a Python function to sort a list?",
    "Can my employer fire me without notice?",
    "What does the Constitution say about freedom of speech?",
]


def main() -> int:
    s = get_settings()
    embedder = EmbeddingService(s.embedding_model)
    store = VectorStore(s.chroma_persist_dir, s.chroma_collection)
    expander = QueryExpander()
    if store.count() == 0:
        print("index is empty; run backend/scripts/ingest_corpus.py first")
        return 1

    def top(question: str) -> list:
        return store.query(embedder.embed([expander.expand(question)])[0], 8)

    print("IN-SCOPE   top1  hit@1 hit@6  question")
    in_scores, hit1, hit6 = [], 0, 0
    for q, want in IN_SCOPE:
        res = top(q)
        secs = [r.chunk.section for r in res]
        in_scores.append(res[0].similarity)
        hit1 += secs[0] == want
        hit6 += want in secs[:6]
        print(f"           {res[0].similarity:.3f}  {'Y' if secs[0] == want else '-':5} {'Y' if want in secs[:6] else '-':5} {q}")
    print("OUT-SCOPE  top1  nearest section")
    out_scores = []
    for q in OUT_OF_SCOPE:
        res = top(q)
        out_scores.append(res[0].similarity)
        print(f"           {res[0].similarity:.3f}  §{res[0].chunk.section:6} {q}")

    lo, hi = max(out_scores), min(in_scores)
    print(f"\nhit@1 {hit1}/{len(IN_SCOPE)}, hit@6 {hit6}/{len(IN_SCOPE)}")
    print(f"worst in-scope top1 = {hi:.3f}; best out-of-scope top1 = {lo:.3f}")
    if hi > lo:
        print(
            f"any RETRIEVAL_MIN_SIMILARITY in ({lo:.3f}, {hi:.3f}) separates this set "
            f"(current: {s.retrieval_min_similarity}). The floor also filters supporting chunks, "
            "so prefer the low end of the range to keep more of them."
        )
    else:
        print("no clean separation on this set — inspect the overlaps before choosing a floor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
