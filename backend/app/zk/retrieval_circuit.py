"""Verifiable retrieval: prove the similarity of a question to *every* committed IPC chunk, then
re-run the retriever's own selection on the proven scores.

The circuit is q (1 x 384, public) -> E @ q (1 x n_chunks, public), with the chunk embedding
matrix E baked in as constants (so the verification key commits to it), in the committed chunk
order (commit.corpus_order). Fixed point at scale 12: scores are exact to about 0.0005.

Anyone holding the proof can check that the cited sources are what the retrieval rules select
from these scores - the top-k by similarity, explicit section matches, the role preference, the
0.55 floor and the size budget - so no relevant section was dropped and none was added by hand.
Scope: single questions (a follow-up's merged context search is not covered), and the query
embedding itself (MiniLM) is a public input, not proven; a verifier can recompute it.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from app.core.models import Chunk, SourceEvidence
from app.core.lay_terms import statutory_terms
from app.core.retriever import Candidate, Retriever

SCALE = 12


def chunk_matrix(store) -> tuple[list[Chunk], np.ndarray]:
    """All chunks in the vector store with their stored embeddings, in committed order."""
    res = store._call(lambda col: col.get(include=["documents", "metadatas", "embeddings"]))
    rows = sorted(zip(res["ids"], res["documents"], res["metadatas"], res["embeddings"]), key=lambda r: r[0])
    chunks = [store._to_chunk(cid, doc, meta) for cid, doc, meta, _ in rows]
    return chunks, np.asarray([r[3] for r in rows], dtype=np.float32)


def query_vector(retriever: Retriever, question: str) -> list[float]:
    """The embedding the retrieval circuit proves scores for (the search query)."""
    return [float(v) for v in retriever.embedder.embed([retriever.expander.search_query(question)])[0]]


def scope_vector(retriever: Retriever, question: str) -> list[float]:
    """The embedding the scope check reads (the plain question). Not proven: recomputed by verifiers."""
    return [float(v) for v in retriever.embedder.embed([retriever.expander.expand(question)])[0]]


def select(retriever: Retriever, question: str, chunks: Sequence[Chunk], sims: Sequence[float],
           scope_embedding: Sequence[float] | None = None) -> list[SourceEvidence]:
    """Retriever.retrieve_with_diagnostics for one question, with exact similarities over all
    chunks instead of the approximate index: the same candidate rules, floor (including the scope
    check, from the plain question's embedding), rerank and assembly."""
    order = sorted(range(len(chunks)), key=lambda i: sims[i], reverse=True)
    cands = [Candidate(chunks[i], float(sims[i])) for i in order[: retriever.top_k]]
    from app.core.retriever import defined_sections, title_index

    # the definition-question lookup, rebuilt from the committed chunks (as the live retriever does)
    named = retriever.expander.section_refs(question) + defined_sections(question, title_index(chunks))
    for section in named:
        same = [i for i in order if chunks[i].section == section][:3]
        cands += [Candidate(chunks[i], float(sims[i]), explicit=True) for i in same]
    ranked = retriever.rerank(cands)
    floor = retriever.min_similarity
    if scope_embedding is not None:
        floor, _ = retriever.floor_for(retriever.best_substantive(ranked), scope_embedding,
                                       lexicon=bool(statutory_terms(question)))
    return retriever.assemble_context(ranked, retriever.max_chunks, retriever.max_context_chars, floor=floor)


def retrieval_rules(retriever: Retriever) -> dict[str, Any]:
    """The selection parameters a verifier must use to re-run the rules (published in the manifest)."""
    scope = retriever.scope
    return {"top_k": retriever.top_k, "min_similarity": retriever.min_similarity, "role_pref": retriever.role_pref,
            "boilerplate_pref": retriever.boilerplate_pref, "max_chunks": retriever.max_chunks,
            "max_context_chars": retriever.max_context_chars, "explicit_section_matches": 3,
            "scope_floor": retriever.scope_floor, "scope_model": scope.to_dict() if scope is not None else None}


def retriever_from_rules(rules: dict[str, Any], embedder=None) -> Retriever:
    """A Retriever with no index, for re-running the selection on proven scores."""
    import numpy as np

    from app.core.scope import ScopeCheck

    sm = rules.get("scope_model")
    scope = ScopeCheck(np.asarray(sm["weights"]), sm["bias"], sm["threshold"], sm.get("embedding_model", "")) if sm else None
    return Retriever(embedder, None, top_k=rules["top_k"], min_similarity=rules["min_similarity"],
                     role_pref=rules["role_pref"], max_chunks=rules["max_chunks"], max_context_chars=rules["max_context_chars"],
                     scope=scope, scope_floor=rules.get("scope_floor", 0.40), boilerplate_pref=rules.get("boilerplate_pref", {}))


def torch_module(E: np.ndarray):
    import torch

    class SimilarityModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("E", torch.tensor(E, dtype=torch.float32))

        def forward(self, q):
            return q @ self.E.T

    return SimilarityModule()


def compare(selected: Sequence[SourceEvidence], live: Sequence[Any]) -> dict[str, Any]:
    """Do the proven selection and what the chat actually cited agree (same chunks, same order)?"""
    a = [s.chunk_id for s in selected]
    b = [s if isinstance(s, str) else s.chunk_id for s in live]
    return {"match": a == b, "proven": a, "live": b,
            "missing_from_live": [c for c in a if c not in b], "extra_in_live": [c for c in b if c not in a]}
