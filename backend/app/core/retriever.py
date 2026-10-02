"""QueryExpander and Retriever: question -> ranked, bounded SourceEvidence list."""

import re
from dataclasses import dataclass

from app.core.embeddings import Embedder
from app.core.models import Chunk, SourceEvidence
from app.core.vector_store import VectorStore

DEFAULT_ROLE_PREF: dict[str, float] = {"Law": 1.2, "Ruling": 1.1}

# Phase 1: a small hand-written dictionary. Keys are matched case-insensitively on word boundaries.
LEGAL_ABBREVIATIONS: dict[str, str] = {
    "IPC": "Indian Penal Code",
    "CrPC": "Code of Criminal Procedure",
    "Cr.P.C.": "Code of Criminal Procedure",
    "BNS": "Bharatiya Nyaya Sanhita",
    "u/s": "under section",
    "r/w": "read with",
    "sec.": "section",
    "secs.": "sections",
    "FIR": "first information report",
    "SC": "Supreme Court",
    "HC": "High Court",
    "PP": "public prosecutor",
    "w.r.t.": "with respect to",
}

_SECTION_REF = re.compile(
    r"(?:\bsections?|\bsecs?\.?|\bs\.|\bu/s\.?)\s*(\d{1,3}[a-z]{0,2})\b"
    r"|\b(\d{1,3}[a-z]{0,2})\s*(?:of\s+(?:the\s+)?)?(?:ipc|indian penal code)\b",
    re.IGNORECASE,
)
# What follows a section number when it belongs to a different law: "(1) of the Code of Criminal
# Procedure", " Cr.P.C.", " of the POCSO Act", " of the Arms Act, 1959".
_OTHER_ENACTMENT = re.compile(
    r"(?:\s*\(\w+\))*\s*(?:,\s*)?(?:of\s+(?:the\s+)?)?"
    r"(?:code\s+of\s+criminal\s+procedure|cr\.?\s?p\.?\s?c|crpc|bnss|bns\b|bharatiya|evidence\s+act|constitution"
    r"|(?!indian\s+penal)[a-z][\w.()\-]*(?:\s+[a-z][\w.()\-]*){0,6}\s+act\b)",
    re.IGNORECASE,
)


class QueryExpander:
    def __init__(self, abbreviations: dict[str, str] | None = None) -> None:
        abbreviations = abbreviations or LEGAL_ABBREVIATIONS
        self._patterns = [
            (re.compile(rf"(?<![\w/]){re.escape(abbr)}(?![\w/])", re.IGNORECASE), full)
            for abbr, full in sorted(abbreviations.items(), key=lambda kv: -len(kv[0]))
        ]

    def expand(self, question: str) -> str:
        expanded = question
        for pattern, full in self._patterns:
            expanded = pattern.sub(full, expanded)
        return expanded

    @staticmethod
    def section_refs(question: str) -> list[str]:
        """IPC section numbers explicitly mentioned, e.g. "u/s 304a IPC" -> ["304A"]. A number
        that belongs to another enactment ("Section 378(1) of the Code of Criminal Procedure",
        "s. 6 of the POCSO Act") is not an IPC reference and is skipped."""
        refs: list[str] = []
        for m in _SECTION_REF.finditer(question):
            if m.group(1) and _OTHER_ENACTMENT.match(question, m.end()):
                continue
            ref = (m.group(1) or m.group(2)).upper()
            if ref not in refs:
                refs.append(ref)
        return refs


@dataclass
class Candidate:
    chunk: Chunk
    similarity: float
    score: float = 0.0
    explicit: bool = False  # matched a section number the user named


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        expander: QueryExpander | None = None,
        top_k: int = 8,
        min_similarity: float = 0.55,
        role_pref: dict[str, float] | None = None,
        max_chunks: int = 6,
        max_context_chars: int = 6000,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.expander = expander or QueryExpander()
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.role_pref = role_pref if role_pref is not None else dict(DEFAULT_ROLE_PREF)
        self.max_chunks = max_chunks
        self.max_context_chars = max_context_chars

    def retrieve(self, question: str, context: str | None = None) -> list[SourceEvidence]:
        """`context` is the previous user turn, if any. Follow-ups ("within how many years?")
        rarely retrieve well alone, so the question is also searched with the context prepended;
        both result sets are merged, so a change of topic still matches on its own words."""
        queries = [question] + ([f"{context}\n{question}"] if context else [])
        embeddings = self.embedder.embed([self.expander.expand(q) for q in queries])
        cands: list[Candidate] = []
        for embedding in embeddings:
            cands += [Candidate(r.chunk, r.similarity) for r in self.store.query(embedding, self.top_k)]
        for section in self.expander.section_refs(question):
            for r in self.store.query(embeddings[0], k=3, where={"section": section}):
                cands.append(Candidate(r.chunk, r.similarity, explicit=True))
        return self.assemble_context(self.rerank(cands), self.max_chunks, self.max_context_chars)

    def rerank(self, cands: list[Candidate]) -> list[Candidate]:
        """Score = similarity x role preference; explicit section matches rank first.
        Duplicate chunks are merged, keeping the explicit flag."""
        merged: dict[str, Candidate] = {}
        for c in cands:
            prev = merged.get(c.chunk.chunk_id)
            if prev is None:
                merged[c.chunk.chunk_id] = c
            else:
                prev.explicit = prev.explicit or c.explicit
                prev.similarity = max(prev.similarity, c.similarity)
        for c in merged.values():
            c.score = c.similarity * self.role_pref.get(c.chunk.role, 1.0)
        return sorted(merged.values(), key=lambda c: (c.explicit, c.score), reverse=True)

    def assemble_context(
        self, cands: list[Candidate], max_chunks: int = 6, max_context_chars: int = 6000
    ) -> list[SourceEvidence]:
        """Bounded evidence list. Returns [] when no candidate clears the similarity floor,
        which ChatRouter turns into a scope-boundary refusal."""
        evidence: list[SourceEvidence] = []
        used = 0
        for c in cands:
            if len(evidence) >= max_chunks:
                break
            if c.similarity < self.min_similarity and not c.explicit:
                continue
            if used + len(c.chunk.text) > max_context_chars:
                continue
            used += len(c.chunk.text)
            evidence.append(
                SourceEvidence(
                    marker=len(evidence) + 1,
                    chunk_id=c.chunk.chunk_id,
                    doc_id=c.chunk.doc_id,
                    citation_path=c.chunk.citation_path,
                    text=c.chunk.text,
                    similarity=round(c.similarity, 4),
                    score=round(c.score, 4),
                    role=c.chunk.role,
                    section=c.chunk.section,
                    title=c.chunk.title,
                )
            )
        return evidence
