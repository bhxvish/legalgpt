"""QueryExpander and Retriever: question -> ranked, bounded SourceEvidence list."""

import re
from dataclasses import dataclass

from app.core.embeddings import Embedder
from app.core.lay_terms import statutory_terms
from app.core.models import Chunk, SourceEvidence
from app.core.vector_store import VectorStore

DEFAULT_ROLE_PREF: dict[str, float] = {"Law": 1.2, "Ruling": 1.1}
# Chunks that match vague questions on generic wording but are rarely the answer: state amendments
# and the general provisions of Chapter I (title, extent, punishment of offences within India).
# Ignored for sections the user names.
DEFAULT_BOILERPLATE_PREF: dict[str, float] = {"state_amendment": 0.85, "chapter_i": 0.8}

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

    def search_query(self, question: str) -> str:
        """What is embedded for retrieval: the expanded question, plus the statutory wording of
        any everyday offence description it contains (app/core/lay_terms.py)."""
        terms = statutory_terms(question)
        return self.expand(question) + ("\n" + "; ".join(terms) if terms else "")

    @staticmethod
    def is_follow_up(question: str) -> bool:
        """Does the question depend on the previous one? Pronouns ("what is the punishment for
        it?"), connective openers ("and if he was drunk?") and fragments ("within how many
        years?") do; a self-contained question ("How do I file my income tax return online?") does
        not, and searching it with the previous question's words would retrieve that topic's
        sections for an unrelated question. A question naming a section stands on its own."""
        words = re.findall(r"[a-z0-9']+", question.lower())
        if not words or QueryExpander.section_refs(question):
            return False
        if len(words) <= _FOLLOW_UP_MAX_WORDS:
            return True
        if any(" ".join(words[:n]) in _FOLLOW_UP_OPENERS for n in (1, 2, 3)):
            return True
        # a longer question is a self-contained narrative: its pronouns refer to its own people
        return len(words) <= _ANAPHORA_MAX_WORDS and any(w in _ANAPHORA for w in words)

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


_REFUSE = 1.01  # a floor no similarity reaches (explicit section matches still pass)
_FOLLOW_UP_MAX_WORDS = 5
_ANAPHORA_MAX_WORDS = 12
_FOLLOW_UP_OPENERS = {"and", "but", "also", "so", "then", "or", "what about", "how about", "what if", "and if",
                      "in that case", "in such a case", "if so",
                      # fragments that continue the previous question: "within how many years of marriage?"
                      "within", "for", "after", "before", "under", "with", "without", "during", "by", "upto", "till"}
_ANAPHORA = {"it", "its", "this", "that", "these", "those", "such", "same", "above", "aforesaid", "said",
             "he", "she", "him", "her", "his", "they", "them", "their"}


@dataclass
class Candidate:
    chunk: Chunk
    similarity: float
    score: float = 0.0
    explicit: bool = False  # matched a section number the user named


@dataclass
class RetrievalResult:
    sources: list[SourceEvidence]
    best_similarity: float  # over every candidate considered, including those below the floor
    min_similarity: float  # the floor in force
    candidates: int
    used_context: bool = False  # the previous question was also searched (a follow-up)
    scope_probability: float | None = None  # set when the scope check decided the floor (see scope.py)


def build_retriever(settings, embedder: Embedder | None = None, scope: bool = True) -> "Retriever":
    """The app's retriever: index, floors and (if trained for this embedding model) the scope check."""
    import logging

    from app.core.embeddings import EmbeddingService
    from app.core.scope import ScopeCheck

    check = ScopeCheck.load() if scope else None
    if check is not None and check.embedding_model != settings.embedding_model:
        logging.getLogger(__name__).warning(
            "scope check was trained for %s, not %s: disabled (re-run scripts/train_scope.py)",
            check.embedding_model, settings.embedding_model)
        check = None
    return Retriever(
        embedder=embedder or EmbeddingService(settings.embedding_model),
        store=VectorStore(settings.chroma_persist_dir, settings.chroma_collection),
        min_similarity=settings.retrieval_min_similarity,
        scope=check,
        scope_floor=settings.retrieval_scope_floor,
    )


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
        scope=None,
        scope_floor: float = 0.40,
        boilerplate_pref: dict[str, float] | None = None,
    ) -> None:
        """`scope` (app.core.scope.ScopeCheck): when nothing clears `min_similarity` but the
        question is about an IPC offence, sources down to `scope_floor` are used instead - for
        everyday fact patterns, which score lower against statute text than legal questions do."""
        self.embedder = embedder
        self.store = store
        self.expander = expander or QueryExpander()
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.role_pref = role_pref if role_pref is not None else dict(DEFAULT_ROLE_PREF)
        self.max_chunks = max_chunks
        self.max_context_chars = max_context_chars
        self.scope = scope
        self.scope_floor = scope_floor
        self.boilerplate_pref = boilerplate_pref if boilerplate_pref is not None else dict(DEFAULT_BOILERPLATE_PREF)

    def retrieve(self, question: str, context: str | None = None) -> list[SourceEvidence]:
        """`context` is the previous user turn, if any. Follow-ups ("within how many years?")
        rarely retrieve well alone, so the question is also searched with the context prepended;
        both result sets are merged, so a change of topic still matches on its own words."""
        return self.retrieve_with_diagnostics(question, context).sources

    def retrieve_with_diagnostics(self, question: str, context: str | None = None) -> "RetrievalResult":
        """retrieve(), plus the best similarity among *all* candidates — also when none cleared
        the floor and the answer will be a refusal (the explainability layer reports it)."""
        use_context = bool(context) and self.expander.is_follow_up(question)
        queries = [question] + ([f"{context}\n{question}"] if use_context else [])
        # the scope check reads the plain question: the lexicon's added wording would make
        # civil questions ("my order arrived damaged") look like crimes
        texts = [self.expander.search_query(q) for q in queries] + [self.expander.expand(question)]
        *embeddings, scope_embedding = list(self.embedder.embed(texts))
        cands: list[Candidate] = []
        for embedding in embeddings:
            cands += [Candidate(r.chunk, r.similarity) for r in self.store.query(embedding, self.top_k)]
        for section in self.expander.section_refs(question):
            for r in self.store.query(embeddings[0], k=3, where={"section": section}):
                cands.append(Candidate(r.chunk, r.similarity, explicit=True))
        ranked = self.rerank(cands)
        best = max((c.similarity for c in ranked), default=0.0)
        floor, scope_p = self.floor_for(self.best_substantive(ranked), scope_embedding,
                                        lexicon=bool(statutory_terms(question)))
        return RetrievalResult(
            sources=self.assemble_context(ranked, self.max_chunks, self.max_context_chars, floor=floor),
            best_similarity=best,
            min_similarity=floor,
            candidates=len(ranked),
            used_context=use_context,
            scope_probability=scope_p,
        )

    def floor_for(self, best_similarity: float, scope_embedding, lexicon: bool = False) -> tuple[float, float | None]:
        """The similarity floor for this question, and the scope probability if it was asked.

        Without lexicon wording: the usual floor, and below it the scope check may lower it.
        With lexicon wording (an everyday offence description was matched): the added statutory
        words raise similarity for civil questions too ("my plants were damaged"), so the question
        is answered only if the scope check says it is about a crime - and then down to the
        scope floor. Without a scope check the lexicon only re-ranks."""
        if self.scope is None:
            return self.min_similarity, None
        if not lexicon and best_similarity >= self.min_similarity:
            return self.min_similarity, None
        p = self.scope.probability(scope_embedding)
        if p >= self.scope.threshold:
            return min(self.scope_floor, self.min_similarity), p
        return (_REFUSE if lexicon else self.min_similarity), p

    def best_substantive(self, ranked: list[Candidate]) -> float:
        """Best similarity among chunks that can answer a question: a state amendment or a Chapter I
        general provision clearing the floor on its own is not a real match (it triggers the scope
        check instead)."""
        return max((c.similarity for c in ranked if c.explicit or self.kind_pref(c.chunk) == 1.0), default=0.0)

    def kind_pref(self, chunk: Chunk) -> float:
        if "State Amendments" in chunk.citation_path:
            return self.boilerplate_pref.get("state_amendment", 1.0)
        if chunk.chapter == "I":
            return self.boilerplate_pref.get("chapter_i", 1.0)
        return 1.0

    def rerank(self, cands: list[Candidate]) -> list[Candidate]:
        """Score = similarity x role preference x boilerplate preference; explicit section
        matches rank first.
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
            c.score = c.similarity * self.role_pref.get(c.chunk.role, 1.0) * (1.0 if c.explicit else self.kind_pref(c.chunk))
        return sorted(merged.values(), key=lambda c: (c.explicit, c.score), reverse=True)

    def assemble_context(
        self, cands: list[Candidate], max_chunks: int = 6, max_context_chars: int = 6000, floor: float | None = None
    ) -> list[SourceEvidence]:
        """Bounded evidence list. Returns [] when no candidate clears the similarity floor,
        which ChatRouter turns into a scope-boundary refusal."""
        evidence: list[SourceEvidence] = []
        used = 0
        for c in cands:
            if len(evidence) >= max_chunks:
                break
            if c.similarity < (self.min_similarity if floor is None else floor) and not c.explicit:
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
