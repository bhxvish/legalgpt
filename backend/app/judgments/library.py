"""The judgment library: real court judgments, split into role-labelled passages, searchable for citation.

Each judgment is segmented into sentences, and every sentence gets a rhetorical role: the human
label where one exists (Annotate / Review tabs), otherwise the sentence-role ensemble's prediction.
Consecutive sentences of one role form a passage (at most a few sentences), stored with the case's
title, court and year in a separate Chroma collection, "judgments", next to the IPC index. The IPC
collection, and the zero-knowledge retrieval proof over it, are unaffected.

Two uses:
- the chat cites what a court *held*: only Ruling, Law Applied and Precedent passages are cited
  (never a party's Argument, which is not law);
- case analysis finds similar past cases by comparing the user's facts with Facts passages.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.models import Chunk, SourceEvidence

COLLECTION = "judgments"
CITE_ROLES = ("Ruling", "Law Applied", "Precedent")
FACT_ROLES = ("Facts",)
MAX_SENTENCES, MAX_CHARS = 4, 1100


@dataclass
class JudgmentDoc:
    case_id: str
    title: str
    court: str
    year: int | None
    sentences: list[str]
    roles: list[str | None]
    source: str  # "library" (fetched Supreme Court judgments) or "corpus" (the annotated collection)


def short_citation(title: str, court: str, year: int | None) -> str:
    """"Narender v. State of Delhi (2021)", "State of U.P. v. Ram (SC, 2023)"."""
    court_short = "SC" if "Supreme" in court else court
    inside = ", ".join(x for x in (court_short, str(year) if year else "") if x)
    return f"{title} ({inside})" if inside else title


def passages(doc: JudgmentDoc) -> list[Chunk]:
    """Consecutive same-role sentences, at most MAX_SENTENCES / MAX_CHARS per passage."""
    out: list[Chunk] = []
    i, n = 0, len(doc.sentences)
    while i < n:
        role, j, size = doc.roles[i], i, 0
        while j < n and doc.roles[j] == role and j - i < MAX_SENTENCES and size + len(doc.sentences[j]) <= MAX_CHARS:
            size += len(doc.sentences[j]) + 1
            j += 1
        j = max(j, i + 1)
        if role:
            text = " ".join(doc.sentences[i:j])
            para = f"sentences {i + 1}-{j}" if j - i > 1 else f"sentence {i + 1}"
            out.append(Chunk(
                chunk_id=f"jdg::{doc.case_id}::{i}", doc_id=doc.case_id, text=text,
                citation_path=f"{short_citation(doc.title, doc.court, doc.year)}, {para}",
                chapter=f"{doc.court}|{doc.year or ''}|{doc.source}", section="", title=doc.title, role=role))
        i = j
    return out


def chunk_meta(chunk: Chunk) -> tuple[str, int | None, str]:
    court, year, source = (chunk.chapter.split("|") + ["", "", ""])[:3]
    return court, int(year) if year.isdigit() else None, source


class JudgmentSearch:
    """Searches the judgments collection. At most `per_case` passages per judgment, so one long
    judgment cannot fill every slot."""

    def __init__(self, store: Any, embedder: Any, min_similarity: float = 0.5, k: int = 2, per_case: int = 1) -> None:
        self.store, self.embedder = store, embedder
        self.min_similarity, self.k, self.per_case = min_similarity, k, per_case

    def available(self) -> bool:
        try:
            return self.store.count() > 0
        except Exception:
            return False

    def search(self, query: str, roles: Sequence[str] = CITE_ROLES, k: int | None = None,
               min_similarity: float | None = None, exclude_cases: Sequence[str] = (),
               within_case: str | None = None) -> list[tuple[Chunk, float]]:
        k = k or self.k
        floor = self.min_similarity if min_similarity is None else min_similarity
        vec = np.asarray(self.embedder.embed([query]))[0]
        where: dict = {"role": {"$in": list(roles)}} if len(roles) > 1 else {"role": roles[0]}
        if within_case:
            where = {"$and": [where, {"doc_id": within_case}]}
        hits = self.store.query(vec, k=max(20, 6 * k), where=where)
        out, per = [], {}
        for h in sorted(hits, key=lambda h: -h.similarity):
            c = h.chunk
            # one judgment can be in the library twice (annotated copy + fetched copy): count by title
            key = re.sub(r"[^a-z0-9]+", "", c.title.lower())[:40] or c.doc_id
            if h.similarity < floor or c.doc_id in exclude_cases or per.get(key, 0) >= self.per_case:
                continue
            per[key] = per.get(key, 0) + 1
            out.append((c, float(h.similarity)))
            if len(out) >= k:
                break
        return out

    @staticmethod
    def to_sources(hits: Sequence[tuple[Chunk, float]], first_marker: int) -> list[SourceEvidence]:
        out = []
        for n, (c, sim) in enumerate(hits):
            court, year, _ = chunk_meta(c)
            out.append(SourceEvidence(
                marker=first_marker + n, chunk_id=c.chunk_id, doc_id=c.doc_id, citation_path=c.citation_path,
                text=c.text, similarity=round(sim, 4), score=round(sim, 4), role=c.role, section="", title=c.title,
                kind="judgment", court=court, year=year))
        return out


# ----------------------------------------------------------------------- loading judgments

def load_library_texts(folder: Path) -> list[tuple[str, str, str, int | None, str]]:
    """(case_id, title, court, year, text) for every fetched judgment with its metadata file."""
    out = []
    for txt in sorted(Path(folder).glob("*.txt")):
        meta_path = txt.with_suffix(".json")
        if not meta_path.exists():
            continue
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        out.append((txt.stem, tidy_title(m.get("title", txt.stem)), m.get("court", "Supreme Court of India"),
                    m.get("year"), txt.read_text(encoding="utf-8")))
    return out


def tidy_title(title: str) -> str:
    """"STATE OF U.P. versus RAM KUMAR" -> "State of U.P. v. Ram Kumar" (readable in a citation)."""
    t = re.sub(r"\s+", " ", title).strip()
    t = re.sub(r"\s*\([^()]*\d{4}[^()]*\)?\s*$", "", t)  # a trailing "(3 February, 2020)" date: the year is kept separately
    if re.sub(r"\s(versus|vs\.?|v/s)\s", " ", t).isupper():  # all capitals apart from "versus"
        small = {"of", "and", "the", "in", "for", "versus", "vs", "vs."}
        t = " ".join(w.lower() if w.lower() in small else (w if re.fullmatch(r"([A-Z]\.)+", w) else w.capitalize())
                     for w in t.split())
    t = re.sub(r"\s+(versus|vs\.?|v/s)\s+", " v. ", t, flags=re.I)
    return t[:140]
