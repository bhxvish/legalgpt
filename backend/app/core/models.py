"""Shared data types for Module 0 (core retrieval & chat)."""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

NodeKind = Literal["chapter", "section", "subsection", "clause", "paragraph"]
Role = Literal["system", "user", "assistant"]


@dataclass
class Node:
    """One structural unit of a parsed legal document.

    `text` holds only the node's own body (not its children's). `citation_path`
    is the human-readable location, e.g. "IPC > Chapter XVI > Section 304A > (1)".
    """

    kind: NodeKind
    label: str
    citation_path: str
    text: str = ""
    title: str = ""
    chapter: str = ""
    section: str = ""
    children: list["Node"] = field(default_factory=list)


@dataclass
class Chunk:
    """A bounded, citation-tagged unit of text ready for embedding."""

    chunk_id: str
    doc_id: str
    text: str
    citation_path: str
    chapter: str = ""
    section: str = ""
    title: str = ""
    role: str = "Law"  # rhetorical role; bare-act text is always "Law"

    def metadata(self) -> dict[str, str]:
        return {
            "doc_id": self.doc_id,
            "citation_path": self.citation_path,
            "chapter": self.chapter,
            "section": self.section,
            "title": self.title,
            "role": self.role,
        }


@dataclass
class Message:
    role: Role
    content: str


class SourceEvidence(BaseModel):
    """A retrieved chunk as presented to the LLM and the UI.

    `marker` is the 1-based number the model cites inline as `[marker]`.
    `similarity` is the cosine similarity between question and chunk embeddings
    (a retrieval-match signal, not a measure of answer correctness).
    """

    marker: int
    chunk_id: str
    doc_id: str
    citation_path: str
    text: str
    similarity: float
    score: float
    role: str
    section: str = ""
    title: str = ""
