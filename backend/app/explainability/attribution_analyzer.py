"""AttributionAnalyzer: which retrieved source (if any) supports each sentence of the answer?

Each answer sentence is compared by cosine similarity (EmbeddingService, normalized vectors) with
every *sentence* of every source — not whole chunks, which MiniLM would truncate at ~256 tokens —
so the best match also yields the supporting excerpt. A sentence is "supported" when its best
match reaches `support_threshold`. Separately, a sentence that cites [n] is checked against
source n itself: well supported by another source but not by the one it cites is worth flagging.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from app.core.citations import normalize_stream
from app.core.embeddings import Embedder
from app.core.models import SourceEvidence
from app.core.text_utils import split_sentences

_MARKER = re.compile(r"\[(\d+)\]")
_MARKUP = re.compile(r"\*\*|__|`|^[#>*\-\s]+", re.M)
MIN_WORDS = 4  # shorter fragments ("Section 279 [1].") carry too little meaning to attribute


@dataclass
class Attribution:
    sentence_idx: int
    sentence: str
    source_marker: int | None  # best-matching source
    similarity: float  # with its best-matching excerpt
    evidence: str  # that excerpt
    supported: bool
    cited_markers: list[int]
    cited_similarity: float | None  # best match within the sources this sentence cites
    cited_supported: bool | None  # None when the sentence cites nothing


class AttributionAnalyzer:
    def __init__(self, embedder: Embedder, support_threshold: float = 0.55) -> None:
        """0.55 measured on real answers (Phase 4 comparison, MiniLM): 90% of sentences scored
        against their own retrieved sources clear it, 10% scored against sources about unrelated
        offences do. Sources about a *neighbouring* offence are not separable by embeddings, so
        attribution catches off-topic/unsupported sentences, not subtle legal errors."""
        self.embedder = embedder
        self.support_threshold = support_threshold

    @staticmethod
    def split_sentences(answer: str) -> list[str]:
        text = "".join(normalize_stream([answer]))
        sentences: list[str] = []
        for line in text.split("\n"):
            line = _MARKUP.sub("", line).strip()
            if line:
                sentences.extend(split_sentences(line))
        return [s for s in sentences if len(re.findall(r"\w+", _MARKER.sub("", s))) >= MIN_WORDS]

    def attribute(self, answer: str, sources: Sequence[SourceEvidence]) -> list[Attribution]:
        sentences = self.split_sentences(answer)
        excerpts = [(s.marker, part) for s in sources for part in split_sentences(s.text.replace("\n", " ")) if part.strip()]
        if not sentences or not excerpts:
            return []
        clean = [_MARKER.sub("", s).strip() for s in sentences]
        vectors = self.embedder.embed(clean + [text for _, text in excerpts])
        a, e = vectors[: len(clean)], vectors[len(clean) :]
        sims = np.clip(a @ e.T, 0.0, 1.0)  # normalized embeddings: dot product = cosine
        markers = np.array([m for m, _ in excerpts])
        out: list[Attribution] = []
        for i, sentence in enumerate(sentences):
            cited = [int(m) for m in _MARKER.findall(sentence)]
            in_cited = np.isin(markers, cited)
            cited_sim = float(sims[i][in_cited].max()) if cited and in_cited.any() else (0.0 if cited else None)
            if cited_sim is not None and cited_sim >= self.support_threshold:
                # the source the answer cites does support it: credit that one, even if another
                # source happens to score a little higher (e.g. s.304 vs the cited s.304A)
                j = int(np.flatnonzero(in_cited)[sims[i][in_cited].argmax()])
            else:
                j = int(sims[i].argmax())
            out.append(Attribution(
                sentence_idx=i,
                sentence=sentence,
                source_marker=int(markers[j]),
                similarity=round(float(sims[i, j]), 4),
                evidence=excerpts[j][1],
                supported=bool(sims[i, j] >= self.support_threshold),
                cited_markers=cited,
                cited_similarity=None if cited_sim is None else round(cited_sim, 4),
                cited_supported=None if cited_sim is None else cited_sim >= self.support_threshold,
            ))
        return out

    @staticmethod
    def support_ratio(attributions: Sequence[Attribution]) -> float:
        """Share of answer sentences supported by some source (0.0 when there are none)."""
        return round(sum(a.supported for a in attributions) / len(attributions), 4) if attributions else 0.0
