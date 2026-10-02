"""ModelComparator: the same held-out questions, the same retrieved evidence, several models.

For each question the Retriever runs once; every model then gets the identical prompt
(PromptBuilder, legal mode). score_citations() checks each answer's citations against the
evidence that was actually retrieved — computed, never estimated:

- [n] markers must point to a retrieved source (1..len(sources));
- every IPC section the answer names must appear in a retrieved source;
- every precedent the answer names ("X v. Y") must appear in a retrieved source.

citation_accuracy = grounded citations / all citations (None when an answer cites nothing).
It measures grounding, not correctness: citing retrieved-but-irrelevant evidence still counts as
grounded. uses_markers records whether the answer follows the required [n] citation format.
"""

import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.core.citations import normalize_stream
from app.core.llm_client import LLMClient
from app.core.models import SourceEvidence
from app.core.prompt_builder import NOT_COVERED_REPLY, PromptBuilder

_MARKER = re.compile(r"\[(\d+)\]")
_SECTION = re.compile(r"\b(?:sections?|secs?\.?|s\.|u/s\.?)\s*(\d{1,3}[A-Z]{0,2})\b", re.I)
_PRECEDENT = re.compile(r"\b([A-Z][A-Za-z.&']+(?:\s+[A-Z][A-Za-z.&']+){0,4})\s+v(?:s)?\.?\s+(?:the\s+)?([A-Z][A-Za-z.&']+)")


@dataclass
class CitationScore:
    markers: list[int]
    invalid_markers: list[int]
    sections: list[str]
    ungrounded_sections: list[str]
    precedents: list[str]
    ungrounded_precedents: list[str]
    citation_accuracy: float | None
    refused: bool
    uses_markers: bool  # follows the required inline [n] citation format at all


@dataclass
class ModelAnswer:
    model_id: str
    answer: str
    score: CitationScore | None
    seconds: float
    error: str = ""


@dataclass
class ComparisonRow:
    question: str
    origin: str  # where the held-out question came from
    sources: list[dict[str, Any]]
    answers: dict[str, ModelAnswer] = field(default_factory=dict)


class ModelComparator:
    def __init__(self, clients: dict[str, LLMClient], questions: Sequence[tuple[str, str]], prompt_builder: PromptBuilder | None = None) -> None:
        """`clients` maps a column name ("groq", "base", "tuned") to a model; `questions` are
        (question, origin) pairs."""
        self.clients = clients
        self.questions = list(questions)
        self.prompt_builder = prompt_builder or PromptBuilder()

    @staticmethod
    def score_citations(answer: str, sources: Sequence[SourceEvidence]) -> CitationScore:
        text = "".join(normalize_stream([answer]))  # same canonical markers as the chat stream
        evidence = "\n".join(s.text for s in sources).lower()
        source_sections = {s.section.upper() for s in sources if s.section}

        markers = [int(m) for m in _MARKER.findall(text)]
        invalid = [m for m in markers if not 1 <= m <= len(sources)]

        sections = sorted({s.upper() for s in _SECTION.findall(text)})
        ungrounded_sections = [s for s in sections
                               if s not in source_sections and not re.search(rf"\b{re.escape(s.lower())}\b", evidence)]

        precedents = sorted({f"{a} v. {b}" for a, b in _PRECEDENT.findall(text)
                             if a.lower() not in {"state", "section", "the"}})
        ungrounded_precedents = [p for p in precedents if p.split(" v. ")[0].split()[-1].lower() not in evidence]

        total = len(markers) + len(sections) + len(precedents)
        bad = len(invalid) + len(ungrounded_sections) + len(ungrounded_precedents)
        return CitationScore(
            markers=markers,
            invalid_markers=invalid,
            sections=sections,
            ungrounded_sections=ungrounded_sections,
            precedents=precedents,
            ungrounded_precedents=ungrounded_precedents,
            citation_accuracy=round((total - bad) / total, 4) if total else None,
            refused=NOT_COVERED_REPLY.split("—")[0].strip().lower() in text.lower(),
            uses_markers=bool(markers),
        )

    def run(self, retriever: Any, log: Any = print) -> list[ComparisonRow]:
        evidence = []
        for question, origin in self.questions:
            sources = retriever.retrieve(question)
            evidence.append((question, origin, sources))
        rows = [ComparisonRow(q, o, [s.model_dump(include={"marker", "citation_path", "section", "similarity"}) for s in src])
                for q, o, src in evidence]
        # one model at a time, so local models can be released before the next loads (4 GB GPU)
        for name, client in self.clients.items():
            log(f"model {name}: {client.model_id}")
            for row, (question, _, sources) in zip(rows, evidence):
                if not sources:  # the chat app would refuse without calling any model
                    row.answers[name] = ModelAnswer(client.model_id, "(no evidence cleared the retrieval floor)", None, 0.0)
                    continue
                t0 = time.perf_counter()
                try:
                    answer = client.generate(self.prompt_builder.build(question, sources, "legal"))
                    row.answers[name] = ModelAnswer(client.model_id, answer, self.score_citations(answer, sources),
                                                    round(time.perf_counter() - t0, 1))
                except Exception as exc:  # report the failure in the row instead of aborting the run
                    row.answers[name] = ModelAnswer(client.model_id, "", None, round(time.perf_counter() - t0, 1), str(exc))
                log(f"  {row.answers[name].seconds:5.1f}s  {question[:70]}")
            if hasattr(client, "close"):
                client.close()
        return rows

    @staticmethod
    def summary(rows: Sequence[ComparisonRow]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name in rows[0].answers if rows else []:
            scored = [r.answers[name].score for r in rows if r.answers[name].score is not None]
            accs = [s.citation_accuracy for s in scored if s.citation_accuracy is not None]
            out[name] = {
                "model_id": rows[0].answers[name].model_id,
                "answered": len(scored),
                "mean_citation_accuracy": round(sum(accs) / len(accs), 4) if accs else None,
                "answers_with_citations": len(accs),
                "answers_using_markers": sum(s.uses_markers for s in scored),
                "answers_without_citations": sum(1 for s in scored if s.citation_accuracy is None and not s.refused),
                "refusals": sum(s.refused for s in scored),
                "invalid_markers": sum(len(s.invalid_markers) for s in scored),
                "ungrounded_sections": sum(len(s.ungrounded_sections) for s in scored),
                "ungrounded_precedents": sum(len(s.ungrounded_precedents) for s in scored),
                "errors": sum(1 for r in rows if r.answers[name].error),
                "mean_seconds": round(sum(r.answers[name].seconds for r in rows) / len(rows), 1),
            }
        return out

    def save(self, rows: Sequence[ComparisonRow], out_dir: str | Path, meta: dict[str, Any] | None = None) -> tuple[Path, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = self.summary(rows)
        payload = {"meta": meta or {}, "summary": summary, "rows": [asdict(r) for r in rows]}
        json_path = out_dir / "comparison.json"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        names = list(summary)
        md = ["# Base vs tuned comparison", ""]
        for k, v in (meta or {}).items():
            md.append(f"- **{k}:** {v}")
        md += ["", "## Summary", "", "| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for metric in ("model_id", "answered", "mean_citation_accuracy", "answers_with_citations", "answers_using_markers", "answers_without_citations",
                       "refusals", "invalid_markers", "ungrounded_sections", "ungrounded_precedents", "errors", "mean_seconds"):
            md.append(f"| {metric} | " + " | ".join(str(summary[n][metric]) for n in names) + " |")
        for i, r in enumerate(rows, 1):
            md += ["", f"## Q{i}. {r.question}", "", f"*Origin:* {r.origin}", "",
                   "*Retrieved:* " + "; ".join(f"[{s['marker']}] {s['citation_path']} ({s['similarity']:.2f})" for s in r.sources)]
            for n in names:
                a = r.answers[n]
                sc = a.score
                verdict = ("error: " + a.error) if a.error else (
                    "no citations" if sc is None or sc.citation_accuracy is None else
                    f"citation accuracy {sc.citation_accuracy:.2f} (markers {sc.markers}, invalid {sc.invalid_markers}, "
                    f"ungrounded sections {sc.ungrounded_sections}, ungrounded precedents {sc.ungrounded_precedents})")
                md += ["", f"**{n}** — {verdict}{' · refused' if sc and sc.refused else ''} · {a.seconds}s", "",
                       "> " + (a.answer or "").replace("\n", "\n> ")]
        md_path = out_dir / "comparison.md"
        md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
        return json_path, md_path
