"""VerificationService: case text + cited section -> Prolog verdict with an element breakdown.

Proof of concept over a deliberately small set of IPC sections (rules/*.pl). The verdict says
whether the facts *as extracted by the model* satisfy the encoded elements of a section — not
whether the accused is guilty. Every finding carries the evidence quote it rests on.
"""

import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from app.verification.fact_extractor import FactExtractor
from app.verification.prolog_engine import PrologEngine, SectionDef, Verdict
from app.verification.predicate_mapper import PredicateMapper

DISCLAIMER = (
    "Rule-based check of the facts as extracted by an AI model against the encoded elements of the "
    "section. It does not decide guilt, and the extraction can be wrong: read the evidence quotes. "
    "Not legal advice."
)


@dataclass
class ElementResult:
    predicate: str
    description: str
    required: list[str]
    found: str  # the extracted value (true / false / unknown / an act type)
    evidence: str
    status: str  # satisfied | violated | missing


@dataclass
class SectionResult:
    section: str
    title: str
    status: Verdict
    elements: list[ElementResult]


@dataclass
class VerificationResult:
    case_id: str
    cited_section: str
    status: Verdict
    title: str
    elements: list[ElementResult]
    alternatives: list[SectionResult]  # other encoded sections, consistent ones first
    facts: dict[str, dict[str, str]]
    warnings: list[str] = field(default_factory=list)
    model_id: str = ""
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ORDER = {"CONSISTENT": 0, "INSUFFICIENT": 1, "INCONSISTENT": 2}


class VerificationService:
    def __init__(self, extractor: FactExtractor, engine: PrologEngine, mapper: PredicateMapper | None = None, extraction_runs: int = 2) -> None:
        self.extractor = extractor
        self.extraction_runs = extraction_runs  # values must agree across runs (FactExtractor.extract_consistent)
        self.engine = engine
        self.mapper = mapper or PredicateMapper()
        by_number = sorted(engine.sections(), key=lambda s: (int(re.match(r"\d+", s.section).group()), s.section))  # type: ignore[union-attr]
        self._sections = {s.section: s for s in by_number}

    @property
    def sections(self) -> list[SectionDef]:
        return list(self._sections.values())

    def _section_result(self, case_id: str, section: SectionDef, facts: dict[str, dict[str, str]]) -> SectionResult:
        statuses = self.engine.element_statuses(case_id, section.section)
        return SectionResult(
            section.section,
            section.title,
            self.engine.verdict(case_id, section.section),
            [ElementResult(e.predicate, e.description, list(e.required), facts.get(e.predicate, {}).get("value", "unknown"),
                           facts.get(e.predicate, {}).get("evidence", ""), statuses.get(e.predicate, "missing"))
             for e in section.elements],
        )

    def verify(self, case_text: str, cited_section: str) -> VerificationResult:
        section = self._sections.get(cited_section.strip().upper())
        if section is None:
            raise ValueError(f"section {cited_section!r} is not encoded; available: {sorted(self._sections)}")
        extraction = self.extractor.extract_consistent(case_text, self.extraction_runs)
        case_id = f"v_{uuid.uuid4().hex[:12]}"
        clauses = self.mapper.to_prolog_facts(case_id, extraction.facts)
        with self.engine.session(case_id, clauses):  # facts are retracted even if a query raises
            cited = self._section_result(case_id, section, extraction.facts)
            alternatives = self.suggest_sections(case_id, extraction.facts, exclude=section.section)
        return VerificationResult(
            case_id=case_id,
            cited_section=section.section,
            status=cited.status,
            title=section.title,
            elements=cited.elements,
            alternatives=alternatives,
            facts=extraction.facts,
            warnings=extraction.warnings,
            model_id=getattr(self.extractor.llm, "model_id", ""),
        )

    def suggest_sections(self, case_id: str, facts: dict[str, dict[str, str]], exclude: str = "") -> list[SectionResult]:
        """Check every other encoded section against the same (already asserted) facts."""
        results = [self._section_result(case_id, s, facts) for s in self.sections if s.section != exclude]
        return sorted(results, key=lambda r: (_ORDER[r.status], r.section))
