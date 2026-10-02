"""PrologEngine: SWI-Prolog (via pyswip) loaded with the rules/ knowledge base.

pyswip drives one embedded SWI-Prolog per process and is not thread-safe; FastAPI runs requests
on a thread pool, so every operation takes a process-wide lock, and rule files are consulted
once per process. Facts are per case id; session() guarantees they are retracted afterwards,
even when a query raises, so verification runs never leak state into each other.
"""

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.verification.predicate_mapper import quote_atom

Verdict = Literal["CONSISTENT", "INCONSISTENT", "INSUFFICIENT"]
ElementStatus = Literal["satisfied", "violated", "missing"]

_LOCK = threading.RLock()
_LOADED: set[str] = set()


@dataclass(frozen=True)
class ElementDef:
    predicate: str
    required: tuple[str, ...]  # acceptable values
    description: str


@dataclass(frozen=True)
class SectionDef:
    section: str
    title: str
    elements: tuple[ElementDef, ...]


class PrologEngine:
    def __init__(self, rules_dir: str | Path) -> None:
        from pyswip import Prolog

        self.rules_dir = Path(rules_dir)
        self.prolog = Prolog()
        files = sorted(self.rules_dir.glob("*.pl"), key=lambda p: (p.name != "core.pl", p.name))
        if not files:
            raise FileNotFoundError(f"no .pl rule files in {self.rules_dir}")
        with _LOCK:
            for f in files:
                key = str(f.resolve())
                if key not in _LOADED:
                    self.prolog.consult(f.resolve().as_posix())
                    _LOADED.add(key)

    def _query(self, goal: str) -> list[dict]:
        with _LOCK:
            return list(self.prolog.query(goal))

    # ------------------------------------------------------------ knowledge base

    def sections(self) -> list[SectionDef]:
        out = []
        for row in self._query("section(S, T)"):
            s = str(row["S"])
            elements = tuple(
                ElementDef(str(e["P"]), tuple(str(v) for v in e["R"]) if isinstance(e["R"], list) else (str(e["R"]),), str(e["D"]))
                for e in self._query(f"element({quote_atom(s)}, P, R, D)")
            )
            out.append(SectionDef(s, str(row["T"]), elements))
        return out

    # --------------------------------------------------------------- case facts

    def assert_facts(self, clauses: Sequence[str]) -> None:
        with _LOCK:
            for clause in clauses:
                self.prolog.assertz(clause)

    def retract_case(self, case_id: str) -> None:
        with _LOCK:
            self.prolog.retractall(f"fact({quote_atom(case_id)}, _, _)")

    def case_facts(self, case_id: str) -> dict[str, str]:
        return {str(r["P"]): str(r["V"]) for r in self._query(f"fact({quote_atom(case_id)}, P, V)")}

    @contextmanager
    def session(self, case_id: str, clauses: Sequence[str]) -> Iterator["PrologEngine"]:
        """Assert a case's facts for the duration of the block; always retract them."""
        try:
            self.assert_facts(clauses)
            yield self
        finally:
            self.retract_case(case_id)

    # ------------------------------------------------------------------ queries

    def verdict(self, case_id: str, section: str) -> Verdict:
        rows = self._query(f"verdict({quote_atom(case_id)}, {quote_atom(section)}, V)")
        if not rows:
            raise ValueError(f"section {section!r} is not encoded in {self.rules_dir}")
        return str(rows[0]["V"]).upper()  # type: ignore[return-value]

    def element_statuses(self, case_id: str, section: str) -> dict[str, ElementStatus]:
        return {str(r["P"]): str(r["S"]) for r in  # type: ignore[misc]
                self._query(f"element_status({quote_atom(case_id)}, {quote_atom(section)}, P, S)")}
