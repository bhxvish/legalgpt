"""PredicateMapper: validated facts -> Prolog fact/3 clauses.

Only schema predicates and schema values reach Prolog, and the case id is quoted, so no text
from the case or the model can inject Prolog code.
"""

import re

from app.verification.fact_schema import BY_NAME

_CASE_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def quote_atom(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


class PredicateMapper:
    @staticmethod
    def to_prolog_facts(case_id: str, facts: dict[str, dict[str, str]]) -> list[str]:
        """One `fact(CaseId, Predicate, Value)` clause per predicate (without the final period)."""
        if not _CASE_ID.match(case_id):
            raise ValueError(f"invalid case id {case_id!r}")
        clauses = []
        for name, entry in facts.items():
            predicate = BY_NAME.get(name)
            if predicate is None:
                raise ValueError(f"unknown predicate {name!r}")
            value = entry["value"]
            if value not in predicate.values:
                raise ValueError(f"{name}: invalid value {value!r}")
            clauses.append(f"fact({quote_atom(case_id)}, {name}, {value})")
        return clauses
