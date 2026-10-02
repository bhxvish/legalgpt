"""FactSchema: the shared predicate set a fact extraction must populate.

One schema covers the union of elements across every encoded section (rules/*.pl); a test
asserts the two stay in sync. Values are explicit: "unknown" is a first-class answer, because
the rules must never read "the facts don't say" as "false".
"""

import re
from dataclasses import dataclass, field
from typing import Any

BOOL = ("true", "false", "unknown")
ACT_TYPES = ("rash", "negligent", "intentional", "accidental", "unknown")


@dataclass(frozen=True)
class Predicate:
    name: str
    values: tuple[str, ...]
    question: str  # what the extractor must decide, in plain words


PREDICATES: tuple[Predicate, ...] = (
    Predicate("caused_death", BOOL, "Did the accused's act cause the death of a person?"),
    Predicate("caused_hurt", BOOL, "Did the act cause bodily pain, disease or infirmity (hurt) to a person?"),
    Predicate("caused_grievous_hurt", BOOL, "Did the act cause grievous hurt (fracture, permanent loss of a limb, sight or hearing, disfigurement, or hurt endangering life / 20 days' severe pain)?"),
    Predicate("act_type", ACT_TYPES, "Was the act rash (conscious disregard of a known risk), negligent (failure of reasonable care), intentional, or a pure accident?"),
    Predicate("intent_to_kill", BOOL, "Did the accused intend to cause death, or such bodily injury as is likely to cause death?"),
    Predicate("knowledge_likely_death", BOOL, "Did the accused know the act was likely to cause death?"),
    Predicate("endangered_safety", BOOL, "Did the act endanger human life or the personal safety of others (or was it likely to cause hurt)?"),
    Predicate("drove_vehicle", BOOL, "Was the accused driving a vehicle or riding?"),
    Predicate("on_public_way", BOOL, "Did it happen on a public way (road, street, highway)?"),
    Predicate("movable_property", BOOL, "Was the property concerned movable property?"),
    Predicate("taken_from_possession", BOOL, "Was the property taken out of another person's possession?"),
    Predicate("without_consent", BOOL, "Was it taken without that person's consent?"),
    Predicate("dishonest_intention", BOOL, "Did the accused intend to take it dishonestly (to cause wrongful gain or wrongful loss)?"),
    Predicate("property_moved", BOOL, "Was the property moved in order to take it?"),
    Predicate("woman_unnatural_death", BOOL, "Did a woman die of burns, bodily injury, or otherwise than under normal circumstances?"),
    Predicate("within_seven_years_of_marriage", BOOL, "Did her death occur within seven years of her marriage?"),
    Predicate("cruelty_by_husband_or_relative", BOOL, "Was she subjected to cruelty or harassment by her husband or a relative of her husband?"),
    Predicate("cruelty_soon_before_death", BOOL, "Was that cruelty or harassment soon before her death?"),
    Predicate("dowry_demand", BOOL, "Was the cruelty or harassment for, or in connection with, a demand for dowry?"),
)
BY_NAME: dict[str, Predicate] = {p.name: p for p in PREDICATES}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def quote_in_text(quote: str, normalized_text: str) -> bool:
    """The quote appears in the text, allowing elisions ("Bal Krishan ... suffered injuries"):
    each elided piece must appear, in order."""
    pieces = [_norm(p) for p in re.split(r"\.\.\.+|…", quote)]
    pieces = [p for p in pieces if p]
    if not pieces:
        return False
    pos = 0
    for piece in pieces:
        found = normalized_text.find(piece, pos)
        if found == -1:
            return False
        pos = found + len(piece)
    return True


@dataclass
class ValidationResult:
    ok: bool
    facts: dict[str, dict[str, str]]  # normalized: {predicate: {"value", "evidence"}}
    errors: list[str] = field(default_factory=list)  # make the extraction unusable (triggers a repair)
    warnings: list[str] = field(default_factory=list)  # tolerated, but reported to the user


class FactSchema:
    predicates = PREDICATES

    def validate(self, facts: Any, case_text: str | None = None) -> ValidationResult:
        """Structure and values are errors. Missing predicates become "unknown" (warning).
        With `case_text`, a non-unknown value whose evidence quote is not found in the text is
        downgraded to "unknown" (warning): an unsupported extraction must not decide a verdict."""
        errors: list[str] = []
        warnings: list[str] = []
        if not isinstance(facts, dict):
            return ValidationResult(False, {}, [f"expected a JSON object, got {type(facts).__name__}"])
        unknown_keys = sorted(set(facts) - set(BY_NAME))
        if unknown_keys:
            errors.append(f"unknown predicates: {unknown_keys}")
        text = _norm(case_text) if case_text else None
        out: dict[str, dict[str, str]] = {}
        for p in PREDICATES:
            raw = facts.get(p.name)
            if raw is None:
                warnings.append(f"{p.name}: not returned, treated as unknown")
                out[p.name] = {"value": "unknown", "evidence": ""}
                continue
            if isinstance(raw, (str, bool)):
                raw = {"value": raw, "evidence": ""}
            if not isinstance(raw, dict) or "value" not in raw:
                errors.append(f"{p.name}: expected {{\"value\": ..., \"evidence\": ...}}")
                continue
            value = str(raw["value"]).strip().lower()
            evidence = str(raw.get("evidence") or "").strip()
            if value not in p.values:
                errors.append(f"{p.name}: {value!r} is not one of {list(p.values)}")
                continue
            if value != "unknown" and text is not None:
                if not quote_in_text(evidence, text):
                    warnings.append(f"{p.name}: evidence for {value!r} was not found in the text, treated as unknown")
                    value = "unknown"
            out[p.name] = {"value": value, "evidence": evidence}
        return ValidationResult(not errors, out, errors, warnings)
