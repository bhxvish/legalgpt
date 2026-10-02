"""FactExtractor: case text -> schema facts (true/false/unknown + evidence quote) via an LLM.

Reuses Module 0's LLMClient. The prompt lists every predicate with a plain-language question and
gives few-shot examples. If the reply is not valid JSON or fails FactSchema validation, there is
exactly one repair attempt (repair_json), which shows the model its own reply and the errors.
"""

import json
import re
from typing import Any

from app.core.llm_client import LLMClient
from app.core.models import Message
from app.verification.fact_schema import PREDICATES, FactSchema, ValidationResult


class ExtractionError(ValueError):
    """The model did not produce schema-valid facts, even after the repair attempt."""


def _facts(**known: tuple[str, str]) -> str:
    """Few-shot answer: every predicate, unknown unless given as (value, evidence)."""
    return json.dumps({p.name: {"value": known[p.name][0], "evidence": known[p.name][1]} if p.name in known
                       else {"value": "unknown", "evidence": ""} for p in PREDICATES}, ensure_ascii=False)


# Illustrative case descriptions written for this prompt (not real cases).
FEW_SHOTS: tuple[tuple[str, str], ...] = (
    (
        "The accused was driving a tanker at high speed on the national highway, overtaking from the wrong side "
        "despite heavy traffic. The tanker hit a motorcycle and the rider died on the spot. Nothing suggests "
        "the accused knew the rider or intended any harm.",
        _facts(
            caused_death=("true", "the rider died on the spot"),
            act_type=("rash", "driving a tanker at high speed on the national highway, overtaking from the wrong side despite heavy traffic"),
            intent_to_kill=("false", "Nothing suggests the accused knew the rider or intended any harm"),
            endangered_safety=("true", "overtaking from the wrong side despite heavy traffic"),
            drove_vehicle=("true", "The accused was driving a tanker"),
            on_public_way=("true", "on the national highway"),
        ),
    ),
    (
        "PW-1 stated that the accused entered her shop while she was away and carried off her mobile phone from "
        "the counter without asking her. The phone was recovered from the accused, who had offered it for sale.",
        _facts(
            movable_property=("true", "her mobile phone"),
            taken_from_possession=("true", "carried off her mobile phone from the counter"),
            without_consent=("true", "without asking her"),
            dishonest_intention=("true", "who had offered it for sale"),
            property_moved=("true", "carried off her mobile phone"),
        ),
    ),
    (
        "The deceased was married to the accused in 2019 and died of burn injuries in 2021 at the matrimonial "
        "home. Her father deposed that a month before her death the husband and his mother beat her for not "
        "bringing a motorcycle and cash from her parents.",
        _facts(
            woman_unnatural_death=("true", "died of burn injuries"),
            within_seven_years_of_marriage=("true", "married to the accused in 2019 and died of burn injuries in 2021"),
            cruelty_by_husband_or_relative=("true", "the husband and his mother beat her"),
            cruelty_soon_before_death=("true", "a month before her death"),
            dowry_demand=("true", "for not bringing a motorcycle and cash from her parents"),
            caused_death=("unknown", ""),
        ),
    ),
    (
        "During a quarrel over a boundary wall, the accused slapped the complainant twice and pushed him, "
        "causing a swelling on his cheek. The medical certificate records a simple injury.",
        _facts(
            caused_hurt=("true", "causing a swelling on his cheek"),
            caused_grievous_hurt=("false", "The medical certificate records a simple injury"),
            act_type=("intentional", "the accused slapped the complainant twice and pushed him"),
            caused_death=("false", "a simple injury"),
        ),
    ),
)


def _system_prompt() -> str:
    lines = [
        "You extract facts from a description of an Indian criminal case for a rule-based checker.",
        "For every predicate below, answer with its value and a short evidence quote copied word for word",
        "from the case text. Use \"unknown\" (with an empty evidence quote) whenever the text does not clearly",
        "settle the question — never guess, and never use outside knowledge about the case.",
        "",
        "Predicates (name: allowed values — question):",
    ]
    lines += [f"- {p.name}: {' | '.join(p.values)} — {p.question}" for p in PREDICATES]
    lines += [
        "",
        'Reply with one JSON object only, no prose and no code fences: {"<predicate>": {"value": "...", "evidence": "..."}, ...}',
        "Include every predicate exactly once.",
    ]
    return "\n".join(lines)


class FactExtractor:
    def __init__(self, llm: LLMClient, schema: FactSchema | None = None) -> None:
        self.llm = llm
        self.schema = schema or FactSchema()
        self.attempts = 0  # model calls made by the last extract()

    def _messages(self, case_text: str) -> list[Message]:
        msgs = [Message("system", _system_prompt())]
        for text, answer in FEW_SHOTS:
            msgs += [Message("user", f"CASE:\n{text}"), Message("assistant", answer)]
        return msgs + [Message("user", f"CASE:\n{case_text}")]

    @staticmethod
    def parse(raw: str) -> Any:
        """The JSON object in a reply, tolerating code fences and prose around it."""
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in the reply")
        return json.loads(text[start : end + 1])

    def _check(self, raw: str, case_text: str) -> tuple[ValidationResult | None, str]:
        try:
            result = self.schema.validate(self.parse(raw), case_text)
        except ValueError as exc:  # json.JSONDecodeError is a ValueError
            return None, f"the reply is not valid JSON ({exc})"
        return (result, "") if result.ok else (None, "; ".join(result.errors))

    def extract(self, case_text: str) -> ValidationResult:
        messages = self._messages(case_text)
        raw = self.llm.generate(messages)
        self.attempts = 1
        result, error = self._check(raw, case_text)
        if result is None:
            result = self.repair_json(raw, error, messages, case_text)
        return result

    def extract_consistent(self, case_text: str, runs: int = 2) -> ValidationResult:
        """Extract `runs` times and keep only values every run agrees on; a predicate the runs
        disagree on becomes "unknown" (with a warning). Reasoning models vary between runs even
        at temperature 0, and a verifier must not rest a verdict on a coin flip."""
        results = [self.extract(case_text) for _ in range(max(1, runs))]
        first = results[0]
        warnings = [w for r in results for w in r.warnings]
        for name, entry in first.facts.items():
            values = {r.facts[name]["value"] for r in results}
            if name == "act_type" and len(values) > 1 and values <= {"rash", "negligent"}:
                # courts routinely write "rash and negligent", and every encoded section that
                # needs one accepts either: not a disagreement that matters to any rule
                warnings.append(f"act_type: runs said {' and '.join(sorted(values))}; both satisfy the encoded rules")
                continue
            if len(values) > 1:
                warnings.append(f"{name}: extraction runs disagreed ({', '.join(sorted(values))}), treated as unknown")
                first.facts[name] = {"value": "unknown", "evidence": ""}
        first.warnings = list(dict.fromkeys(warnings))
        return first

    def repair_json(self, raw: str, error: str, messages: list[Message], case_text: str) -> ValidationResult:
        """The single retry: show the model its reply and what was wrong with it."""
        retry = messages + [
            Message("assistant", raw),
            Message("user", f"That reply could not be used: {error}. Reply again with only the corrected JSON object, "
                            "every predicate exactly once, values from the allowed lists."),
        ]
        repaired = self.llm.generate(retry)
        self.attempts = 2
        result, error2 = self._check(repaired, case_text)
        if result is None:
            raise ExtractionError(f"fact extraction failed after one repair attempt: {error2}")
        result.warnings.insert(0, f"the first extraction was invalid ({error}) and was repaired")
        return result
