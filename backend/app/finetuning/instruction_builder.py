"""InstructionBuilder: labelled judgment sentences -> instruction-tuning examples.

Four example types per case, built only from that case's own labelled sentences:

- facts_to_law       facts -> the provisions applied and how (Law Applied + Ruling sentences)
- facts_args_to_ruling  facts + arguments -> the court's decision (Ruling sentences)
- grounded_answer    the exact SOURCES/QUESTION prompt the chat app sends (PromptBuilder), answered
                     only from the numbered sources with inline [n] citations
- grounded_refusal   sources from one case, a question about an unrelated case -> the exact
                     "I don't know" reply the legal system prompt requires (built across cases)

The last two teach the tuned model the behaviour Module 0 demands at inference time.
"""

import random
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from app.annotation.models import LabeledSentence
from app.core.models import SourceEvidence
from app.core.prompt_builder import LEGAL_SYSTEM_PROMPT, NOT_COVERED_REPLY, PromptBuilder

ANALYSIS_SYSTEM_PROMPT = (
    "You are LegalGPT, an assistant for Indian criminal law. Answer precisely and only from the "
    "material given. This is legal information, not legal advice."
)
SPLITS = ("train", "val", "test")
_SECTION = re.compile(r"\b(?:sections?|s\.|u/s\.?)\s*(\d{1,3}[A-Z]{0,2})", re.I)


@dataclass
class InstructionExample:
    instruction: str
    input: str
    output: str
    source_case_id: str
    task: str = ""
    system: str = ANALYSIS_SYSTEM_PROMPT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstructionBuilder:
    def __init__(self, max_input_chars: int = 2400, max_output_chars: int = 1200, max_sources: int = 6, seed: int = 13) -> None:
        self.max_input_chars = max_input_chars
        self.max_output_chars = max_output_chars
        self.max_sources = max_sources
        self.seed = seed

    # ------------------------------------------------------------- examples

    @staticmethod
    def _by_role(sentences: Sequence[LabeledSentence]) -> dict[str, list[LabeledSentence]]:
        roles: dict[str, list[LabeledSentence]] = defaultdict(list)
        for s in sorted(sentences, key=lambda s: s.idx):
            roles[s.label].append(s)
        return roles

    @staticmethod
    def _take(sentences: Sequence[LabeledSentence], budget: int) -> str:
        """Whole sentences, in order, up to `budget` characters."""
        out: list[str] = []
        used = 0
        for s in sentences:
            if out and used + len(s.text) + 1 > budget:
                break
            out.append(s.text)
            used += len(s.text) + 1
        return " ".join(out)

    def build_from_case(self, sentences: Sequence[LabeledSentence]) -> list[InstructionExample]:
        if not sentences:
            return []
        case_id = sentences[0].case_id
        roles = self._by_role(sentences)
        facts, law, ruling = roles.get("Facts", []), roles.get("Law Applied", []), roles.get("Ruling", [])
        args, precedent = roles.get("Argument", []), roles.get("Precedent", [])
        examples: list[InstructionExample] = []

        if facts and (law or ruling):
            examples.append(InstructionExample(
                instruction="Given the facts of this Indian criminal case, identify the legal provisions that apply and explain how they apply.",
                input=self._take(facts, self.max_input_chars),
                output=self._take(law + ruling, self.max_output_chars),
                source_case_id=case_id,
                task="facts_to_law",
            ))
        if facts and args and ruling:
            half = self.max_input_chars // 2
            examples.append(InstructionExample(
                instruction="Based on the facts and the parties' arguments, state the court's decision and its reasoning.",
                input=f"Facts: {self._take(facts, half)}\n\nArguments: {self._take(args, half)}",
                output=self._take(ruling, self.max_output_chars),
                source_case_id=case_id,
                task="facts_args_to_ruling",
            ))
        grounded = self._grounded_answer(case_id, facts, law, precedent, ruling)
        if grounded:
            examples.append(grounded)
        return examples

    def _sources(self, case_id: str, pool: list[LabeledSentence], rng: random.Random) -> list[SourceEvidence]:
        picked = rng.sample(pool, min(self.max_sources, len(pool)))
        return [
            SourceEvidence(marker=i, chunk_id=s.sentence_id, doc_id=case_id, citation_path=f"{case_id} > {s.label}",
                           text=s.text, similarity=1.0, score=1.0, role=s.label)
            for i, s in enumerate(picked, start=1)
        ]

    def _grounded_answer(self, case_id: str, facts, law, precedent, ruling) -> InstructionExample | None:
        """Mirror of the inference prompt: numbered sources + question; the answer restates the
        relevant sources, each sentence cited with its [n] marker (law first, then the ruling)."""
        if not facts or not (law or ruling):
            return None
        rng = random.Random(f"{self.seed}:{case_id}:grounded")
        pool = law[:2] + ruling[:3] + precedent[:1]
        sources = self._sources(case_id, pool, rng)
        cited = sorted((s for s in sources if s.role in ("Law Applied", "Ruling")),
                       key=lambda s: (s.role != "Law Applied", s.marker))[:3]
        question = (f"{self._take(facts[:3], 600)} Which provisions of law apply to these facts, "
                    "and what did the court decide?")
        return InstructionExample(
            instruction=f"SOURCES:\n{PromptBuilder.format_sources(sources)}\n\nQUESTION: {question}",
            input="",
            output=" ".join(f"{s.text} [{s.marker}]" for s in cited),
            source_case_id=case_id,
            task="grounded_answer",
            system=LEGAL_SYSTEM_PROMPT,
        )

    def build_refusals(self, cases: dict[str, Sequence[LabeledSentence]]) -> list[InstructionExample]:
        """For each case, a question about its facts paired with sources from a case that cites
        none of the same sections: the correct answer is the not-covered reply."""
        rng = random.Random(f"{self.seed}:refusals")
        ids = sorted(cases)
        sections = {c: {m.upper() for s in cases[c] for m in _SECTION.findall(s.text)} for c in ids}
        out: list[InstructionExample] = []
        for case_id in ids:
            facts = self._by_role(cases[case_id]).get("Facts", [])
            others = [o for o in ids if o != case_id and not (sections[o] & sections[case_id])]
            if not facts or not others:
                continue
            other = rng.choice(others)
            roles = self._by_role(cases[other])
            pool = roles.get("Law Applied", [])[:3] + roles.get("Precedent", [])[:3]
            if not pool:
                continue
            sources = self._sources(other, pool, rng)
            question = f"{self._take(facts[:2], 500)} What offence is made out on these facts?"
            out.append(InstructionExample(
                instruction=f"SOURCES:\n{PromptBuilder.format_sources(sources)}\n\nQUESTION: {question}",
                input="",
                output=NOT_COVERED_REPLY,
                source_case_id=case_id,  # the case the question is about; sources come from `other`
                task="grounded_refusal",
                system=LEGAL_SYSTEM_PROMPT,
            ))
        return out

    def build_corpus(self, sentences: Sequence[LabeledSentence]) -> list[InstructionExample]:
        cases: dict[str, list[LabeledSentence]] = defaultdict(list)
        for s in sentences:
            cases[s.case_id].append(s)
        examples = [ex for c in sorted(cases) for ex in self.build_from_case(cases[c])]
        return examples + self.build_refusals(cases)

    # ----------------------------------------------------------- formatting

    @staticmethod
    def messages(ex: InstructionExample, with_answer: bool = True) -> list[dict[str, str]]:
        user = ex.instruction if not ex.input else f"{ex.instruction}\n\n{ex.input}"
        msgs = [{"role": "system", "content": ex.system}, {"role": "user", "content": user}]
        if with_answer:
            msgs.append({"role": "assistant", "content": ex.output})
        return msgs

    def format_dataset_entry(self, ex: InstructionExample, tokenizer: Any | None = None) -> str:
        """Serialize into the base model's chat template (Qwen2.5's when its tokenizer is given),
        or a plain instruction format without one."""
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(self.messages(ex), tokenize=False)
        parts = [f"### System\n{ex.system}", f"### Instruction\n{ex.instruction}"]
        if ex.input:
            parts.append(f"### Input\n{ex.input}")
        parts.append(f"### Response\n{ex.output}")
        return "\n\n".join(parts)

    # ---------------------------------------------------------------- split

    @staticmethod
    def split_by_case(
        examples: Sequence[InstructionExample], ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 13
    ) -> dict[str, list[InstructionExample]]:
        """Whole cases go to one split, so no case appears in more than one. A refusal example
        belongs to the case its question is about; its sources come from another case, which
        does not leak that case's question or answer."""
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"ratios must sum to 1, got {ratios}")
        cases = sorted({ex.source_case_id for ex in examples})
        random.Random(seed).shuffle(cases)
        n_val, n_test = round(len(cases) * ratios[1]), round(len(cases) * ratios[2])
        if len(cases) >= 3:
            n_val, n_test = max(n_val, 1 if ratios[1] else 0), max(n_test, 1 if ratios[2] else 0)
        assign = {c: "test" for c in cases[:n_test]}
        assign.update({c: "val" for c in cases[n_test : n_test + n_val]})
        assign.update({c: "train" for c in cases[n_test + n_val :]})
        out: dict[str, list[InstructionExample]] = {s: [] for s in SPLITS}
        for ex in examples:
            out[assign[ex.source_case_id]].append(ex)
        return out
