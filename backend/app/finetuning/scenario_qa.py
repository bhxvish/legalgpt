"""ScenarioQaBuilder: training examples for everyday fact patterns ("someone sold me a fake phone").

The v0.4 adapter learned to answer questions that name an offence, but refused 82% of everyday
scenarios, also when the right section was among its sources: it had never seen a question whose
words do not appear in the statute. These examples use the inference setting exactly: a scenario,
the sources the real Retriever returns for it (offline, and with the LLM rewrite when one is cached),
and an answer built from the matching section's own text, so it is correct by construction:

    The facts appear to fit <offence> under Section N of the IPC [n]. Section N makes it an offence
    for anyone who <conduct> [n]. Whether it applies depends on what can be shown about the facts ...

Questions come only from benchmark scenarios outside the 60 held-out ones (run_baselines.py,
seed 7), so the benchmark still measures unseen questions. Their labels are silver (AI-written):
only examples whose retrieved sources contain a labelled section are used, and no refusals are
added (refusals taught v0.2 to refuse; the IPC Q&A set already has 10%).
"""

import random
from collections.abc import Callable, Sequence

from app.core.models import Chunk, SourceEvidence
from app.core.prompt_builder import LEGAL_SYSTEM_PROMPT, NOT_COVERED_REPLY, PromptBuilder
from app.finetuning.instruction_builder import InstructionExample
from app.finetuning.ipc_qa import body_of, lower_first, offence_parts

CAVEAT = ("Whether it applies depends on what can be shown about the facts, especially any intention, "
          "knowledge or dishonesty the section requires.")


class ScenarioQaBuilder:
    def __init__(self, retriever, chunks: Sequence[Chunk], seed: int = 13) -> None:
        self.retriever = retriever
        self.chunks = {c.chunk_id: c for c in chunks}
        self.rng = random.Random(seed)

    @staticmethod
    def _section_source(sources: Sequence[SourceEvidence], section: str) -> SourceEvidence | None:
        """The source holding the section's own text (not an illustration or amendment), else any."""
        same = [s for s in sources if s.section == section]
        return next((s for s in same if s.citation_path.endswith(f"Section {section}")), same[0] if same else None)

    def answer(self, sources: Sequence[SourceEvidence], acceptable: Sequence[str]) -> str | None:
        """Built from the first labelled section among the sources (best ranked), plus a pointer to a
        second one if present (a definition and its punishment section, e.g. 415 and 420)."""
        found = [s.section for s in sources if s.section in set(acceptable)]
        found = list(dict.fromkeys(found))
        if not found:
            return None
        src = self._section_source(sources, found[0])
        chunk = self.chunks.get(src.chunk_id)
        if chunk is None:
            return None
        t = (chunk.title or f"Section {chunk.section}").strip().rstrip(".")
        m = src.marker
        parts = offence_parts(chunk)
        if parts is not None:
            conduct, punishment = parts
            text = (f"The facts appear to fit {lower_first(t)} under Section {chunk.section} of the IPC [{m}]. "
                    f"Section {chunk.section} makes it an offence for anyone who{conduct} [{m}]. {CAVEAT}\n"
                    f"Punishment: {punishment} [{m}].")
        else:
            text = (f"The facts appear to fit Section {chunk.section} of the IPC ({t}) [{m}], which provides: "
                    f"{body_of(chunk)} [{m}]\n{CAVEAT}")
        if len(found) > 1:
            other = self._section_source(sources, found[1])
            other_t = (self.chunks[other.chunk_id].title or "").strip().rstrip(".") if other.chunk_id in self.chunks else ""
            text += f"\nSee also Section {other.section}" + (f" ({other_t})" if other_t else "") + f" [{other.marker}]."
        return text

    def build(self, scenarios: Sequence[dict], rewrite_for: Callable[[str], object] | None = None,
              refusal_share: float = 0.0) -> list[InstructionExample]:
        """`scenarios`: {"id", "question", "acceptable"}. `rewrite_for(question)`: a cached LLM rewrite
        or None; when it changes the sources, the rewritten setting is a second example.
        `refusal_share`: the share of scenario examples that are refusals - sources retrieved for the
        scenario that contain none of its labelled sections. v0.5 had none and learned to force-fit the
        nearest offence (chain snatching -> rash driving), so v0.6 uses about a quarter."""
        out: list[InstructionExample] = []
        misses: list[InstructionExample] = []
        for sc in scenarios:
            settings = [None]
            rw = rewrite_for(sc["question"]) if rewrite_for else None
            if rw is not None:
                settings.append(rw)
            seen: set[tuple[str, ...]] = set()
            for rewrite in settings:
                sources = self.retriever.retrieve(sc["question"], rewrite=rewrite)
                key = tuple(s.chunk_id for s in sources)
                if not sources or key in seen:
                    continue
                seen.add(key)
                text = self.answer(sources, sc["acceptable"])
                ex = InstructionExample(
                    instruction=f"SOURCES:\n{PromptBuilder.format_sources(sources)}\n\nQUESTION: {sc['question']}",
                    input="", output=text or NOT_COVERED_REPLY, source_case_id=f"scenario:{sc['id']}",
                    task="scenario_answer" if text else "scenario_refusal", system=LEGAL_SYSTEM_PROMPT,
                )
                (out if text else misses).append(ex)
        if refusal_share > 0 and misses:
            n = round(len(out) * refusal_share / (1 - refusal_share))
            out += self.rng.sample(misses, min(n, len(misses)))
        return out
