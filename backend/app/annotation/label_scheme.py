"""LabelScheme: the rhetorical-role labels annotators assign to judgment sentences.

This module is the single source of truth for labels, their definitions, keyboard shortcuts
and worked examples. docs/annotation_guideline.md is generated from it
(scripts/build_guideline.py) and a test fails if the two drift apart.

Five roles come from the LLD (Facts, Law Applied, Precedent, Argument, Ruling). A sixth,
None, was added (decision recorded in PLANNING.md) for headers, cause titles and procedural
boilerplate that fit none of the five; Phase 3 may train with it or filter it out.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelDef:
    name: str
    shortcut: str
    definition: str
    guidance: str
    examples: tuple[str, ...]


# Worked examples are representative sentences written in the style of Indian criminal
# judgments to illustrate each role. They are not quotations from real judgments.
_LABELS: tuple[LabelDef, ...] = (
    LabelDef(
        name="Facts",
        shortcut="1",
        definition=(
            "The events of the case and its procedural history as the court records them: who did "
            "what, when and where, and how the matter reached this court."
        ),
        guidance=(
            "Includes the prosecution story as narrated by the court, the FIR, investigation, charges "
            "framed, and what the lower courts decided. If the sentence reports what a party argues "
            "about the facts, it is Argument instead."
        ),
        examples=(
            "On the night of 14 March 2009, the deceased was returning home on his motorcycle when the "
            "truck driven by the appellant struck him from behind.",
            "An FIR was registered at the Kotwali police station under Sections 279 and 304A of the "
            "Indian Penal Code, and after investigation a charge-sheet was filed against the accused.",
            "The trial court convicted the appellant and sentenced him to two years' rigorous "
            "imprisonment, which the Sessions Court affirmed in appeal.",
        ),
    ),
    LabelDef(
        name="Law Applied",
        shortcut="2",
        definition=(
            "The text, elements or meaning of a statutory provision (such as an IPC section) that the "
            "court sets out or applies, stated as law rather than as a finding on these facts."
        ),
        guidance=(
            "Quoting or paraphrasing a section, listing its ingredients, or explaining what a "
            "provision requires. When the court applies the law to these facts and reaches a "
            "conclusion, use Ruling. When the law comes from an earlier case, use Precedent."
        ),
        examples=(
            "Section 304A of the Indian Penal Code punishes whoever causes the death of any person by "
            "doing any rash or negligent act not amounting to culpable homicide.",
            "To attract Section 498A, the prosecution must establish that the woman was subjected to "
            "cruelty by her husband or a relative of her husband.",
            "Under Section 34, when a criminal act is done by several persons in furtherance of their "
            "common intention, each of them is liable as if it were done by him alone.",
        ),
    ),
    LabelDef(
        name="Precedent",
        shortcut="3",
        definition=(
            "References to earlier judgments: the case cited, the principle it laid down, or the "
            "court's reason for following or distinguishing it."
        ),
        guidance=(
            "Any sentence whose main content is what another court held. A sentence that only names "
            "a case in passing while stating this court's conclusion is Ruling."
        ),
        examples=(
            "In Virsa Singh v. State of Punjab, AIR 1958 SC 465, this Court explained what the "
            "prosecution must prove to bring a case under clause thirdly of Section 300.",
            "The principles governing medical negligence under Section 304A were laid down in Jacob "
            "Mathew v. State of Punjab, (2005) 6 SCC 1.",
            "That decision turned on a sudden quarrel and has no application to a premeditated "
            "assault such as the present one.",
        ),
    ),
    LabelDef(
        name="Argument",
        shortcut="4",
        definition=(
            "Submissions made by the parties' counsel — the prosecution, the defence, the "
            "complainant or an amicus — as reported by the court."
        ),
        guidance=(
            "Usually signalled by 'learned counsel submitted', 'it was contended', 'the State "
            "argues'. Keep the label even if the submission recites facts or law; the court is "
            "reporting someone else's position, not deciding."
        ),
        examples=(
            "Learned counsel for the appellant contended that the accident occurred because the "
            "deceased suddenly swerved into the path of the truck.",
            "It was submitted on behalf of the State that the injuries were caused on vital parts of "
            "the body and the intention to kill was evident.",
            "The defence argued that the dying declaration was unreliable as the deceased had "
            "suffered 90% burns and could not have been in a fit state of mind.",
        ),
    ),
    LabelDef(
        name="Ruling",
        shortcut="5",
        definition=(
            "The present court's own reasoning, findings and decision: how it evaluates the evidence, "
            "applies the law to these facts, and disposes of the case."
        ),
        guidance=(
            "Includes appreciation of evidence ('we find the testimony of PW-2 trustworthy'), "
            "conclusions on each issue, and the final order (conviction, acquittal, sentence, appeal "
            "allowed or dismissed)."
        ),
        examples=(
            "We find that the evidence of the eyewitnesses is consistent and is corroborated by the "
            "medical evidence.",
            "The prosecution has failed to prove that the appellant was driving rashly or "
            "negligently, and the benefit of doubt must go to him.",
            "The appeal is accordingly allowed, and the conviction and sentence of the appellant are "
            "set aside.",
        ),
    ),
    LabelDef(
        name="None",
        shortcut="6",
        definition=(
            "Headers, cause titles, coram, dates, page furniture and procedural boilerplate that "
            "carry none of the five roles."
        ),
        guidance=(
            "Use sparingly. If a sentence has any substantive content, choose the closest of the "
            "five roles instead."
        ),
        examples=(
            "IN THE SUPREME COURT OF INDIA CRIMINAL APPELLATE JURISDICTION",
            "Heard learned counsel for the parties.",
            "Leave granted.",
        ),
    ),
)


class LabelScheme:
    LABELS: tuple[str, ...] = tuple(d.name for d in _LABELS)
    _BY_NAME: dict[str, LabelDef] = {d.name: d for d in _LABELS}

    @classmethod
    def validate(cls, label: str) -> bool:
        return label in cls._BY_NAME

    @classmethod
    def describe(cls, label: str) -> str:
        """One-sentence definition, shown as the tooltip in the annotation UI."""
        if not cls.validate(label):
            raise ValueError(f"unknown label {label!r}; expected one of {', '.join(cls.LABELS)}")
        return cls._BY_NAME[label].definition

    @classmethod
    def definition(cls, label: str) -> LabelDef:
        cls.describe(label)  # validates
        return cls._BY_NAME[label]

    @classmethod
    def as_dict(cls) -> list[dict[str, object]]:
        return [
            {"name": d.name, "shortcut": d.shortcut, "description": d.definition, "guidance": d.guidance}
            for d in _LABELS
        ]
