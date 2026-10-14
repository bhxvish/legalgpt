"""Fact patterns with gold verdicts computed from the rules, plus counterfactual and paraphrase variants.

A *case* is an assignment of values to a section's elements, rendered as short text. Each element
has sentences for its satisfying value and for its contradicting value (two wordings each), and can
be left out (unknown) when nothing else in the story implies it. The gold verdict and per-element
statuses come from the published rule layout (docs/zk/verdict/layout.json, built from the Prolog
rules), never from hand labels, so a case is correct by construction *if* its sentences express
the intended values - which is what human reviewers check (benchmark/labelling/).

Variants of one base case form a group:
  base        every element satisfied (CONSISTENT), wording 0
  paraphrase  the same facts in wording 1 (same verdict expected: consistency)
  flip        one element changed to a contradicting value (INCONSISTENT expected)
  omit        one element's sentence removed (INSUFFICIENT expected)
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import REPO_ROOT
from app.zk.verdict_circuit import FactEncoder, VerdictLayout

NAMES = [("Ravi", "Sunil"), ("Meena", "Kavita"), ("Arjun", "Deepak"), ("Farhan", "Joseph"), ("Lakshmi", "Priya"),
         ("Harpreet", "Gurdeep"), ("Anil", "Vinod"), ("Sana", "Ayesha")]
DAYS = ["a Sunday morning", "the evening of 3 March", "a weekday afternoon", "the night of 14 August",
        "a rainy Monday", "the morning of 26 January", "a Friday evening", "the afternoon of 2 October"]


@dataclass
class Element:
    predicate: str
    good: str  # the value that satisfies the element
    good_text: list[str]  # two wordings
    bad: str | None = None  # a contradicting value (None: no flip test)
    bad_text: list[str] = field(default_factory=list)
    omittable: bool = True  # False when the rest of the story implies the value


@dataclass
class Case:
    id: str
    section: str
    group: str
    kind: str  # base | paraphrase | flip | omit
    change: dict[str, Any]
    text: str
    facts: dict[str, str]  # intended values (unknown when omitted)
    gold_verdict: str
    gold_elements: dict[str, str]


def _layout() -> VerdictLayout:
    return VerdictLayout.from_dict(json.loads((REPO_ROOT / "docs" / "zk" / "verdict" / "layout.json").read_text(encoding="utf-8")))


# ---------------------------------------------------------------- scenario frames per section
# Frames supply the story-specific wording; {A} is the accused, {V} the other person.

NEGLIGENCE_FRAMES = [  # used by 304A, 337, 338: (how it happened rashly, how it happened by pure accident)
    {"rash": ["{A} drove a lorry at a very high speed through a crowded market, overtaking recklessly, and hit {V}.",
              "{A} raced a lorry far above the speed limit through a busy market, swerving between vehicles, and struck {V}."],
     "accident": ["{A} was driving a lorry slowly and carefully when a hidden brake defect, which no inspection could have found, made it swerve into {V}.",
                  "Although {A} drove the lorry carefully and within the limit, a sudden steering failure that nobody could have foreseen sent it into {V}."]},
    {"rash": ["{A}, a contractor, demolished an old wall without any barricade or warning while people were walking past, and it fell on {V}.",
              "Without putting up any barrier or warning sign, {A} knocked down an old wall next to a busy footpath, and the wall came down on {V}."],
     "accident": ["{A}, a contractor, had fenced off the site and posted guards, but a sudden earthquake brought the wall down on {V}.",
                  "Despite barricades and every safety step taken by {A}, an unexpected earthquake made the wall collapse onto {V}."]},
    {"rash": ["{A} fired a gun into the air again and again in the middle of a crowded wedding procession, and a bullet struck {V}.",
              "In a packed wedding crowd, {A} kept firing a pistol upwards in celebration, ignoring guests who begged him to stop, and {V} was hit."],
     "accident": ["{A} was carrying an unloaded-looking antique gun at a wedding when a hidden manufacturing fault made it discharge by itself and hit {V}.",
                  "An old gun that {A} was holding at a wedding went off on its own because of a concealed factory defect, and the shot hit {V}."]},
    {"act": "negligent",
     "rash": ["{A}, a nurse, injected a drug into {V} without checking the prescription chart, which clearly listed {V}'s allergy to it.",
              "Without reading {V}'s chart, which recorded an allergy to the drug, {A}, a nurse, gave {V} an injection of it."],
     "accident": ["{A}, a nurse, checked the chart and gave {V} the prescribed drug, but the factory had sealed a different drug into the labelled vial, which nobody could have detected.",
                  "{A} followed the chart exactly, but the vial had been wrongly filled at the factory in a way no one could have spotted, and {V} received the wrong drug."]},
    {"act": "negligent",
     "rash": ["{A}, the lifeguard on duty, left the crowded pool unattended for an hour to chat on the phone, and {V} got into difficulty in the deep end.",
              "Instead of watching the busy pool, {A}, the lifeguard, went off for an hour, and {V} got into trouble in the deep end."],
     "accident": ["{A}, the lifeguard, was watching closely when {V} suffered a sudden seizure underwater that no one could have foreseen.",
                  "Although {A} was alert at the poolside, {V} had an unforeseeable seizure under the water."]},
    {"rash": ["{A}, a crane operator, swung a heavy load over a busy street without checking below, and it struck {V}.",
              "Ignoring the people walking below, {A} moved a crane's heavy load across a crowded street and it hit {V}."],
     "accident": ["{A} was operating a crane by the book, with the street closed, when a cable that had passed every test snapped and the load struck {V}.",
                  "With the area cordoned off and every check done, {A}'s crane cable broke without warning and the load hit {V}."]},
]

HURT_FRAMES = [  # used by 323: (deliberate act, accident)
    {"place": "a ration shop", "intent": ["{A} deliberately slapped {V} twice.", "{A} hit {V} on purpose, slapping {V} hard across the face."],
     "accident": ["While turning around, {A} accidentally knocked into {V} without even seeing {V}.",
                  "{A} slipped on the wet floor and fell against {V} by accident."]},
    {"place": "a bus stop", "intent": ["{A} punched {V} in the stomach on purpose.", "{A} intentionally struck {V} with a fist."],
     "accident": ["A sudden jolt of the crowd pushed {A} into {V}, entirely by accident.",
                  "{A} lost balance when the crowd surged and accidentally bumped hard into {V}."]},
    {"place": "a cricket ground", "intent": ["Angry at losing, {A} deliberately hit {V} on the arm with a bat.",
                                            "{A} swung the bat at {V} on purpose and struck {V}'s arm."],
     "accident": ["{A} was batting normally when the bat slipped out of {A}'s hands by accident and hit {V}.",
                  "During ordinary play, the bat flew out of {A}'s grip accidentally and struck {V}."]},
    {"place": "a parking lot", "intent": ["{A} deliberately kicked {V} in the leg.", "{A} kicked {V} on purpose."],
     "accident": ["{A} was reversing a heavy trolley and it rolled into {V}'s leg by accident.",
                  "A trolley {A} was pushing slipped on the slope and accidentally struck {V}."]},
    {"place": "a village meeting", "intent": ["{A} deliberately threw a stone at {V}, hitting {V}.", "{A} aimed a stone at {V} on purpose and struck {V}."],
     "accident": ["A stone flicked up by a passing cart hit {V}; {A}, who was standing nearby, played no deliberate part.",
                  "{A} kicked a football in a game and it accidentally struck {V}, who was standing far off the pitch."]},
    {"place": "a wedding hall", "intent": ["{A} deliberately pushed {V} against a wall.", "{A} shoved {V} into the wall on purpose."],
     "accident": ["{A} tripped over a cable and fell onto {V} by accident.", "{A} stumbled over a loose carpet and accidentally fell against {V}."]},
]

THEFT_FRAMES = [
    {"item": "mobile phone", "place": "a crowded train", "spot": "bag"},
    {"item": "bicycle", "place": "a housing colony", "spot": "courtyard"},
    {"item": "gold chain", "place": "a temple", "spot": "handbag"},
    {"item": "laptop", "place": "an office", "spot": "desk"},
    {"item": "wallet", "place": "a gym", "spot": "locker"},
    {"item": "scooter", "place": "a market", "spot": "parking spot"},
]

DRIVE_FRAMES = [{"vehicle": "car", "town": "Pune"}, {"vehicle": "motorcycle", "town": "Lucknow"},
                {"vehicle": "bus", "town": "Kochi"}, {"vehicle": "tractor", "town": "Nashik"},
                {"vehicle": "truck", "town": "Jaipur"}, {"vehicle": "auto-rickshaw", "town": "Chennai"}]

DOWRY_FRAMES = [{"W": "Sunita", "H": "Rajesh", "item": "a car and cash"}, {"W": "Pooja", "H": "Manoj", "item": "a motorcycle"},
                {"W": "Fatima", "H": "Imran", "item": "gold jewellery"}, {"W": "Anjali", "H": "Vikas", "item": "two lakh rupees"},
                {"W": "Rekha", "H": "Suresh", "item": "a refrigerator"}, {"W": "Divya", "H": "Karthik", "item": "a plot of land"}]


def _sections() -> dict[str, dict[str, Any]]:
    """Section -> {intro, frames, elements(frame) -> [Element]}."""

    def neg_elements(outcome: str):
        def build(fr):
            els = []
            if outcome == "death":
                els.append(Element("caused_death", "true", ["{V} died as a result.", "{V} did not survive and was declared dead at the hospital."],
                                   "false", ["{V} was injured but recovered fully.", "{V} survived after treatment in hospital."]))
            elif outcome == "hurt":
                els.append(Element("caused_hurt", "true", ["{V} suffered bruises and a cut that healed within a week.",
                                                            "{V} was left with painful bruises and a small wound."],
                                   "false", ["{V} escaped without any injury at all.", "{V} was completely unhurt."]))
            else:
                els.append(Element("caused_grievous_hurt", "true", ["{V} suffered a fractured leg and spent a month in hospital.",
                                                                     "{V}'s arm was broken in two places."],
                                   "false", ["{V} suffered only a minor scratch that healed in two days, with no fracture or serious injury.",
                                             "{V} got only a small graze; there was no fracture or other serious injury."]))
            els.insert(0, Element("act_type", fr.get("act", "rash"), fr["rash"], "accidental", fr["accident"], omittable=False))
            if outcome == "death":
                els.append(Element("intent_to_kill", "false", ["{A} had no intention of killing or harming {V}.",
                                                               "Nothing suggests {A} wanted to hurt {V} in any way."],
                                   "true", ["{A} had planned this, wanting {V} dead.", "{A} meant to kill {V}."],
                                   omittable=False))  # "never occurred to A that someone might die" implies no intent
                els.append(Element("knowledge_likely_death", "false", ["{A} did not realise that this could kill anyone.",
                                                                        "It never occurred to {A} that someone might die."],
                                   "true", ["{A} knew that this was very likely to kill someone.",
                                            "{A} was fully aware that the act would probably cause a death."]))
            else:
                els.append(Element("endangered_safety", "true", ["The act put the lives of the people around in danger.",
                                                                  "Everyone nearby was put at serious risk."], omittable=False))
            return els
        return build

    return {
        "304A": {"intro": "On {day}, an incident took place involving {A} and {V}.", "frames": NEGLIGENCE_FRAMES, "elements": neg_elements("death")},
        "337": {"intro": "On {day}, an incident took place involving {A} and {V}.", "frames": NEGLIGENCE_FRAMES, "elements": neg_elements("hurt")},
        "338": {"intro": "On {day}, an incident took place involving {A} and {V}.", "frames": NEGLIGENCE_FRAMES, "elements": neg_elements("grievous")},
        "323": {"intro": "On {day}, {A} and {V} had an argument at {place}.", "frames": HURT_FRAMES, "elements": lambda fr: [
            Element("caused_hurt", "true", ["{V} suffered pain and swelling.", "{V} was left with a painful bruise."],
                    "false", ["{V} felt no pain and had no injury at all.", "{V} was not hurt in the slightest."],
                    omittable=False),  # the act sentences describe striking, which already implies hurt
            Element("act_type", "intentional", fr["intent"], "accidental", fr["accident"]),
        ]},
        "279": {"intro": "On {day}, {A} took out a {vehicle}.", "frames": DRIVE_FRAMES, "elements": lambda fr: [
            Element("drove_vehicle", "true", ["{A} was the one driving the {vehicle}.", "{A} was behind the controls of the {vehicle}."],
                    omittable=False),
            Element("on_public_way", "true", ["This happened on the main public road running through {town}.",
                                              "It took place on a busy public street in {town}."],
                    "false", ["This happened entirely inside the locked compound of a private factory that the public cannot enter.",
                              "It all took place inside a closed private estate, which is not open to the public."]),
            Element("act_type", "rash", ["{A} drove at a dangerously high speed, weaving between vehicles and ignoring signals.",
                                          "{A} raced far above the speed limit, overtaking recklessly."],
                    "accidental", ["{A} was driving slowly and carefully, obeying every rule, when a hidden brake defect that no inspection could have found made the {vehicle} swerve.",
                                   "{A} drove with full care and within the limit, but a sudden steering failure nobody could have foreseen made the {vehicle} veer off course."]),
            Element("endangered_safety", "true", ["People nearby had to jump out of the way to avoid being hit; their lives were put in danger.",
                                                  "Several people were nearly run over and were put in real danger."],
                    "false", ["There was nobody else anywhere nearby, and no person was put at any risk.",
                              "The area was completely empty, so no one's safety was ever at risk."],
                    omittable=False),  # "drove at a dangerously high speed, weaving between vehicles" implies danger
        ]},
        "379": {"intro": "On {day}, {A} was at {place}.", "frames": THEFT_FRAMES, "elements": lambda fr: [
            Element("movable_property", "true", ["The {item} was the kind of thing one can carry away.", "It was an ordinary {item} that could be picked up and carried."],
                    omittable=False),
            Element("taken_from_possession", "true", ["The {item} belonged to {V} and was kept in {V}'s {spot}.",
                                                      "{V} owned the {item} and had left it in {V}'s {spot}."],
                    "false", ["The {item} had been thrown away by its owner and was lying abandoned, belonging to no one.",
                              "Its owner had discarded the {item}; it lay abandoned and in no one's possession."],
                    omittable=False),  # "{V} had not given anyone permission to take it" implies it was {V}'s
            Element("property_moved", "true", ["{A} picked up the {item} and carried it away.", "{A} took the {item} and walked off with it."],
                    "false", ["{A} looked at the {item} but left it exactly where it was.", "{A} never touched the {item}; it stayed where it was."]),
            Element("without_consent", "true", ["{V} had not given anyone permission to take it.", "{V} had never agreed to anyone taking it."],
                    "false", ["{V} had told {A} to take it and use it.", "{V} had willingly lent it to {A}."]),
            Element("dishonest_intention", "true", ["{A} meant to keep the {item} and sell it.", "{A} planned to sell the {item} for money."],
                    "false", ["{A} honestly believed the {item} was {A}'s own.", "{A} took it by mistake, sincerely thinking it belonged to {A}."]),
        ]},
        "304B": {"intro": "{W} was married to {H}.", "frames": DOWRY_FRAMES, "elements": lambda fr: [
            Element("woman_unnatural_death", "true", ["{W} died of burns at her matrimonial home.", "{W} was found hanging in her in-laws' house."],
                    "false", ["{W} died of a long illness in hospital, and the doctors found nothing unusual about her death.",
                              "{W} died of natural causes from a heart condition that had been treated for years."], omittable=False),
            Element("within_seven_years_of_marriage", "true", ["She died three years after the wedding.", "Her death came in the fourth year of the marriage."],
                    "false", ["She died twelve years after the wedding.", "Her death came in the fifteenth year of the marriage."]),
            Element("cruelty_by_husband_or_relative", "true", ["{H} and his mother repeatedly beat and harassed her.",
                                                               "Her husband's family constantly abused and tormented her."], omittable=False),
            Element("cruelty_soon_before_death", "true", ["The last beating was only a few days before she died.",
                                                          "She was beaten again just a week before her death."],
                    "false", ["The ill-treatment had stopped completely five years before her death.",
                              "All the harassment had ended many years before she died."]),
            Element("dowry_demand", "true", ["They demanded {item} from her parents as dowry, and the ill-treatment was over this demand.",
                                             "The abuse was over their demand for {item} as dowry."],
                    "false", ["The quarrels had nothing to do with dowry; no dowry was ever demanded.",
                              "No one ever asked her family for dowry or anything else; the quarrels were about other matters."]),
        ]},
    }


def _render(intro: str, sentences: list[str], slots: dict[str, str]) -> str:
    return " ".join(s.format(**slots) for s in [intro, *sentences])


def generate(frames_per_section: int | None = None) -> list[Case]:
    layout, enc = _layout(), FactEncoder()
    cases: list[Case] = []
    for section, spec in _sections().items():
        frames = spec["frames"][: frames_per_section or len(spec["frames"])]
        for fi, frame in enumerate(frames):
            a, v = NAMES[(fi * 3 + len(section)) % len(NAMES)]
            slots = {**frame, "A": a, "V": v, "day": DAYS[(fi + len(section)) % len(DAYS)]}
            els = spec["elements"](frame)
            group = f"{section}-f{fi}"

            def make(kind: str, wording: int, change: dict | None = None):
                facts, sentences = {}, []
                for el in els:
                    if change and change["predicate"] == el.predicate:
                        if change["to"] == "unknown":
                            facts[el.predicate] = "unknown"
                            continue
                        facts[el.predicate] = el.bad
                        sentences.append(el.bad_text[wording])
                    else:
                        facts[el.predicate] = el.good
                        sentences.append(el.good_text[wording])
                decoded = layout.decode(layout.evaluate(enc.encode(facts)))[section]
                cid = f"{group}-{kind}" + (f"-{change['predicate']}" if change else "")
                cases.append(Case(cid, section, group, kind, change or {}, _render(spec["intro"], sentences, slots), facts,
                                  decoded["status"], {e["predicate"]: e["status"] for e in decoded["elements"]}))

            make("base", 0)
            make("paraphrase", 1)
            for el in els:
                if el.bad is not None:
                    make("flip", 0, {"predicate": el.predicate, "from": el.good, "to": el.bad})
                if el.omittable:
                    make("omit", 0, {"predicate": el.predicate, "from": el.good, "to": "unknown"})
    return cases


def write_jsonl(cases: list[Case], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
