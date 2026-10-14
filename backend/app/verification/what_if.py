"""What would change a fact-check verdict? Counterfactual explanations computed from the rules.

The verdict is a fixed function of the element statuses (rules/core.pl, compute_verdict/3): any
violated element -> INCONSISTENT, else any missing -> INSUFFICIENT, else CONSISTENT. So the effect
of changing one element can be computed exactly, with no model call:

  missing facts   for INSUFFICIENT: the elements the text does not settle, with the plain question
                  the facts would have to answer (from the shared schema)
  blockers        for INCONSISTENT: the contradicted elements, and the verdict if each were
                  satisfied instead (CONSISTENT only if it is the only problem)
  sensitivity     for CONSISTENT: every element the verdict rests on (changing any one breaks it)
"""

from typing import Any

from app.verification.fact_schema import BY_NAME


def verdict_of(statuses: list[str]) -> str:
    if "violated" in statuses:
        return "INCONSISTENT"
    if "missing" in statuses:
        return "INSUFFICIENT"
    return "CONSISTENT"


def what_if(elements: list[dict[str, Any]]) -> dict[str, Any]:
    """elements: [{predicate, description, required, found, status}, ...] of one section."""
    statuses = [e["status"] for e in elements]
    verdict = verdict_of(statuses)
    changes = []
    for i, e in enumerate(elements):
        if e["status"] == "satisfied":
            continue
        fixed = statuses[:i] + ["satisfied"] + statuses[i + 1:]
        changes.append({"predicate": e["predicate"], "description": e["description"], "status": e["status"],
                        "found": e["found"], "required": e["required"],
                        "question": BY_NAME[e["predicate"]].question if e["predicate"] in BY_NAME else "",
                        "verdict_if_satisfied": verdict_of(fixed)})
    missing = [c for c in changes if c["status"] == "missing"]
    blockers = [c for c in changes if c["status"] == "violated"]
    if verdict == "CONSISTENT":
        summary = (f"Every element is established. The verdict depends on all {len(elements)} of them: "
                   "if any one were contradicted it would become INCONSISTENT, and if any were left unstated it "
                   "would become INSUFFICIENT.")
    elif verdict == "INCONSISTENT":
        names = ", ".join(c["description"] for c in blockers)
        summary = (f"The facts contradict {len(blockers)} element(s): {names}. "
                   + ("Had the facts established this, the verdict would be "
                      f"{blockers[0]['verdict_if_satisfied']}." if len(blockers) == 1 else
                      "All of them would have to change for the section to apply."))
    else:
        summary = (f"{len(missing)} element(s) are not settled by the facts. To decide, the facts would need to "
                   "answer: " + " ".join(f"({i + 1}) {c['question']}" for i, c in enumerate(missing)))
    return {"verdict": verdict, "summary": summary, "missing": missing, "blockers": blockers,
            "sensitive_elements": [e["predicate"] for e in elements] if verdict == "CONSISTENT" else []}
