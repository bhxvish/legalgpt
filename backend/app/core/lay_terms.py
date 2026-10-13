"""Lay-to-statutory query expansion for everyday fact patterns.

People describe crimes in everyday words ("someone slapped me", "he never delivered the laptop"),
while the IPC uses its own vocabulary ("voluntarily causes hurt", "cheats"). A general-purpose
embedding model does not reliably bridge the two, so a question that matches one of these patterns
gets the statute's wording appended to its search query (the question shown to the model is not
changed). Each rule names an offence category, not a particular question.

Triggers are everyday words only, never the IPC's own terms ("murder", "robbery", "cheating",
"forgery", "trespass", "assault"...): a question that already uses the statute's vocabulary
retrieves well, and adding more wording could only pull it towards a neighbouring section (an early
version sent "punishment for murder" to the definition in s.300 instead of s.302).
"""

import re

# (pattern over the question, statutory wording appended to the search query)
LAY_TERMS: tuple[tuple[str, str], ...] = (
    # property offences
    (r"\b(stole|stolen|steal|stealing|thie(f|ves)|pickpocket\w*|took [\w' ]{0,40}without (his |her |my |their )?(permission|consent|asking|knowledge))",
     "theft: dishonestly taking movable property out of a person's possession without consent"),
    (r"\b(snatch\w*|robbed|mugg\w*|at (knife|gun)\s?point|looted)\b",
     "robbery: theft by causing hurt or fear of instant hurt; dacoity"),
    (r"\b(fraud\w*|scam\w*|dup(ed|ing)|tricked|conned|fake|never delivered|false promise\w*|posing as)\b",
     "cheating: deceiving a person and dishonestly inducing delivery of property"),
    (r"\b(fake (document|signature|certificate|stamp|id)\w*|copied (his|her|my) signature)",
     "forgery: making a false document or false electronic record"),
    (r"\b(kept the money|embezzl\w*|spent [\w' ]{0,30}money|siphon\w*|deposits into his own)",
     "criminal breach of trust: dishonest misappropriation of property entrusted"),
    (r"\b(ransom|protection money|demand\w* (money|cash|payment)[\w' ,]{0,40}(threat\w*|unless|or else))",
     "extortion: putting a person in fear of injury to make him deliver property"),
    (r"\b(broke into|break into|breaking into|burgl\w*|broke the lock|break[- ]in)\b",
     "house-breaking; lurking house-trespass by night; theft in dwelling house"),
    (r"\b(entered [\w' ]{0,40}without (permission|consent)|refused to leave)\b",
     "criminal trespass; house-trespass: entering property in possession of another to annoy or intimidate"),
    (r"\b(set fire|burnt down|burned down|damag\w*|destroy\w*|vandal\w*|scratched)\b",
     "mischief: causing destruction or damage to property; mischief by fire"),
    # offences against the body
    (r"\b(slap\w*|punch\w*|beat(en|ing|s)?|hit|kick\w*|push\w*|attack\w*|thrash\w*)\b",
     "voluntarily causing hurt; assault; using criminal force"),
    (r"\b(broke (his|her|my|their) (arm|leg|bone|jaw|nose)\w*|fractur\w*|acid|disfigur\w*|lost (his|her) (eye|sight|hearing))",
     "grievous hurt: fracture, permanent disfiguration, voluntarily causing grievous hurt by dangerous weapons"),
    (r"\b(stab\w*|shot|fired (a|his|her) (gun|pistol)|tried to kill|attempt\w* to kill|intending to kill)\b",
     "attempt to murder: act done with intention or knowledge that it would cause death"),
    (r"\b(careless\w*|mistake|wrong (injection|medicine|dose)|reckless\w*|by accident|accidentally)\b",
     "rash or negligent act; causing death by negligence; act endangering life or personal safety of others"),
    (r"\b(driv\w*|overtak\w*|speeding|lorry|truck|bus|car|motorcycle|bike)\b[\w' ,.]{0,80}\b(hit|ran over|crash\w*|knocked|died|killed|injur\w*)\b",
     "rash driving or riding on a public way; causing death by rash or negligent driving"),
    (r"\b(locked|confined|tied (him|her|them) up|tied up|held (him|her|them)|would not let (him|her|them) (leave|go)|captive)\b",
     "wrongful confinement: restraining a person so as to prevent him from proceeding beyond limits"),
    (r"\b(took (a|the|her|his) (child|baby|toddler|boy|girl)|ran away with (him|her|the child))\b",
     "kidnapping from lawful guardianship: taking a minor without consent of the guardian"),
    (r"\b(threat\w*|said (he|she|they) would (kill|beat|hurt|harm)|would break (my|his|her) legs)\b",
     "criminal intimidation: threatening injury to person, reputation or property"),
    # offences against women and the family
    (r"\b(in-?laws?|husband'?s? (family|relatives|mother))\b",
     "cruelty by husband or relatives of husband; dowry demand; harassment of wife"),
    (r"\b(follow(ed|s|ing) (her|a woman|a girl)|keeps? messaging|tracked her|watch(ed|ing) (her|a woman))\b",
     "stalking: following a woman and contacting her despite clear disinterest"),
    (r"\b(molest\w*|groped|touched (her|a woman) (inappropriately|indecently))\b",
     "assault or criminal force to woman with intent to outrage her modesty"),
    # reputation
    (r"\b(false (claims?|reports?|allegations?|statements?)|ruin\w* (his|her|my|their) reputation|called (him|her) a thief)\b",
     "defamation: imputation intending to harm the reputation of a person"),
)

_COMPILED = tuple((re.compile(p, re.IGNORECASE), legal) for p, legal in LAY_TERMS)


def statutory_terms(question: str) -> list[str]:
    """The statutory wording for every lay pattern the question matches, in rule order."""
    return [legal for pattern, legal in _COMPILED if pattern.search(question)]
