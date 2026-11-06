"""Search on the facts, not on who the people are.

Which IPC section applies does not depend on a person's caste, wealth, home state or adult age, yet a
sentence like "The accused is from a Dalit family" changes the search: its words move the embedding and
the query rewrite. So such a sentence, or a word like "wealthy" before a person, is left out of the text
that is searched. The model still reads the whole question, and the app shows what was left out.

What can carry legal weight stays: anyone under 18, public servants, employees, relatives, and any
sentence that also says what someone did. Text with nothing to remove comes back unchanged.
"""

import re

_STATES = (r"andhra pradesh|arunachal pradesh|assam|bihar|chhattisgarh|goa|gujarat|haryana|himachal pradesh|jharkhand|"
           r"karnataka|kerala|madhya pradesh|maharashtra|manipur|meghalaya|mizoram|nagaland|odisha|punjab|rajasthan|"
           r"sikkim|tamil nadu|telangana|tripura|uttar pradesh|uttarakhand|west bengal|kashmir|ladakh|delhi")
_CASTE = r"caste|dalit|brahmins?|brahman|kshatriya|vaishya|shudra|rajput|adivasi|tribal|obc"
_WEALTH = (r"wealthy|rich|affluent|well-off|poor|needy|low-income|middle-class|millionaire|billionaire|"
           r"daily-wage|daily wage|homeless|slum|labourers?|businessman|businesswoman|industrialist")
_ORIGIN = rf"migrant|outsider|local resident|native of|north indian|south indian|{_STATES}"
_ADULT = r"retired|pensioner|senior citizen|elderly|in (?:his|her|their) (?:twenties|thirties|forties|fifties|sixties|seventies|eighties)"

_IDENTITY = re.compile(rf"(?i)\b(?:{_CASTE}|{_WEALTH}|{_ORIGIN}|{_ADULT})\b")
_AGE = re.compile(r"(?i)\b(\d{1,3})[- ]years?[- ]old\b|\baged (\d{1,3})\b")
_SUBJECT = (r"(?:(?:the|this|that) )?(?:accused|victim|complainant|offender|culprit|thief|person|man|woman|"
            r"he|she|they|i|we)(?: responsible| concerned| involved)?")
_STATEMENT = re.compile(rf"(?i)^{_SUBJECT}\s+(?:is|are|am|was|were|belongs?|comes?|hails?)\b")
_MORE = re.compile(r"(?i)\b(?:and|but|because|who|which|when|while|after|before|so|then|though)\b")
_KEEP = re.compile(r"(?i)\b(?:public servant|government|police|officer|clerk|employee|servant|agent|banker|merchant|"
                   r"carrier|minor|child|juvenile|unsound|pregnant|husband|wife|relative|guardian)\b")
_MAX_WORDS = 14

_PERSON = (r"man|woman|men|women|person|people|boy|girl|youth|labourer|worker|farmer|shopkeeper|businessman|"
           r"driver|neighbour|tenant|villager|family|widow|student")
_DESCRIBED = re.compile(rf"(?i)\b(an?\s+)?(?:wealthy|rich|affluent|well-off|poor|needy|dalit|brahmin|upper-caste|"
                        rf"lower-caste|tribal|migrant)\s+(?=(?:{_PERSON})\b)")
_FROM = re.compile(rf"(?i)\b({_PERSON}) from (?:{_STATES})\b")


def _is_background(sentence: str) -> bool:
    """A short statement of who someone is, and nothing else."""
    if sentence.rstrip().endswith("?") or len(sentence.split()) > _MAX_WORDS:
        return False
    if not _STATEMENT.match(sentence.strip()) or _MORE.search(sentence) or _KEEP.search(sentence):
        return False
    ages = [int(a or b) for a, b in _AGE.findall(sentence)]
    if any(age < 18 for age in ages):
        return False
    return bool(ages) or bool(_IDENTITY.search(sentence))


def _article(match: re.Match) -> str:
    """The article for the word that now follows: "an upper-caste man" -> "a man"."""
    if not match.group(1):
        return ""
    following = match.string[match.end():match.end() + 1].lower()
    word = "an " if following in "aeiou" else "a "
    return word.capitalize() if match.group(1)[0].isupper() else word


def strip_background(text: str) -> tuple[str, list[str]]:
    """(the text to search, what was left out). The text is returned untouched if there is nothing to
    leave out, or if nothing would remain."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [s for s in sentences if not _is_background(s)]
    removed = [s for s in sentences if _is_background(s)]
    if not kept:
        return text, []
    clean = " ".join(kept) if removed else text
    words = [m.group(0).strip() for m in _DESCRIBED.finditer(clean)] + [m.group(0) for m in _FROM.finditer(clean)]
    clean = _FROM.sub(lambda m: m.group(1), _DESCRIBED.sub(_article, clean))
    return clean, removed + words
