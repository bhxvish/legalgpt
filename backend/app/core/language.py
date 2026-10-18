"""Hindi questions: translated to English for retrieval, answered in Hindi with the same citations.

The retrieval model (all-MiniLM-L6-v2) and the IPC text are English, so a question in Devanagari
finds nothing on its own. The chat's model (or, for a local model, the hosted model that also
does the query rewrite) translates the question first; that English text is what is searched,
rewritten and proved, and it is shown in the Why panel. The answer prompt keeps the original
question and asks for a Hindi answer that keeps section numbers and [n] citations as they are.
"""

import logging
import re
from collections import OrderedDict

from app.core.models import Message

logger = logging.getLogger(__name__)

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LETTER = re.compile(r"[A-Za-zऀ-ॿ]")

TRANSLATE_PROMPT = """Translate the user's question about Indian criminal law into plain English.
Keep names, places, amounts, dates and any section numbers exactly as given. Keep legal terms in
their usual English form (e.g. "चोरी" -> "theft", "धोखाधड़ी" -> "cheating"). Do not answer the question
and do not add anything. Reply with the English translation only."""

HINDI_RULE = """8. The question is in Hindi. Write the whole answer in Hindi (Devanagari), in plain words a \
non-lawyer understands. Keep section numbers, the [n] citations, case names and amounts exactly as in \
the SOURCES; you may give a legal term's English name in brackets once. If the sources do not cover the \
question, reply: "मुझे नहीं पता — प्राप्त स्रोत इस प्रश्न को कवर नहीं करते।\""""

SCOPE_REFUSAL_HI = (
    "मैं लीगल मोड में इसका उत्तर नहीं दे सकता: अनुक्रमित कानूनी स्रोतों (भारतीय दंड संहिता, 1860) में से कोई भी "
    "इस प्रश्न से इतना मेल नहीं खाता कि स्रोतों पर आधारित उत्तर दिया जा सके। जिस अपराध या धारा के बारे में आप "
    "पूछ रहे हैं, उसका नाम लिखकर देखें, या ऐसे उत्तर के लिए जनरल मोड चुनें जो स्रोतों पर आधारित नहीं है।"
)


TRANSLATION_FAILED = (
    "आपके प्रश्न का अंग्रेज़ी में अनुवाद नहीं हो सका (मॉडल की उपयोग सीमा या कोई त्रुटि), इसलिए IPC में खोज नहीं हो पाई। "
    "कृपया थोड़ी देर बाद फिर कोशिश करें, या प्रश्न अंग्रेज़ी में पूछें।\n\n"
    "The question could not be translated into English (a model's usage limit or an error), so the IPC could not be "
    "searched. Please try again shortly, or ask in English."
)


def detect(text: str) -> str:
    """"hi" when most letters are Devanagari, else "en"."""
    letters = LETTER.findall(text)
    if not letters:
        return "en"
    return "hi" if len(DEVANAGARI.findall(text)) / len(letters) >= 0.5 else "en"


def devanagari_share(text: str) -> float:
    """Share of letters in Devanagari (the eval's 'answered in Hindi' check)."""
    letters = LETTER.findall(re.sub(r"\[\d+\]", "", text))
    return len(DEVANAGARI.findall(text)) / len(letters) if letters else 0.0


class Translator:
    def __init__(self, cache_size: int = 512) -> None:
        self._cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        self.cache_size = cache_size

    def to_english(self, question: str, llm, strict: bool = False) -> str | None:
        """The English translation, or None if the call failed (the question is then searched
        as written, which usually ends in the scope refusal). `strict`: raise instead (benchmarks)."""
        key = (llm.model_id, question.strip())
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        try:
            ask = getattr(llm, "generate_brief", llm.generate)
            reply = ask([Message("system", TRANSLATE_PROMPT), Message("user", question)])
        except Exception as exc:
            if strict:
                raise
            logger.warning("translation failed (%s): %s", llm.model_id, exc)
            return None
        text = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.DOTALL).strip().strip('"').strip()
        if not text or detect(text) != "en":
            return None
        self._cache[key] = text[:1000]
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return self._cache[key]
