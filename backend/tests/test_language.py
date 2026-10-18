"""Hindi questions: detection, translation for retrieval, a Hindi answer with the same citations."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.chat_router import ChatRouter
from app.core.language import Translator, detect, devanagari_share
from app.core.models import Message
from app.core.query_rewriter import QueryRewriter
from tests.conftest import StubLLM
from tests.test_query_rewrite import QUESTION, REPLY, retriever  # noqa: F401  (fixture)

HINDI = "बैंक का अधिकारी बनकर एक कॉलर ने मेरा ओटीपी ले लिया और मेरा खाता खाली कर दिया।"
ANSWER = "यह प्रतिरूपण द्वारा छल है, धारा 419 [1]।"


def test_detect() -> None:
    assert detect(HINDI) == "hi"
    assert detect("What is section 302?") == "en"
    assert detect("धारा 302 क्या है? (murder)") == "hi"
    assert detect("12345") == "en"
    assert devanagari_share(ANSWER) > 0.9 and devanagari_share("Section 419 [1]") == 0


def test_translation_is_cached_and_a_non_english_reply_is_rejected() -> None:
    llm, tr = StubLLM(reply=QUESTION), Translator()
    assert tr.to_english(HINDI, llm) == QUESTION == tr.to_english(HINDI, llm)
    assert len(llm.calls) == 1
    assert Translator().to_english(HINDI, StubLLM(reply=HINDI)) is None
    assert Translator().to_english(HINDI, StubLLM(fail=True)) is None


class _Model(StubLLM):
    """Translates, rewrites (JSON) and answers in Hindi."""

    def generate(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        if "Translate the user's question" in messages[0].content:
            return QUESTION
        return REPLY if "Reply with JSON only" in messages[0].content else self.reply


def _events(retriever, llm, question):  # noqa: F811
    app = FastAPI()
    app.include_router(ChatRouter(retriever, llm, rewriter=QueryRewriter()).router)
    body = TestClient(app).post("/api/chat", json={"question": question}).text
    out = {}
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        if lines["event"] == "token":
            out["answer"] = out.get("answer", "") + json.loads(lines["data"])["text"]
        else:
            out[lines["event"]] = json.loads(lines["data"])
    return out


def test_a_hindi_question_is_searched_in_english_and_answered_in_hindi(retriever) -> None:  # noqa: F811
    llm = _Model(reply=ANSWER)
    ev = _events(retriever, llm, HINDI)
    assert [s["section"] for s in ev["sources"]] == ["419"]
    assert ev["retrieval"]["translation"] == {"from": "hi", "text": QUESTION}
    answer_prompt = llm.calls[-1]
    assert "Write the whole answer in Hindi" in answer_prompt[0].content
    assert HINDI in answer_prompt[-1].content and QUESTION in answer_prompt[-1].content
    assert ev["answer"] == ANSWER


def test_english_questions_are_not_translated(retriever) -> None:  # noqa: F811
    llm = _Model(reply="Cheating by personation, Section 419 [1].")
    ev = _events(retriever, llm, QUESTION)
    assert "translation" not in ev["retrieval"]
    assert not any("Translate the user's question" in m[0].content for m in llm.calls)
    assert "Hindi" not in llm.calls[-1][0].content


def test_an_unmatched_hindi_question_is_refused_in_hindi(retriever) -> None:  # noqa: F811
    class Fails(_Model):
        def generate(self, messages):
            raise RuntimeError("quota")

    ev = _events(retriever, Fails(), "मेरे मकान मालिक ने मेरी जमा राशि वापस नहीं की।")
    assert ev["meta"]["refused"] and "लीगल मोड" in ev["answer"]
