import pytest

from app.core.fallback import DAILY_COOLDOWN_S, FallbackClient
from app.core.llm_client import LLMClient, LLMClientError, LLMRateLimitError


class Stub(LLMClient):
    def __init__(self, model_id, error=None, reply="ok"):
        self.model_id, self.error, self.reply, self.calls = model_id, error, reply, 0

    def generate(self, messages):
        self.calls += 1
        if self.error:
            raise self.error
        return f"{self.reply} from {self.model_id}"

    def stream(self, messages):
        yield self.generate(messages)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_first_model_answers_when_available():
    groq, gemini = Stub("groq"), Stub("gemini")
    fb = FallbackClient([groq, gemini])
    assert fb.generate([]) == "ok from groq"
    assert fb.model_id == "groq" and fb.note == "" and gemini.calls == 0


def test_moves_on_when_out_of_quota_and_reports_it():
    groq = Stub("groq", LLMRateLimitError("The hosted model's daily usage limit (Groq free tier) has been reached."))
    gemini = Stub("gemini")
    fb = FallbackClient([groq, gemini])
    assert fb.generate([]) == "ok from gemini"
    assert fb.model_id == "gemini"
    assert "groq" in fb.note and "used gemini" in fb.note


def test_a_daily_limit_is_skipped_for_an_hour_then_retried():
    clock = Clock()
    groq = Stub("groq", LLMRateLimitError("daily usage limit reached"))
    gemini = Stub("gemini")
    fb = FallbackClient([groq, gemini], clock=clock)
    fb.generate([])
    fb.generate([])
    assert groq.calls == 1  # not asked again while it is out of quota
    clock.t = DAILY_COOLDOWN_S + 1
    fb.generate([])
    assert groq.calls == 2


def test_other_errors_also_fall_through_to_the_local_model():
    groq = Stub("groq", LLMRateLimitError("daily"))
    gemini = Stub("gemini", LLMClientError("gemini (x) is overloaded (HTTP 503)"))
    local = Stub("qwen+gguf-adapter:v0.7")
    fb = FallbackClient([groq, gemini])
    fb.add(local)
    fb.add(local)  # added once
    assert fb.generate([]) == "ok from qwen+gguf-adapter:v0.7"
    assert len(fb.clients) == 3


def test_all_failing_raises_one_error_naming_each_model():
    fb = FallbackClient([Stub("groq", LLMRateLimitError("daily")), Stub("gemini", LLMRateLimitError("daily"))])
    with pytest.raises(LLMClientError) as e:
        fb.generate([])
    assert "groq" in str(e.value) and "gemini" in str(e.value)


def test_verification_models_use_free_providers_only(monkeypatch):
    from app import main
    from app.config import Settings

    monkeypatch.setattr("app.core.openai_compat.provider_key", lambda p, env_file=None: "key")
    s = Settings(_env_file=None, GROQ_API_KEY="k",
                 EXTRA_MODELS="gemini:gemini-x-flash-lite,openai:gpt-5.4-mini,gemini:gemini-x-flash")
    ids = [m.model_id for m in main.verification_models(s)]
    assert ids[1:] == ["gemini:gemini-x-flash", "gemini:gemini-x-flash-lite"]  # no paid OpenAI; full before lite
