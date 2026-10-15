"""Local models: the llama.cpp heavy model, OFFLINE_MODE rewrites, dtype choice, ZK adapter fingerprint."""

import json
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.chat_router import ChatRouter
from app.core.llamacpp_client import LlamaCppClient
from app.core.models import Message
from app.core.query_rewriter import QueryRewriter
from tests.conftest import StubLLM


class _Llama:
    """Stands in for llama_cpp.Llama."""

    def __init__(self) -> None:
        self.calls = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def create_chat_completion(self, messages, max_tokens, temperature, stream=False):
        self.calls.append(messages)
        if stream:
            return iter([{"choices": [{"delta": {"content": "Section "}}]}, {"choices": [{"delta": {}}]},
                         {"choices": [{"delta": {"content": "379 [1]."}}]}])
        return {"choices": [{"message": {"content": "Section 379 [1]."}}]}


def test_llamacpp_client_generates_and_streams(tmp_path) -> None:
    llama = _Llama()
    c = LlamaCppClient(tmp_path / "qwen2.5-7b-instruct-q4_k_m.gguf", tmp_path / "v0.7-7b.gguf", llama=llama)
    msgs = [Message("system", "s"), Message("user", "q")]
    assert c.generate(msgs) == "Section 379 [1]."
    assert "".join(c.stream(msgs)) == "Section 379 [1]."
    assert llama.calls[0] == [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    assert c.model_id == "qwen2.5-7b-instruct-q4_k_m+gguf-adapter:v0.7-7b" and "+lora" not in c.model_id
    assert llama.resets == 2  # a fresh context for every answer (reproducible output)
    assert c.is_local


def test_split_gguf_model_id_names_the_model(tmp_path) -> None:
    c = LlamaCppClient(tmp_path / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", tmp_path / "v0.7-7b-l4.gguf", llama=_Llama())
    assert c.model_id == "qwen2.5-7b-instruct-q4_k_m+gguf-adapter:v0.7-7b-l4"


def test_missing_model_file_is_a_clear_error(tmp_path) -> None:
    import pytest

    from app.core.llm_client import LLMClientError

    with pytest.raises(LLMClientError):
        LlamaCppClient(tmp_path / "missing.gguf").generate([Message("user", "q")])


class _Local(StubLLM):
    is_local = True
    model_id = "local"


def _router(offline: bool) -> ChatRouter:
    hosted, local = StubLLM(), _Local()
    return ChatRouter(None, hosted, models={"groq": hosted, "adapter": local}, default_model="groq",  # type: ignore[arg-type]
                      rewriter=QueryRewriter(), offline=offline)


def test_local_models_rewrite_themselves_only_in_offline_mode() -> None:
    online, offline = _router(False), _router(True)
    assert online._rewrite_llm(online.models["adapter"]) is online.models["groq"]
    assert offline._rewrite_llm(offline.models["adapter"]) is offline.models["adapter"]
    assert online._rewrite_llm(online.models["groq"]) is online.models["groq"]


def test_zk_record_fingerprints_only_the_published_adapter() -> None:
    from app.zk.service import ZkService

    class _Att:
        context = None

        def answer_record(self, **kw):
            return dict(self.context)

    class _Svc(ZkService):  # the published manifest, without files on disk
        manifest = {"corpus": {"merkle_root": "r"}, "adapter": {"name": "v0.4-plain-2ep", "sha256": "abc"}}

    svc = _Svc.__new__(_Svc)
    svc.attestor, svc.leaf_of = _Att(), lambda cid: 0
    kw = dict(question="q", prompt=[], answer="a", sources=[], refused=False, follow_up=False)
    assert svc.attest_answer(model_id="Qwen/Qwen2.5-1.5B-Instruct+lora:v0.4-plain-2ep", **kw)["adapter_sha256"] == "abc"
    assert "adapter_sha256" not in svc.attest_answer(model_id="Qwen/Qwen2.5-3B-Instruct+lora:v0.7-3b", **kw)
    assert "adapter_sha256" not in svc.attest_answer(model_id="qwen2.5-7b+gguf-adapter:v0.7-7b", **kw)


def test_local_gguf_setting_runs_the_local_model_through_llamacpp(tmp_path, monkeypatch) -> None:
    from app.config import Settings
    from app.main import build_models

    monkeypatch.setattr("app.config.REPO_ROOT", tmp_path)
    s = Settings(_env_file=None, LOCAL_MODEL_GGUF=str(tmp_path / "qwen2.5-3b-instruct-q4_k_m.gguf"),
                 LOCAL_LORA_GGUF=str(tmp_path / "v0.7-3b-l4.gguf"), LORA_DIR=str(tmp_path))
    models, notes, _ = build_models(s)
    assert isinstance(models["adapter"], LlamaCppClient)  # nothing is loaded until the first question
    assert models["adapter"].model_id == "qwen2.5-3b-instruct-q4_k_m+gguf-adapter:v0.7-3b-l4"
    assert "llama.cpp" in notes["adapter"]


def test_zk_record_fingerprints_a_published_gguf_adapter() -> None:
    from app.zk.service import ZkService

    class _Att:
        context = None

        def answer_record(self, **kw):
            return dict(self.context)

    class _Svc(ZkService):
        manifest = {"corpus": {"merkle_root": "r"}, "adapter": {"name": "v0.7-3b-l4", "sha256": "abc"},
                    "adapter_gguf": {"v0.7-3b-l4": "def", "v0.7-7b-l4": "ghi"}}

    svc = _Svc.__new__(_Svc)
    svc.attestor, svc.leaf_of = _Att(), lambda cid: 0
    kw = dict(question="q", prompt=[], answer="a", sources=[], refused=False, follow_up=False)
    assert svc.attest_answer(model_id="qwen2.5-3b-instruct-q4_k_m+gguf-adapter:v0.7-3b-l4", **kw)["adapter_sha256"] == "def"
    assert svc.attest_answer(model_id="qwen2.5-7b-instruct-q4_k_m+gguf-adapter:v0.7-7b-l4", **kw)["adapter_sha256"] == "ghi"
    assert "adapter_sha256" not in svc.attest_answer(model_id="x+gguf-adapter:unknown", **kw)


def test_loading_one_local_model_frees_the_other(tmp_path, monkeypatch) -> None:
    """A 4 GB GPU holds the 3B or the 7B, not both: loading one closes the other."""
    import app.core.llamacpp_client as lc

    for f in ("a.gguf", "b.gguf"):
        (tmp_path / f).write_bytes(b"x")
    made = []

    class _FakeLlama(_Llama):
        def __init__(self, **kw):
            super().__init__()
            self.closed = False
            made.append(self)

        def close(self):
            self.closed = True

    import types
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=_FakeLlama))
    small, heavy = lc.LlamaCppClient(tmp_path / "a.gguf"), lc.LlamaCppClient(tmp_path / "b.gguf")
    small.generate([Message("user", "q")])
    heavy.generate([Message("user", "q")])
    assert made[0].closed and small._llama is None and heavy._llama is made[1]
    small.generate([Message("user", "q")])  # loads again, freeing the heavy one
    assert made[1].closed
