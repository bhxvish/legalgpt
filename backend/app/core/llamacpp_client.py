"""LlamaCppClient: a larger local model (the 7B "heavy" option) through llama.cpp.

A 7B model needs ~5-6 GB in 4-bit, more than a 4 GB laptop GPU. With PyTorch it would spill to
the CPU and take minutes per answer. llama.cpp runs a GGUF-quantised base model (Q4_K_M,
~4.7 GB) with the LoRA adapter converted to GGUF (backend/scripts/colab/README.md), putting as
many layers on the GPU as fit (HEAVY_GPU_LAYERS) and the rest on the CPU: tens of seconds per
answer instead of minutes. Everything stays on this machine.

Needs `llama-cpp-python` (a CUDA build for GPU offload; the CPU build works, slower).
"""

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.core.llm_client import LLMClient, LLMClientError
from app.core.models import Message


class LlamaCppClient(LLMClient):
    is_local = True  # chat_router: hands the query rewrite to a hosted model unless OFFLINE_MODE

    def __init__(self, model_path: str | Path, lora_path: str | Path | None = None, n_gpu_layers: int = 16,
                 n_ctx: int = 4096, max_tokens: int = 512, temperature: float = 0.1, llama: Any = None) -> None:
        self.model_path, self.lora_path = Path(model_path), Path(lora_path) if lora_path else None
        self.n_gpu_layers, self.n_ctx = n_gpu_layers, n_ctx
        self.max_tokens, self.temperature = max_tokens, temperature
        lora = f"+gguf-adapter:{self.lora_path.stem}" if self.lora_path else ""
        self.model_id = f"{self.model_path.stem}{lora}"
        self._llama = llama  # tests pass a stub
        self._lock = threading.Lock()  # one generation at a time (llama.cpp contexts are not thread-safe)

    def _model(self) -> Any:
        if self._llama is None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise LLMClientError("the heavy local model needs llama-cpp-python (pip install llama-cpp-python)") from exc
            if not self.model_path.exists():
                raise LLMClientError(f"HEAVY_MODEL_GGUF not found: {self.model_path}")
            self._llama = Llama(model_path=str(self.model_path), lora_path=str(self.lora_path) if self.lora_path else None,
                                n_gpu_layers=self.n_gpu_layers, n_ctx=self.n_ctx, verbose=False)
        return self._llama

    def _messages(self, messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def generate(self, messages: list[Message]) -> str:
        with self._lock:
            out = self._model().create_chat_completion(messages=self._messages(messages), max_tokens=self.max_tokens,
                                                       temperature=self.temperature)
        return out["choices"][0]["message"].get("content") or ""

    def stream(self, messages: list[Message]) -> Iterator[str]:
        with self._lock:
            for chunk in self._model().create_chat_completion(messages=self._messages(messages), max_tokens=self.max_tokens,
                                                              temperature=self.temperature, stream=True):
                text = chunk["choices"][0].get("delta", {}).get("content")
                if text:
                    yield text
