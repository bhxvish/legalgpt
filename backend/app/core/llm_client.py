"""LLMClient interface and its implementations.

GroqClient (Phase 1) calls a hosted model. LocalHFClient / AdapterClient (Phase 4) run a small
open model locally, optionally with a LoRA adapter. Callers depend on LLMClient only.
"""

import json
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.core.models import Message


class LLMClientError(RuntimeError):
    """Raised when the backing model cannot be reached or is misconfigured."""


class LLMRateLimitError(LLMClientError):
    """The hosted model refused the request because a usage limit was reached."""


class LLMClient(ABC):
    model_id: str

    @abstractmethod
    def generate(self, messages: list[Message]) -> str:
        """Return the full completion for `messages`."""

    @abstractmethod
    def stream(self, messages: list[Message]) -> Iterator[str]:
        """Yield the completion incrementally as text fragments."""


def _wait_text(message: str) -> str:
    """'Please try again in 9m28.08s' -> 'about 10 minutes' (empty if Groq gave no time)."""
    m = re.search(r"try again in ((?:\d+h)?(?:\d+m)?(?:[\d.]+s)?)", message)
    if not m or not m.group(1):
        return ""
    parts = {unit: float(num) for num, unit in re.findall(r"([\d.]+)([hms])", m.group(1))}
    seconds = parts.get("h", 0) * 3600 + parts.get("m", 0) * 60 + parts.get("s", 0)
    if seconds < 60:
        n = max(1, round(seconds))
        return f"about {n} second{'s' if n != 1 else ''}"
    minutes = -(-seconds // 60)  # round up
    return f"about {int(minutes)} minute{'s' if minutes != 1 else ''}"


def groq_error(exc: Exception) -> Exception:
    """Turn Groq SDK errors users can act on into a readable LLMClientError; others pass through."""
    import groq

    message = str(getattr(exc, "body", None) or exc)
    if isinstance(exc, groq.RateLimitError):
        daily = "per day" in message or "(TPD)" in message or "(RPD)" in message
        wait = _wait_text(message)
        what = ("The hosted model's daily usage limit (Groq free tier) has been reached"
                if daily else "The hosted model is receiving too many requests right now (Groq rate limit)")
        retry = f"Try again in {wait}" if wait else "Try again later"
        return LLMRateLimitError(f"{what}. {retry}.")
    if isinstance(exc, groq.AuthenticationError):
        return LLMClientError("Groq rejected GROQ_API_KEY; check the key in .env and restart the backend.")
    if isinstance(exc, groq.APIConnectionError):
        return LLMClientError("Could not reach Groq (network error); check the internet connection.")
    return exc


class GroqClient(LLMClient):
    """Chat model served by Groq (default openai/gpt-oss-120b; reasoning tokens arrive separately and are not streamed)."""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.1,
        max_tokens: int = 2048,  # reasoning models spend part of this budget on hidden reasoning
        client: Any | None = None,
        max_retries: int = 5,  # 429s on the free tier (8k tokens/min) clear in seconds; the SDK honours retry-after
    ) -> None:
        self.api_key = api_key
        self.model_id = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise LLMClientError("GROQ_API_KEY is not set; add it to .env (see .env.example).")
            from groq import Groq

            self._client = Groq(api_key=self.api_key, max_retries=self.max_retries)
        return self._client

    def _payload(self, messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _create(self, messages: list[Message], **kwargs: Any) -> Any:
        try:
            return self.client.chat.completions.create(
                model=self.model_id,
                messages=self._payload(messages),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs,
            )
        except Exception as exc:
            raise groq_error(exc) from exc

    def generate(self, messages: list[Message]) -> str:
        return self._create(messages).choices[0].message.content or ""

    def stream(self, messages: list[Message]) -> Iterator[str]:
        for event in self._create(messages, stream=True):
            if event.choices and (delta := event.choices[0].delta.content):
                yield delta


class LocalHFClient(LLMClient):
    """A Hugging Face causal LM run in-process: 4-bit on a CUDA GPU when available, otherwise
    full precision on CPU (slow, but works without bitsandbytes). Loads lazily on first use.

    With `adapter_path`, a PEFT LoRA adapter is applied on top of the frozen base model."""

    def __init__(
        self,
        base_model: str,
        adapter_path: str | Path | None = None,
        max_new_tokens: int = 512,
        device: str | None = None,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.max_new_tokens = max_new_tokens
        self._device = device
        self.model_id = base_model + (f"+lora:{self.adapter_path.name}" if self.adapter_path else "")
        self._model: Any = None
        self._tokenizer: Any = None
        self._lock = threading.Lock()

    def _load(self) -> tuple[Any, Any]:
        with self._lock:
            if self._model is None:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

                device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
                kwargs: dict[str, Any] = {}
                if device == "cuda":
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    kwargs["device_map"] = {"": 0}
                else:
                    kwargs["dtype"] = torch.float32
                self._tokenizer = AutoTokenizer.from_pretrained(self.adapter_path or self.base_model)
                model = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs)
                if self.adapter_path:
                    from peft import PeftModel

                    model = PeftModel.from_pretrained(model, str(self.adapter_path))
                model.eval()
                self._model = model
        return self._model, self._tokenizer

    def _inputs(self, messages: list[Message]) -> Any:
        model, tok = self._load()
        ids = tok.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
        return {k: v.to(model.device) for k, v in ids.items()}

    def _gen_kwargs(self) -> dict[str, Any]:
        tok = self._load()[1]
        # greedy decoding: reproducible comparisons, no sampling noise in citations
        return {"max_new_tokens": self.max_new_tokens, "do_sample": False, "pad_token_id": tok.pad_token_id or tok.eos_token_id}

    def generate(self, messages: list[Message]) -> str:
        import torch

        model, tok = self._load()
        inputs = self._inputs(messages)
        with torch.no_grad():
            out = model.generate(**inputs, **self._gen_kwargs())
        return tok.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    def stream(self, messages: list[Message]) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        model, tok = self._load()
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        inputs = self._inputs(messages)
        thread = threading.Thread(target=model.generate, kwargs={**inputs, **self._gen_kwargs(), "streamer": streamer}, daemon=True)
        thread.start()
        yield from streamer
        thread.join()

    def close(self) -> None:
        """Release the model (and GPU memory) so another local model can load."""
        with self._lock:
            self._model = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


class AdapterClient(LocalHFClient):
    """The LoRA-tuned model from Phase 4: base model + adapter, base model read from the
    adapter's metadata (adapter_meta.json written by LoRATrainer)."""

    def __init__(self, adapter_path: str | Path, max_new_tokens: int = 512, device: str | None = None) -> None:
        adapter_path = Path(adapter_path)
        meta_file = adapter_path / "adapter_meta.json"
        if not meta_file.exists():
            raise LLMClientError(f"{adapter_path} is not a LoRA adapter checkpoint (no adapter_meta.json)")
        base = json.loads(meta_file.read_text(encoding="utf-8"))["base_model"]
        super().__init__(base, adapter_path, max_new_tokens=max_new_tokens, device=device)


def latest_adapter(adapters_dir: str | Path) -> Path | None:
    runs = [p for p in Path(adapters_dir).glob("*") if (p / "adapter_meta.json").exists()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None
