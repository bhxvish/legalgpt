"""LoRATrainer: 4-bit QLoRA fine-tuning of a small instruction model on InstructionExamples.

Runs in the GPU environment (.venv-gpu, backend/requirements-gpu.txt). Only the LoRA adapter is
saved (tens of MB); the frozen base model is never modified or copied.
"""

import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.finetuning.instruction_builder import InstructionBuilder, InstructionExample

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
META_FILE = "adapter_meta.json"


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 2e-4
    epochs: int = 3
    load_in_4bit: bool = True
    batch_size: int = 1
    grad_accum: int = 8
    max_length: int = 1536  # tokens; the longest example is ~1,300
    seed: int = 13
    max_steps: int = -1  # >0 caps optimizer steps (smoke tests)


@dataclass
class TrainingReport:
    train_loss: float
    eval_loss: float | None
    eval_loss_before: float | None  # the untuned base model on the same eval set
    adapter_path: str
    seconds: float
    n_train: int
    n_eval: int
    config: dict[str, Any]
    corpus: dict[str, Any] = field(default_factory=dict)
    peak_gpu_mem_gb: float | None = None
    log_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LoRATrainer:
    def __init__(self, cfg: TrainConfig, builder: InstructionBuilder | None = None) -> None:
        self.cfg = cfg
        self.builder = builder or InstructionBuilder()
        self.model: Any = None
        self.tokenizer: Any = None

    def load_base_model(self) -> Any:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs: dict[str, Any] = {"dtype": torch.bfloat16}
        if self.cfg.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            kwargs["device_map"] = {"": 0}
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.base_model, **kwargs)
        return self.model

    def attach_adapter(self) -> Any:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if self.model is None:
            self.load_base_model()
        if self.cfg.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)
        lora = LoraConfig(
            r=self.cfg.r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=LORA_TARGETS,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora)
        return self.model

    def _dataset(self, examples: list[InstructionExample]) -> Any:
        """Conversational prompt/completion rows: the loss covers only the assistant's answer."""
        from datasets import Dataset

        rows = [
            {"prompt": InstructionBuilder.messages(ex, with_answer=False),
             "completion": [{"role": "assistant", "content": ex.output}]}
            for ex in examples
        ]
        return Dataset.from_list(rows)

    def train(
        self,
        train_ds: list[InstructionExample],
        eval_ds: list[InstructionExample],
        adapter_path: str | Path,
        corpus: dict[str, Any] | None = None,
    ) -> TrainingReport:
        import torch
        from trl import SFTConfig, SFTTrainer

        if self.model is None or not hasattr(self.model, "peft_config"):
            self.attach_adapter()
        adapter_path = Path(adapter_path)
        steps_per_epoch = math.ceil(len(train_ds) / (self.cfg.batch_size * self.cfg.grad_accum))
        total_steps = self.cfg.max_steps if self.cfg.max_steps > 0 else steps_per_epoch * self.cfg.epochs
        args = SFTConfig(
            output_dir=str(adapter_path / "_trainer"),
            num_train_epochs=self.cfg.epochs,
            max_steps=self.cfg.max_steps,
            per_device_train_batch_size=self.cfg.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=self.cfg.grad_accum,
            learning_rate=self.cfg.lr,
            lr_scheduler_type="cosine",
            warmup_steps=max(1, round(0.05 * total_steps)),
            max_length=self.cfg.max_length,
            completion_only_loss=True,  # loss on the answer only, never on the prompt
            gradient_checkpointing=True,
            bf16=torch.cuda.is_available(),
            optim="paged_adamw_8bit" if self.cfg.load_in_4bit else "adamw_torch",
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none",
            seed=self.cfg.seed,
        )
        trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=self._dataset(train_ds),
            eval_dataset=self._dataset(eval_ds),
            processing_class=self.tokenizer,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        eval_before = trainer.evaluate()["eval_loss"] if eval_ds else None
        start = time.perf_counter()
        result = trainer.train()
        seconds = time.perf_counter() - start
        eval_after = trainer.evaluate()["eval_loss"] if eval_ds else None

        adapter_path.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(adapter_path / "_trainer", ignore_errors=True)  # scratch dir; checkpoints are not kept (save_strategy="no")
        trainer.model.save_pretrained(adapter_path)  # adapter weights only
        self.tokenizer.save_pretrained(adapter_path)
        report = TrainingReport(
            train_loss=float(result.training_loss),
            eval_loss=eval_after,
            eval_loss_before=eval_before,
            adapter_path=str(adapter_path),
            seconds=round(seconds, 1),
            n_train=len(train_ds),
            n_eval=len(eval_ds),
            config=asdict(self.cfg),
            corpus=corpus or {},
            peak_gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
            log_history=trainer.state.log_history,
        )
        (adapter_path / META_FILE).write_text(json.dumps(
            {"base_model": self.cfg.base_model, "load_in_4bit": self.cfg.load_in_4bit, "report": report.to_dict()},
            indent=2), encoding="utf-8")
        return report
