"""RRLClassifier: InLegalBERT fine-tuned to predict a sentence's rhetorical role."""

import copy
import json
import math
import random
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.labeling_assistant.dataset import LengthBucketSampler, RRLDataset, RRLExample, collate
from app.labeling_assistant.evaluator import ClassifierEvaluator, Metrics
from app.labeling_assistant.models import Prediction

DEFAULT_MODEL = "law-ai/InLegalBERT"
META_FILE = "rrl_meta.json"


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 16
    lr: float = 3e-5
    head_lr: float = 1e-3  # the classification head starts from random weights, so it learns faster
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    freeze_layers: int = 8  # bottom encoder layers (and embeddings) kept frozen; 0 = full fine-tuning
    class_weighted: bool = True  # inverse-frequency loss weights for the skewed label distribution
    grad_clip: float = 1.0
    seed: int = 13
    max_steps: int | None = None  # cap for smoke tests
    bucket_pool_batches: int = 50  # sortish batching pool; 0 = plain random batches
    bf16: bool = False  # bfloat16 autocast on CUDA: full fine-tuning at 256 tokens fits a 4 GB GPU


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_loss: float | None
    val_accuracy: float | None
    val_macro_f1: float | None
    seconds: float


@dataclass
class TrainReport:
    labels: list[str]
    config: dict[str, Any]
    n_train: int
    n_val: int
    device: str
    history: list[EpochLog] = field(default_factory=list)
    best_epoch: int = 0
    best_val_macro_f1: float | None = None
    seconds: float = 0.0
    corpus: dict[str, Any] = field(default_factory=dict)  # version + sha256 of the frozen corpus

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RRLClassifier:
    def __init__(
        self,
        labels: Sequence[str],
        model_name: str = DEFAULT_MODEL,
        max_length: int = 128,
        device: str | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        context: bool = False,
        temperature: float = 1.0,
    ) -> None:
        """Loads `model_name` with a fresh classification head, unless `model`/`tokenizer` are
        given (load() and tests pass them). context: read each sentence with its neighbours and
        position (see dataset.context_texts). temperature: logits are divided by it before the
        softmax; calibrate() fits it so confidences match accuracy."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # silence the expected "head newly initialized" report
        self.labels = list(labels)
        self.model_name = model_name
        self.max_length = max_length
        self.context = context
        self.temperature = temperature
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.model = model or AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.labels),
            id2label=dict(enumerate(self.labels)),
            label2id={label: i for i, label in enumerate(self.labels)},
        )
        self.model.to(self.device)
        self.report: TrainReport | None = None

    def dataset(self, examples: Sequence[RRLExample], context_pool: Sequence[RRLExample] | None = None) -> RRLDataset:
        return RRLDataset(examples, self.tokenizer, self.labels, self.max_length, self.context, context_pool)

    # --------------------------------------------------------------- training

    def _freeze(self, n_layers: int) -> None:
        base = getattr(self.model, self.model.base_model_prefix)
        for p in self.model.parameters():
            p.requires_grad = True
        if n_layers > 0:
            for p in base.embeddings.parameters():
                p.requires_grad = False
            for layer in base.encoder.layer[:n_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

    def _class_weights(self, ds: RRLDataset) -> torch.Tensor:
        counts = Counter(ds[i]["labels"] for i in range(len(ds)))
        total = sum(counts.values())
        w = [total / (len(self.labels) * counts[i]) if counts.get(i) else 0.0 for i in range(len(self.labels))]
        return torch.tensor(w, dtype=torch.float, device=self.device)

    def train(
        self,
        train_ds: RRLDataset,
        val_ds: RRLDataset | None,
        cfg: TrainConfig,
        log: Callable[[str], None] = print,
    ) -> TrainReport:
        """Fine-tune; after each epoch evaluate on `val_ds` and keep the weights with the best
        validation macro-F1 (or the last epoch's, without a validation set)."""
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self._freeze(cfg.freeze_layers)

        head = [p for n, p in self.model.named_parameters() if p.requires_grad and n.startswith("classifier")]
        body = [p for n, p in self.model.named_parameters() if p.requires_grad and not n.startswith("classifier")]
        optimizer = torch.optim.AdamW(
            [{"params": body, "lr": cfg.lr}, {"params": head, "lr": cfg.head_lr}], weight_decay=cfg.weight_decay
        )
        sampler = LengthBucketSampler(train_ds.lengths(), cfg.batch_size, cfg.seed, pool_batches=cfg.bucket_pool_batches)
        loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate(self.tokenizer))
        total_steps = len(loader) * cfg.epochs if cfg.max_steps is None else min(cfg.max_steps, len(loader) * cfg.epochs)
        warmup = max(1, int(cfg.warmup_ratio * total_steps))

        def lr_lambda(step: int) -> float:  # linear warmup, then linear decay
            return step / warmup if step < warmup else max(0.0, (total_steps - step) / max(1, total_steps - warmup))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        loss_fn = torch.nn.CrossEntropyLoss(weight=self._class_weights(train_ds) if cfg.class_weighted else None)

        report = TrainReport(self.labels, asdict(cfg), len(train_ds), len(val_ds) if val_ds else 0, self.device)
        best_state: dict[str, torch.Tensor] | None = None
        best_score = -math.inf
        step, start = 0, time.perf_counter()
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log(f"training {trainable / 1e6:.1f}M of {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M "
            f"parameters on {len(train_ds)} sentences, {len(loader)} batches/epoch, device {self.device}")

        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            t0, losses = time.perf_counter(), []
            for i, batch in enumerate(loader, start=1):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16 and self.device == "cuda"):
                    logits = self.model(**batch).logits
                loss = loss_fn(logits.float(), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                losses.append(loss.item())
                step += 1
                if i % 50 == 0:
                    log(f"  epoch {epoch} batch {i}/{len(loader)} loss {np.mean(losses[-50:]):.4f} "
                        f"({time.perf_counter() - t0:.0f}s)")
                if cfg.max_steps is not None and step >= cfg.max_steps:
                    break
            entry = EpochLog(epoch, float(np.mean(losses)), None, None, None, 0.0)
            if val_ds is not None and len(val_ds):
                metrics, val_loss = self.evaluate(val_ds, loss_fn)
                entry.val_loss, entry.val_accuracy, entry.val_macro_f1 = val_loss, metrics.accuracy, metrics.macro_f1
                score = metrics.macro_f1
            else:
                score = float(epoch)
            entry.seconds = round(time.perf_counter() - t0, 1)
            report.history.append(entry)
            log(f"epoch {epoch}: train loss {entry.train_loss:.4f}"
                + (f", val loss {entry.val_loss:.4f}, val accuracy {entry.val_accuracy:.3f}, val macro-F1 {entry.val_macro_f1:.3f}"
                   if entry.val_loss is not None else "") + f" ({entry.seconds:.0f}s)")
            if score > best_score:
                best_score, report.best_epoch = score, epoch
                best_state = copy.deepcopy(self.model.state_dict())
            if cfg.max_steps is not None and step >= cfg.max_steps:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        report.best_val_macro_f1 = best_score if val_ds is not None and len(val_ds) else None
        report.seconds = round(time.perf_counter() - start, 1)
        self.report = report
        return report

    # ------------------------------------------------------------- inference

    @torch.no_grad()
    def _logits(self, ds: RRLDataset, batch_size: int = 32) -> np.ndarray:
        self.model.eval()
        sampler = LengthBucketSampler(ds.lengths(), batch_size, seed=0, shuffle=False)
        out = np.zeros((len(ds), len(self.labels)), dtype=np.float32)
        items = [{k: v for k, v in ds[i].items() if k != "labels"} for i in range(len(ds))]
        pad = collate(self.tokenizer)
        for idxs in sampler:
            batch = {k: v.to(self.device) for k, v in pad([items[i] for i in idxs]).items()}
            out[idxs] = self.model(**batch).logits.float().cpu().numpy()
        return out

    def _probabilities(self, ds: RRLDataset, batch_size: int = 32) -> np.ndarray:
        return torch.softmax(torch.from_numpy(self._logits(ds, batch_size)) / self.temperature, dim=-1).numpy()

    def calibrate(self, val_ds: RRLDataset) -> float:
        """Temperature scaling (Guo et al., 2017): one number T, fitted on held-out cases, that
        rescales confidence so "90% sure" means right about 90% of the time. Accuracy is unchanged
        (the top label never changes). Fully fine-tuned models are usually overconfident (T > 1)."""
        logits = torch.from_numpy(self._logits(val_ds))
        gold = torch.tensor([val_ds.label2id[e.label] for e in val_ds.examples])
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), gold)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(log_t.detach().exp())
        return self.temperature

    @torch.no_grad()
    def evaluate(self, ds: RRLDataset, loss_fn: Any | None = None) -> tuple[Metrics, float]:
        probs = self._probabilities(ds)
        y_true = [ds.examples[i].label for i in range(len(ds))]
        y_pred = [self.labels[j] for j in probs.argmax(axis=1)]
        gold = torch.tensor([ds.label2id[y] for y in y_true])
        loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
        loss = float(loss_fn(torch.log(torch.from_numpy(probs).clamp_min(1e-9)).to(self.device), gold.to(self.device)))
        return ClassifierEvaluator().evaluate(y_true, y_pred, self.labels), loss

    def predict(self, sentences: Sequence[RRLExample] | Sequence[tuple[str, str]]) -> list[Prediction]:
        """`sentences` are RRLExamples or (sentence_id, text) pairs; labels, if present, are ignored."""
        examples = [
            s if isinstance(s, RRLExample) else RRLExample(sentence_id=s[0], case_id="", idx=-1, text=s[1])
            for s in sentences
        ]
        if not examples:
            return []
        unlabelled = [RRLExample(e.sentence_id, e.case_id, e.idx, e.text, None) for e in examples]
        probs = self._probabilities(self.dataset(unlabelled))  # neighbours: the other sentences given
        out = []
        for e, p in zip(examples, probs):
            order = np.argsort(p)[::-1]
            top, second = float(p[order[0]]), float(p[order[1]]) if len(p) > 1 else 0.0
            out.append(Prediction(
                sentence_id=e.sentence_id,
                label=self.labels[int(order[0])],
                probabilities={label: round(float(v), 6) for label, v in zip(self.labels, p)},
                confidence=round(top, 6),
                margin=round(top - second, 6),
                case_id=e.case_id,
                idx=e.idx,
                text=e.text,
            ))
        return out

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        meta = {
            "labels": self.labels,
            "base_model": self.model_name,
            "max_length": self.max_length,
            "context": self.context,
            "temperature": self.temperature,
            "report": self.report.to_dict() if self.report else None,
            **(extra or {}),
        }
        (path / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "RRLClassifier":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path)
        meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))
        clf = cls(
            meta["labels"],
            model_name=meta["base_model"],
            max_length=meta["max_length"],
            device=device,
            model=AutoModelForSequenceClassification.from_pretrained(path),
            tokenizer=AutoTokenizer.from_pretrained(path),
            context=meta.get("context", False),  # checkpoints before context/calibration: plain sentences
            temperature=meta.get("temperature", 1.0),
        )
        clf.meta = meta  # type: ignore[attr-defined]
        return clf
