"""RRLDataset: AnnotationStore sentences as tokenized examples for InLegalBERT."""

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from app.annotation.models import LabeledSentence
from app.annotation.store import AnnotationStore


@dataclass(frozen=True)
class RRLExample:
    sentence_id: str
    case_id: str
    idx: int
    text: str
    label: str | None = None  # None for unlabelled sentences to predict


class RRLDataset(Dataset):
    """Tokenizes on construction (the corpus is small) and keeps the source examples alongside,
    so predictions can be mapped back to sentence ids."""

    def __init__(self, examples: Sequence[RRLExample], tokenizer: Any, labels: Sequence[str], max_length: int = 128) -> None:
        self.examples = list(examples)
        self.labels = list(labels)
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        unknown = sorted({e.label for e in self.examples if e.label is not None and e.label not in self.label2id})
        if unknown:
            raise ValueError(f"labels {unknown} are not in the classifier's label set {self.labels}")
        enc = tokenizer([e.text for e in self.examples], truncation=True, max_length=max_length)
        self.input_ids: list[list[int]] = enc["input_ids"]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item: dict[str, Any] = {"input_ids": self.input_ids[i]}
        label = self.examples[i].label
        if label is not None:
            item["labels"] = self.label2id[label]
        return item

    def lengths(self) -> list[int]:
        return [len(ids) for ids in self.input_ids]

    @property
    def case_ids(self) -> list[str]:
        return sorted({e.case_id for e in self.examples})

    # ------------------------------------------------------------ builders

    @staticmethod
    def examples_from(sentences: Sequence[LabeledSentence]) -> list[RRLExample]:
        return [RRLExample(s.sentence_id, s.case_id, s.idx, s.text, s.label) for s in sentences]

    @classmethod
    def from_version(
        cls, store: AnnotationStore, version: str, split: str, tokenizer: Any, labels: Sequence[str], max_length: int = 128
    ) -> "RRLDataset":
        """One split ("train" / "val" / "test") of a frozen, digest-verified corpus version."""
        manifest, sentences = store.load_version(version, verify=True)
        if split not in manifest.splits:
            raise ValueError(f"unknown split {split!r}; version {version} has {list(manifest.splits)}")
        cases = set(manifest.splits[split])
        return cls(cls.examples_from([s for s in sentences if s.case_id in cases]), tokenizer, labels, max_length)


def collate(tokenizer: Any) -> Any:
    """Pads each batch only to its own longest sentence."""

    def fn(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = tokenizer.pad([{"input_ids": it["input_ids"]} for it in items], return_tensors="pt")
        if "labels" in items[0]:
            batch["labels"] = torch.tensor([it["labels"] for it in items], dtype=torch.long)
        return batch

    return fn


class LengthBucketSampler(Sampler[list[int]]):
    """"Sortish" batching: shuffle, then sort by length only within pools of `pool_batches`
    batches. Batches have similar lengths (much less padding, ~2x faster on CPU) without sorting
    the whole corpus — which would make batches label-homogeneous wherever length correlates with
    the label (long sentences are often quoted Precedent) and bias training.

    With shuffle=False (inference) the whole set is sorted: order does not matter there.
    pool_batches=0 disables length grouping (plain random batches)."""

    def __init__(self, lengths: Sequence[int], batch_size: int, seed: int, shuffle: bool = True, pool_batches: int = 50) -> None:
        self.lengths, self.batch_size, self.seed, self.shuffle = list(lengths), batch_size, seed, shuffle
        self.pool_batches = pool_batches
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.lengths)))
        if not self.shuffle:
            order.sort(key=lambda i: self.lengths[i])
        else:
            rng.shuffle(order)
            if self.pool_batches > 0:
                pool = self.pool_batches * self.batch_size
                order = [i for start in range(0, len(order), pool)
                         for i in sorted(order[start : start + pool], key=lambda i: self.lengths[i])]
        batches = [order[i : i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size
