"""Why did InLegalBERT suggest this role? Word attributions and similar labelled examples.

  integrated_gradients  Integrated Gradients (Sundararajan et al., 2017) on the input embeddings, 128
                        steps by default (the completeness gap fell from 1.9 logits at 32 steps to
                        0.01 at 128 on a test sentence):
                        for each token, how much it moved the chosen role's logit compared with a
                        baseline where every word is [PAD] (special tokens and context positions
                        kept). Attributions sum to logit(input) - logit(baseline) up to the
                        integration error, which is reported as `completeness_gap`.
  word attributions     word-piece scores summed into the words of the sentence, with their
                        character spans, plus the share of attribution that went to the context
                        segment (position + neighbouring sentences) rather than the sentence
  occlusion             leave-one-word-out: the drop in the role's probability when a word is
                        removed (used to check how faithful the attributions are)
  neighbours            the most similar human-labelled sentences by the classifier's own pooled
                        representation (an example-based explanation: "it looks like these")

The classifier is not thread-safe for gradients, so every call takes a lock.
"""

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.labeling_assistant.dataset import RRLExample, context_texts

_LOCK = threading.Lock()


@dataclass
class WordScore:
    text: str
    start: int  # character offsets in the sentence
    end: int
    score: float


class ClassifierExplainer:
    def __init__(self, clf, neighbours_path: Path | None = None) -> None:
        self.clf = clf
        self.tok = clf.tokenizer
        self.model = clf.model
        self._index = None
        self.neighbours_path = Path(neighbours_path) if neighbours_path else None

    # ------------------------------------------------------------------ inputs
    def _context(self, example: RRLExample, case: list[RRLExample] | None) -> str:
        if not self.clf.context:
            return ""
        return context_texts(case or [example]).get(example.sentence_id, "")

    def _encode(self, text: str, context: str) -> dict[str, Any]:
        args = (text, context) if context else (text,)
        enc = self.tok(*args, truncation="longest_first" if context else True, max_length=self.clf.max_length,
                       return_offsets_mapping=True, return_tensors="pt")
        seq = enc.sequence_ids(0) if hasattr(enc, "sequence_ids") else None
        return {"input_ids": enc["input_ids"], "token_type_ids": enc.get("token_type_ids"),
                "offsets": enc["offset_mapping"][0].tolist(), "seq": seq}

    def _forward(self, embeds: torch.Tensor, token_type_ids: torch.Tensor | None) -> torch.Tensor:
        kwargs = {"inputs_embeds": embeds, "attention_mask": torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.long)}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids.expand(embeds.shape[0], -1)
        return self.model(**kwargs).logits.float()

    # ------------------------------------------------------------------ predictions
    @torch.no_grad()
    def probabilities(self, text: str, context: str = "") -> np.ndarray:
        enc = self._encode(text, context)
        dev = self.clf.device
        kwargs = {"input_ids": enc["input_ids"].to(dev)}
        if enc["token_type_ids"] is not None:
            kwargs["token_type_ids"] = enc["token_type_ids"].to(dev)
        logits = self.model(**kwargs).logits.float()[0]
        return torch.softmax(logits / self.clf.temperature, dim=-1).cpu().numpy()

    # ------------------------------------------------------------------ integrated gradients
    def integrated_gradients(self, text: str, context: str, target: int, steps: int = 128, batch: int = 16) -> dict[str, Any]:
        enc = self._encode(text, context)
        dev = self.clf.device
        ids = enc["input_ids"].to(dev)
        tt = enc["token_type_ids"].to(dev) if enc["token_type_ids"] is not None else None
        emb_layer = self.model.get_input_embeddings()
        with torch.no_grad():
            x = emb_layer(ids)  # (1, T, H)
            special = torch.tensor([t in self.tok.all_special_ids for t in ids[0].tolist()], device=dev)
            base_ids = torch.where(special, ids[0], torch.full_like(ids[0], self.tok.pad_token_id)).unsqueeze(0)
            baseline = emb_layer(base_ids)
        alphas = (torch.arange(steps, device=dev, dtype=torch.float) + 0.5) / steps  # midpoint rule
        total = torch.zeros_like(x)
        was_training = self.model.training
        self.model.eval()
        for i in range(0, steps, batch):
            a = alphas[i:i + batch].view(-1, 1, 1)
            path = (baseline + a * (x - baseline)).detach().requires_grad_(True)
            logits = self._forward(path, tt)
            grads, = torch.autograd.grad(logits[:, target].sum(), path)
            total += grads.sum(dim=0, keepdim=True)
        self.model.train(was_training)
        attr = ((total / steps) * (x - baseline)).sum(dim=-1)[0]  # (T,)
        with torch.no_grad():
            f_x = self._forward(x, tt)[0, target].item()
            f_b = self._forward(baseline, tt)[0, target].item()
        return {"token_scores": attr.detach().cpu().numpy(), "enc": enc,
                "completeness_gap": float(abs(attr.sum().item() - (f_x - f_b)))}

    def word_scores(self, text: str, enc: dict[str, Any], token_scores: np.ndarray) -> tuple[list[WordScore], float]:
        """Sum word-piece scores into whitespace-delimited words of the sentence (segment 0) and
        return the share of absolute attribution that went to the context segment."""
        seq = enc["seq"] or [0 if (o[1] > 0) else None for o in enc["offsets"]]
        words: list[WordScore] = []
        sent_abs = ctx_abs = 0.0
        for score, (s, e), sid in zip(token_scores.tolist(), enc["offsets"], seq):
            if sid == 1:
                ctx_abs += abs(score)
                continue
            if sid != 0 or e <= s:
                continue
            sent_abs += abs(score)
            if words and s == words[-1].end:  # no space before it: a "##" piece or punctuation of the same word
                w = words[-1]
                words[-1] = WordScore(text[w.start:e], w.start, e, w.score + score)
            else:
                words.append(WordScore(text[s:e], s, e, score))
        total = sent_abs + ctx_abs
        return words, (ctx_abs / total if total else 0.0)

    # ------------------------------------------------------------------ occlusion
    def occlusion(self, text: str, context: str, target: int, words: list[WordScore]) -> list[float]:
        """Drop in the target role's probability when each word is removed from the sentence."""
        p0 = self.probabilities(text, context)[target]
        out = []
        for w in words:
            reduced = (text[:w.start] + text[w.end:]).replace("  ", " ").strip()
            out.append(float(p0 - self.probabilities(reduced or ".", context)[target]))
        return out

    # ------------------------------------------------------------------ neighbours
    @torch.no_grad()
    def pooled(self, texts: list[str], contexts: list[str], batch: int = 32) -> np.ndarray:
        base = getattr(self.model, self.model.base_model_prefix)
        out = []
        for i in range(0, len(texts), batch):
            t, c = texts[i:i + batch], contexts[i:i + batch]
            enc = self.tok(t, c, truncation="longest_first", max_length=self.clf.max_length, padding=True,
                           return_tensors="pt") if self.clf.context else self.tok(t, truncation=True,
                                                                                  max_length=self.clf.max_length,
                                                                                  padding=True, return_tensors="pt")
            enc = {k: v.to(self.clf.device) for k, v in enc.items()}
            v = base(**enc).pooler_output.float()
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 768), dtype=np.float32)

    def _load_index(self):
        if self._index is None and self.neighbours_path and self.neighbours_path.exists():
            z = np.load(self.neighbours_path)
            meta = json.loads(self.neighbours_path.with_suffix(".json").read_text(encoding="utf-8"))
            self._index = (z["vectors"].astype(np.float32), meta)
        return self._index

    def neighbours(self, text: str, context: str, k: int = 3, exclude_case: str = "") -> list[dict[str, Any]]:
        index = self._load_index()
        if index is None:
            return []
        vecs, meta = index
        q = self.pooled([text], [context])[0]
        sims = vecs @ q
        out = []
        for i in np.argsort(-sims):
            m = meta[i]
            if m["case_id"] == exclude_case or m["text"].strip() == text.strip():
                continue
            out.append({**m, "similarity": round(float(sims[i]), 3)})
            if len(out) == k:
                break
        return out

    # ------------------------------------------------------------------ all together
    def explain(self, example: RRLExample, case: list[RRLExample] | None = None, label: str | None = None,
                steps: int = 128) -> dict[str, Any]:
        with _LOCK:
            context = self._context(example, case)
            probs = self.probabilities(example.text, context)
            target = self.clf.labels.index(label) if label else int(np.argmax(probs))
            ig = self.integrated_gradients(example.text, context, target, steps=steps)
            words, ctx_share = self.word_scores(example.text, ig["enc"], ig["token_scores"])
            # word removal: more faithful than IG on the deletion test (docs/reports/classifier_explanations.md)
            occ = self.occlusion(example.text, context, target, words)
            neighbours = self.neighbours(example.text, context, exclude_case=example.case_id)
        top = max((abs(w.score) for w in words), default=1.0) or 1.0
        top_occ = max((abs(o) for o in occ), default=1.0) or 1.0
        return {
            "label": self.clf.labels[target],
            "probabilities": {lab: round(float(p), 4) for lab, p in zip(self.clf.labels, probs)},
            "words": [{"text": w.text, "start": w.start, "end": w.end, "score": round(w.score, 4),
                       "weight": round(w.score / top, 3), "removal_drop": round(o, 4),
                       "removal_weight": round(o / top_occ, 3)} for w, o in zip(words, occ)],
            "context_share": round(ctx_share, 3),
            "completeness_gap": round(ig["completeness_gap"], 4),
            "neighbours": neighbours,
            "method": (f"Word removal (confidence drop when each word is left out) or Integrated Gradients ({steps} steps, "
                       "[PAD] baseline); neighbours by pooled InLegalBERT embedding"),
        }
