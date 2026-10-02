"""Fine-tune a small open LLM with LoRA on instruction examples built from a frozen corpus.

Run from the GPU environment (backend/requirements-gpu.txt):
    .venv-gpu\\Scripts\\python backend/scripts/train_lora.py --version v0.2
    .venv-gpu\\Scripts\\python backend/scripts/train_lora.py --version v0.2 --max-steps 2   # smoke test

Saves the adapter (not the base model) to data/models/lora/<version>-<time>/ together with
examples.jsonl (every example and its split), so the comparison uses this adapter's own
held-out cases.
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation.store import AnnotationStore  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.finetuning.instruction_builder import InstructionBuilder  # noqa: E402
from app.finetuning.lora_trainer import LoRATrainer, TrainConfig  # noqa: E402


def main() -> int:
    d = TrainConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v0.2")
    ap.add_argument("--base-model", default=d.base_model)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--r", type=int, default=d.r)
    ap.add_argument("--lora-alpha", type=int, default=d.lora_alpha)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    s = get_settings()
    manifest, sentences = AnnotationStore(s.annotation_store_dir).load_version(args.version, verify=True)
    builder = InstructionBuilder(seed=args.seed)
    examples = builder.build_corpus(sentences)
    splits = builder.split_by_case(examples, seed=args.seed)
    case_sets = {k: {e.source_case_id for e in v} for k, v in splits.items()}
    assert not (case_sets["train"] & case_sets["val"] or case_sets["train"] & case_sets["test"] or case_sets["val"] & case_sets["test"])
    print(f"corpus {args.version} (sha256 {manifest.sha256[:12]}…): {len(examples)} examples {dict(Counter(e.task for e in examples))}")
    for k, v in splits.items():
        print(f"  {k:5}: {len(v):3} examples from {len(case_sets[k])} cases {sorted(case_sets[k]) if k != 'train' else ''}")

    cfg = TrainConfig(base_model=args.base_model, r=args.r, lora_alpha=args.lora_alpha, lr=args.lr,
                      epochs=args.epochs, seed=args.seed, max_steps=args.max_steps)
    out = args.out or s.lora_dir / f"{args.version}-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "examples.jsonl").open("w", encoding="utf-8") as f:
        for split, exs in splits.items():
            for e in exs:
                f.write(json.dumps({"split": split, **asdict(e)}, ensure_ascii=False) + "\n")

    print(f"config: {asdict(cfg)}")
    trainer = LoRATrainer(cfg, builder)
    corpus = {"version": args.version, "sha256": manifest.sha256}
    report = trainer.train(splits["train"], splits["val"], out, corpus=corpus)
    print(f"\ntrain loss {report.train_loss:.4f}; eval loss {report.eval_loss_before:.4f} (untuned) -> {report.eval_loss:.4f} (tuned)")
    print(f"{report.seconds:.0f}s, peak GPU memory {report.peak_gpu_mem_gb} GB")
    print(f"saved adapter to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
