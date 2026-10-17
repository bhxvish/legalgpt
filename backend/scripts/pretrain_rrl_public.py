"""Stage A: fine-tune InLegalBERT on the public InRhetoricalRoles corpus (mapped to our 5 roles).

The checkpoint is then the starting point for our own judgments:
    train_rrl_classifier.py --model data/models/rrl_public/<checkpoint> ...
Same input format as the deployed model (sentence + position + neighbours, 256 tokens). Sentences
whose public role has no counterpart (PREAMBLE, ISSUE, NONE) are read as context only.

Usage (after build_rrl_public.py; a GPU is needed in practice):
    python backend/scripts/pretrain_rrl_public.py [--epochs 2] [--bf16]
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation.label_scheme import LabelScheme  # noqa: E402
from app.config import REPO_ROOT  # noqa: E402
from app.labeling_assistant.classifier import DEFAULT_MODEL, RRLClassifier, TrainConfig  # noqa: E402
from app.labeling_assistant.dataset import RRLExample  # noqa: E402


def load(path: Path) -> list[RRLExample]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [RRLExample(r["sentence_id"], r["case_id"], r["idx"], r["text"], r["label"]) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "rrl_public")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="smoke test")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    labels = list(LabelScheme.LABELS)
    pools = {split: load(args.data / f"{split}.jsonl") for split in ("train", "dev")}
    clf = RRLClassifier(labels, model_name=args.model, max_length=args.max_length, context=True)
    ds = {split: clf.dataset([e for e in pool if e.label], context_pool=pool) for split, pool in pools.items()}
    print(f"public corpus: train {len(ds['train'])} labelled sentences ({len({e.case_id for e in pools['train']})} judgments), "
          f"dev {len(ds['dev'])} ({len({e.case_id for e in pools['dev']})})")
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, freeze_layers=0,
                      max_steps=args.max_steps, bf16=args.bf16)
    report = clf.train(ds["train"], ds["dev"], cfg)
    metrics, _ = clf.evaluate(ds["dev"])
    print("\n=== public dev set (mapped to our 5 roles) ===")
    print(metrics.format())
    out = args.out or REPO_ROOT / "data" / "models" / "rrl_public" / f"inrr-{datetime.now():%Y%m%d-%H%M%S}"
    clf.save(out, extra={"stage": "public pretraining (InRhetoricalRoles, Kalamkar et al. 2022)",
                         "config": asdict(cfg), "dev_metrics": metrics.to_dict()})
    (out / "metrics.json").write_text(json.dumps({"train_report": report.to_dict(), "dev": metrics.to_dict()}, indent=2),
                                      encoding="utf-8")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
