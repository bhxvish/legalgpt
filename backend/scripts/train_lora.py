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
from app.config import REPO_ROOT, get_settings  # noqa: E402
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
    ap.add_argument("--save-steps", type=int, default=0, help="checkpoint every N steps and resume from the latest (Colab)")
    ap.add_argument("--no-judgments", action="store_true",
                    help="skip the judgment-based examples (Module 1 corpus), e.g. when the judgments are not uploaded")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ipc-qa", action="store_true",
                    help="add grounded Q&A examples built from the IPC with the real retriever (app/finetuning/ipc_qa.py)")
    ap.add_argument("--scenarios", action="store_true",
                    help="add everyday fact-pattern examples (app/finetuning/scenario_qa.py); never the benchmark's held-out 60")
    ap.add_argument("--scenario-refusals", type=float, default=0.0, metavar="SHARE",
                    help="share of scenario examples that are refusals (sources without a labelled section), e.g. 0.25")
    ap.add_argument("--rewrites", action="store_true",
                    help="also teach the query rewrite (distilled from cached larger-model rewrites that named a labelled section)")
    ap.add_argument("--no-case-refusals", action="store_true",
                    help="drop the judgment-based refusal examples (they taught v0.2 to refuse IPC sources)")
    args = ap.parse_args()

    s = get_settings()
    builder = InstructionBuilder(seed=args.seed)
    if args.no_judgments:
        corpus_sha, examples = "none (--no-judgments)", []
    else:
        manifest, sentences = AnnotationStore(s.annotation_store_dir).load_version(args.version, verify=True)
        corpus_sha, examples = manifest.sha256, builder.build_corpus(sentences)
    if args.no_case_refusals:
        examples = [e for e in examples if e.task != "grounded_refusal"]
    if args.ipc_qa:
        from app.core.corpus import load_chunks
        from app.core.embeddings import EmbeddingService
        from app.core.retriever import Retriever
        from app.core.vector_store import VectorStore
        from app.finetuning.ipc_qa import EVAL_SECTIONS, IpcQaBuilder

        chunks = load_chunks(s.corpus_chunks_path)
        retriever = Retriever(EmbeddingService(s.embedding_model), VectorStore(s.chroma_persist_dir, s.chroma_collection),
                              min_similarity=s.retrieval_min_similarity)
        ipc = IpcQaBuilder(retriever, chunks, seed=args.seed).build(sorted({c.section for c in chunks if c.section}))
        print(f"IPC Q&A: {len(ipc)} examples; {len(EVAL_SECTIONS)} sections held out for evaluation")
        examples += ipc
    if args.scenarios:
        import json as json_mod

        from app.core.corpus import load_chunks
        from app.core.query_rewriter import parse
        from app.core.retriever import build_retriever
        from app.finetuning.scenario_qa import ScenarioQaBuilder

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from eval_scenario_bench import splits as bench_splits

        pool = bench_splits()
        held_out = {q["id"] for q in pool["final"]}
        train_scen = [q for q in pool["tune"] + pool["check"] if q["id"] not in held_out]
        caches = [json_mod.loads(f.read_text(encoding="utf-8"))  # rewrites cached by eval_scenario_bench.py
                  for f in sorted((REPO_ROOT / "data" / "reports" / "bench" / "rewrites").glob("*.json"), reverse=True)]

        def rewrite_for(q: str):
            return next((parse(c[q]) for c in caches if q in c), None)

        if args.rewrites:  # teacher rewrites, best models first; gpt-oss-20b (weak) is not used as a teacher
            teachers = [REPO_ROOT / "data" / "reports" / "bench" / "rewrites" / f for f in
                        ("groq_qwen_qwen3.8-27b.json", "gemini_gemini-3.5-flash-lite.json", "groq_openai_gpt-oss-120b.json")]
            tcaches = [json_mod.loads(f.read_text(encoding="utf-8")) for f in teachers if f.exists()]
            scope_ex = json_mod.loads((REPO_ROOT / "backend" / "app" / "core" / "scope_examples.json").read_text(encoding="utf-8"))
            civil = [q["question"] for q in pool["oos-tune"]] + list(scope_ex["out_of_scope"])
            rw_ex = ScenarioQaBuilder.rewrite_examples(
                train_scen, lambda q: next((parse(c[q]) for c in tcaches if q in c), None), list(dict.fromkeys(civil)))
            n_civil = sum(e.source_case_id.startswith("civil:") for e in rw_ex)
            print(f"rewrites: {len(rw_ex) - n_civil} criminal + {n_civil} civil examples")
            examples += rw_ex

        scen = ScenarioQaBuilder(build_retriever(s), load_chunks(s.corpus_chunks_path), seed=args.seed).build(train_scen, rewrite_for, refusal_share=args.scenario_refusals)
        print(f"scenarios: {len(scen)} examples from {len({e.source_case_id for e in scen})} of {len(train_scen)} questions "
              f"(held-out benchmark scenarios excluded: {len(held_out)})")
        examples += scen
    splits = builder.split_by_case(examples, seed=args.seed)
    case_sets = {k: {e.source_case_id for e in v} for k, v in splits.items()}
    assert not (case_sets["train"] & case_sets["val"] or case_sets["train"] & case_sets["test"] or case_sets["val"] & case_sets["test"])
    print(f"corpus {args.version} (sha256 {corpus_sha[:12]}…): {len(examples)} examples {dict(Counter(e.task for e in examples))}")
    for k, v in splits.items():
        print(f"  {k:5}: {len(v):3} examples from {len(case_sets[k])} cases {sorted(case_sets[k]) if k != 'train' else ''}")

    cfg = TrainConfig(base_model=args.base_model, r=args.r, lora_alpha=args.lora_alpha, lr=args.lr,
                      epochs=args.epochs, seed=args.seed, max_steps=args.max_steps, save_steps=args.save_steps)
    out = args.out or s.lora_dir / f"{args.version}-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "examples.jsonl").open("w", encoding="utf-8") as f:
        for split, exs in splits.items():
            for e in exs:
                f.write(json.dumps({"split": split, **asdict(e)}, ensure_ascii=False) + "\n")

    print(f"config: {asdict(cfg)}")
    trainer = LoRATrainer(cfg, builder)
    corpus = {"version": args.version, "sha256": corpus_sha}
    report = trainer.train(splits["train"], splits["val"], out, corpus=corpus)
    print(f"\ntrain loss {report.train_loss:.4f}; eval loss {report.eval_loss_before:.4f} (untuned) -> {report.eval_loss:.4f} (tuned)")
    print(f"{report.seconds:.0f}s, peak GPU memory {report.peak_gpu_mem_gb} GB")
    print(f"saved adapter to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
