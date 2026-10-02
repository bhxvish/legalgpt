"""Compare the hosted base model, the untuned local base and the LoRA-tuned model on held-out
questions through the same Retriever, and save a side-by-side report.

Held-out questions come from the adapter's *test* cases (never seen in fine-tuning): one
question built from each case's facts, plus one per IPC section the case cites.

Run from the GPU environment:
    .venv-gpu\\Scripts\\python backend/scripts/compare_models.py                 # newest adapter
    .venv-gpu\\Scripts\\python backend/scripts/compare_models.py --skip groq
Writes comparison.json and comparison.md into the adapter's directory.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation.collector import JudgmentCollector  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.core.embeddings import EmbeddingService  # noqa: E402
from app.core.llm_client import AdapterClient, GroqClient, LocalHFClient, latest_adapter  # noqa: E402
from app.core.retriever import Retriever  # noqa: E402
from app.core.vector_store import VectorStore  # noqa: E402
from app.finetuning.model_comparator import ModelComparator  # noqa: E402


def held_out_questions(adapter: Path, max_questions: int) -> list[tuple[str, str]]:
    rows = [json.loads(l) for l in (adapter / "examples.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    test = [r for r in rows if r["split"] == "test" and r["task"] == "facts_to_law"]
    questions: list[tuple[str, str]] = []
    for r in test:
        facts = r["input"].split(". ")
        questions.append((". ".join(facts[:2]).strip().rstrip(".") + ". Which provisions of the Indian Penal Code apply to these facts?",
                          f"facts of held-out case {r['source_case_id']}"))
    for r in test:
        for sec in JudgmentCollector.detect_sections(r["input"] + " " + r["output"])[:3]:
            q = f"What does Section {sec} of the Indian Penal Code provide, and what punishment does it prescribe?"
            if all(q != x for x, _ in questions):
                questions.append((q, f"IPC section cited in held-out case {r['source_case_id']}"))
    return questions[:max_questions]


def main() -> int:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--max-questions", type=int, default=8)
    ap.add_argument("--skip", nargs="*", default=[], choices=["groq", "base", "tuned"])
    args = ap.parse_args()

    adapter = args.adapter or latest_adapter(s.lora_dir)
    if adapter is None:
        raise SystemExit(f"no adapter under {s.lora_dir}; run backend/scripts/train_lora.py first")
    questions = held_out_questions(adapter, args.max_questions)
    print(f"adapter {adapter.name}; {len(questions)} held-out questions")
    tuned = AdapterClient(adapter)
    clients = {
        "groq": GroqClient(s.groq_api_key, s.groq_model),
        "base": LocalHFClient(tuned.base_model),
        "tuned": tuned,
    }
    clients = {k: v for k, v in clients.items() if k not in args.skip}
    retriever = Retriever(EmbeddingService(s.embedding_model), VectorStore(s.chroma_persist_dir, s.chroma_collection),
                          min_similarity=s.retrieval_min_similarity)
    comparator = ModelComparator(clients, questions)
    rows = comparator.run(retriever)
    meta = {"adapter": adapter.name, "base_model": tuned.base_model, "questions": len(questions),
            "retrieval_min_similarity": s.retrieval_min_similarity,
            "note": "citation_accuracy is computed against the evidence actually retrieved for each question"}
    json_path, md_path = comparator.save(rows, adapter, meta)
    print("\n" + json.dumps(comparator.summary(rows), indent=2))
    print(f"\nwrote {json_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
