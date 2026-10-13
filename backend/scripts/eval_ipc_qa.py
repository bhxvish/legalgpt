"""Compare LoRA adapters on IPC questions about sections held out of training (ipc_qa.EVAL_SECTIONS).

For every held-out section that the retriever returns for the question, each adapter answers through
the app's own AdapterClient (same prompt as the chat). Scored per answer:
  answered        not the "I don't know" reply
  right_citation  cites the [n] marker of the asked section's source
  valid_markers   every [n] it uses exists among the sources
plus answer length and time. Writes a markdown + JSON report.

Usage (GPU environment recommended):
    .venv-gpu\\Scripts\\python backend/scripts/eval_ipc_qa.py data/models/lora/v0.2-20261002-203807 data/models/lora/v0.3-ipcqa
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.core.corpus import load_chunks  # noqa: E402
from app.core.embeddings import EmbeddingService  # noqa: E402
from app.core.llm_client import AdapterClient  # noqa: E402
from app.core.prompt_builder import NOT_COVERED_REPLY, PromptBuilder  # noqa: E402
from app.core.retriever import Retriever  # noqa: E402
from app.core.vector_store import VectorStore  # noqa: E402
from app.finetuning.ipc_qa import EVAL_SECTIONS, main_chunk  # noqa: E402

MARKER = re.compile(r"\[(\d+)\]")


def questions(chunks):
    out = []
    for section in EVAL_SECTIONS:
        chunk = main_chunk(chunks, section)
        if chunk is None:
            continue
        title = chunk.title.strip().rstrip(".")
        topic = re.sub(r"^punishment for\s+", "", title, flags=re.I)
        if "shall be punished" in chunk.text and topic.lower() != title.lower():
            out.append((section, chunk, f"What is the punishment for {topic[:1].lower() + topic[1:]}?"))
        out.append((section, chunk, f"What does Section {section} of the IPC provide?"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapters", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="report path without extension (default data/reports/ipc_qa_eval)")
    args = ap.parse_args()
    s = get_settings()
    chunks = load_chunks(s.corpus_chunks_path)
    retriever = Retriever(EmbeddingService(s.embedding_model), VectorStore(s.chroma_persist_dir, s.chroma_collection),
                          min_similarity=s.retrieval_min_similarity)
    items = []
    for section, chunk, q in questions(chunks):
        sources = retriever.retrieve(q)
        hit = next((x for x in sources if x.chunk_id == chunk.chunk_id), None)
        if hit is not None:  # only questions where the right section reached the model
            items.append({"section": section, "question": q, "marker": hit.marker,
                          "markers": [x.marker for x in sources],
                          "messages": PromptBuilder().build(q, sources, "legal", [])})
    print(f"{len(items)} questions on {len({i['section'] for i in items})} held-out sections")

    results = {}
    for path in args.adapters:
        client = AdapterClient(path)
        client._load()
        rows = []
        for it in items:
            t = time.perf_counter()
            answer = client.generate(it["messages"])
            used = [int(m) for m in MARKER.findall(answer)]
            rows.append({"section": it["section"], "question": it["question"], "answer": answer,
                         "seconds": round(time.perf_counter() - t, 2),
                         "answered": NOT_COVERED_REPLY.split(" —")[0].lower() not in answer.lower(),
                         "right_citation": it["marker"] in used,
                         "valid_markers": all(m in it["markers"] for m in used) and bool(used)})
        n = len(rows)
        summary = {"answered": sum(r["answered"] for r in rows) / n,
                   "right_citation": sum(r["right_citation"] for r in rows) / n,
                   "valid_markers": sum(r["valid_markers"] for r in rows) / n,
                   "mean_chars": round(sum(len(r["answer"]) for r in rows) / n),
                   "mean_seconds": round(sum(r["seconds"] for r in rows) / n, 2),
                   "device": f"{client.loaded_dtype}"}
        results[path.name] = {"summary": summary, "rows": rows}
        print(path.name, json.dumps(summary))
        client.close()
        del client

    out = args.out or s.lora_dir.parent.parent / "reports" / "ipc_qa_eval"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    names = list(results)
    lines = ["# LoRA adapters on held-out IPC sections", "",
             f"{len(items)} questions on {len({i['section'] for i in items})} IPC sections never used in training "
             "(app/finetuning/ipc_qa.py EVAL_SECTIONS); only questions whose section the retriever returned.", "",
             "| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, label in [("answered", "answered (not \"I don't know\")"), ("right_citation", "cites the asked section"),
                       ("valid_markers", "all citations valid"), ("mean_chars", "mean answer length (chars)"),
                       ("mean_seconds", "mean seconds per answer")]:
        vals = [results[n]["summary"][key] for n in names]
        fmt = [f"{v:.0%}" if isinstance(v, float) and key not in ("mean_seconds",) else str(v) for v in vals]
        lines.append(f"| {label} | " + " | ".join(fmt) + " |")
    lines += ["", "## Sample answers", ""]
    for i, it in enumerate(items[:6]):
        lines.append(f"**{it['question']}**")
        for n in names:
            lines.append(f"- *{n}*: {results[n]['rows'][i]['answer'][:300]}")
        lines.append("")
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
