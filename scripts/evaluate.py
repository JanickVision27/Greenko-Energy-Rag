#!/usr/bin/env python3
"""Run the bundled evaluation set (``data/questions.csv``) through the real pipeline.

No mock scores: this builds the same ``RAGService`` the API uses, asks every
question, and records what actually happened — retrieved source, citations, the
answer text, the hallucination verdict and whether the expected source came back
first. The Groq status (``ok`` / ``not_configured`` / ``error``) is stored per
question so a run without a key is never mistaken for a model evaluation.

Usage
-----
    python -m scripts.evaluate                 # writes outputs/evaluation_results.json
    python -m scripts.evaluate --top-k 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env", override=True)

from src import chunker  # noqa: E402
from src.llm import MODEL_NAME  # noqa: E402
from src.pipeline import DEFAULT_DATA_DIR, RAGService  # noqa: E402

LOGGER = logging.getLogger("evaluate")


def run(data_dir: Path, questions_csv: Path, top_k: int, output: Path) -> dict:
    """Run labeled questions through the real pipeline and write a JSON report."""
    service = RAGService.build(data_dir, top_k=top_k)
    frame = pd.read_csv(questions_csv)
    for column in ("question", "expected_source"):
        if column not in frame.columns:
            raise ValueError(f"{questions_csv} is missing the '{column}' column")

    records = []
    # Iterate as named row objects instead of repeatedly indexing the DataFrame.
    for row in frame.itertuples(index=False):
        question = str(row.question).strip()
        expected_source = str(row.expected_source).strip()
        result = service.answer(question)

        retrieved_sources = [item["source"] for item in result["retrieved_chunks"]]
        top_source = retrieved_sources[0] if retrieved_sources else None
        # Rank is one-based for readers; None means the expected source was not found.
        expected_rank = (
            retrieved_sources.index(expected_source) + 1 if expected_source in retrieved_sources else None
        )
        records.append(
            {
                "question": question,
                "expected_source": expected_source,
                "retrieved_top_source": top_source,
                "source_correct": top_source == expected_source,
                "expected_source_rank": expected_rank,
                "retrieved_sources": retrieved_sources,
                "retrieval_scores": [item["score"] for item in result["retrieved_chunks"]],
                "answer": result["answer"],
                "citations": result["citations"],
                "hallucination_check": result["hallucination_check"],
                "hallucination_details": result["hallucination_details"],
                "llm_status": result["llm_status"],
                "llm_error": result["llm_error"],
                "model": result["model"],
                "embedding_model": result["embedding_model"],
                "latency_ms": result["latency_ms"],
            }
        )
        LOGGER.info(
            "%s | expected=%s top=%s correct=%s check=%s llm=%s",
            question[:60],
            expected_source,
            top_source,
            top_source == expected_source,
            result["hallucination_check"],
            result["llm_status"],
        )

    total = len(records)
    correct = sum(1 for record in records if record["source_correct"])
    llm_ok = sum(1 for record in records if record["llm_status"] == "ok")
    # Keep aggregate metrics alongside per-question details for easy review.
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline": {
            "llm_model": MODEL_NAME,
            "llm_key_present": bool(os.getenv("GROQ_API_KEY", "").strip()),
            "embedding_model": service.embedder.display_name,
            "embedding_backend": service.embedder.backend,
            "embedding_dim": service.embedder.dimension,
            "embedding_fallback_reason": service.embedder.fallback_reason,
            "top_k": top_k,
            "corpus": chunker.corpus_stats(service.documents, service.chunks),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "summary": {
            "questions": total,
            "source_attribution_correct": correct,
            "source_accuracy": round(correct / total, 4) if total else 0.0,
            "llm_answers_generated": llm_ok,
            "mean_latency_ms": round(sum(r["latency_ms"] for r in records) / total, 1) if total else 0.0,
        },
        "questions": records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s (%d/%d source-correct)", output, correct, total)
    return report


def main() -> int:
    """Parse command-line options, run evaluation, and print a short summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on data/questions.csv")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--questions", type=Path, default=DEFAULT_DATA_DIR / "questions.csv")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "evaluation_results.json")
    # argparse supplies defaults while allowing alternate inputs and output paths.
    args = parser.parse_args()

    report = run(args.data_dir, args.questions, args.top_k, args.output)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
