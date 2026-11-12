"""The RAG service that ties chunking, embeddings, retrieval, generation and the
hallucination check together. Used by the FastAPI app and by the evaluation script
so both exercise exactly the same code path.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from . import chunker, llm
from .chunker import Chunk, Document
from .embedder import Embedder, build_embedder
from .hallucination_check import check_answer
from .retriever import RetrievedChunk, Retriever

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_TOP_K = 3


@dataclass
class RAGService:
    """Loaded, ready-to-query RAG pipeline (all state is in memory)."""

    documents: list[Document]
    chunks: list[Chunk]
    embedder: Embedder
    retriever: Retriever
    top_k: int = DEFAULT_TOP_K

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        top_k: int = DEFAULT_TOP_K,
        embedding_backend: str | None = None,
        chunk_words: int = chunker.CHUNK_WORDS,
        overlap_words: int = chunker.OVERLAP_WORDS,
    ) -> "RAGService":
        started = time.perf_counter()
        documents = chunker.load_documents(data_dir)
        chunks = chunker.chunk_corpus(documents, target_words=chunk_words, overlap_words=overlap_words)

        # Build the embedder over the chunk texts (TF-IDF fallback fits on them).
        embedder = build_embedder([chunk.text for chunk in chunks], backend=embedding_backend)
        embeddings = embedder.encode([chunk.text for chunk in chunks])
        retriever = Retriever(chunks, embeddings)

        elapsed = (time.perf_counter() - started) * 1000
        LOGGER.info(
            "RAG service ready: %d docs, %d chunks, %d-dim %s embeddings, top_k=%d (%.0f ms)",
            len(documents),
            len(chunks),
            embedder.dimension,
            embedder.backend,
            top_k,
            elapsed,
        )
        return cls(documents=documents, chunks=chunks, embedder=embedder, retriever=retriever, top_k=top_k)

    # ----------------------------------------------------------------- query
    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        query_vector = self.embedder.encode(question)
        matches = self.retriever.search(query_vector, top_k or self.top_k)
        # Drop exact-zero matches: with the TF-IDF fallback these are documents
        # that share no vocabulary at all with the question, and citing them
        # would be noise. Dense embeddings essentially never score exactly 0.
        return [match for match in matches if match.score > 0.0]

    def answer(self, question: str, top_k: int | None = None) -> dict:
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty")

        started = time.perf_counter()
        retrieved = self.retrieve(question, top_k)

        if retrieved:
            generation = llm.generate_answer(question, retrieved)
            verdict = check_answer(generation.text, [item.chunk.text for item in retrieved])
        else:
            # Nothing cleared the similarity floor (only possible with the TF-IDF
            # fallback, where a fully out-of-vocabulary question yields a zero
            # vector). Refuse rather than send the model ungrounded context.
            generation = llm.AnswerResult(
                text=llm.NO_ANSWER_SENTENCE,
                status="skipped_no_context",
                error="No document chunk matched the question",
                grounded=False,
            )
            verdict = check_answer(generation.text, [])

        if generation.status != "ok" and not verdict.get("note"):
            verdict["note"] = (
                "Answer is a deterministic fallback, not a model completion "
                f"(llm_status={generation.status})."
            )

        latency_ms = round((time.perf_counter() - started) * 1000, 1)

        return {
            "question": question,
            "answer": generation.text,
            "citations": [
                {
                    "source": item.chunk.source_file,
                    "chunk_id": item.chunk.chunk_id,
                    "heading": item.chunk.heading_context,
                    "relevance": round(float(item.score), 4),
                }
                for item in retrieved
            ],
            "retrieved_chunks": [item.to_dict() for item in retrieved],
            "hallucination_check": verdict["status"],
            "hallucination_details": verdict,
            "model": generation.model,
            "embedding_model": self.embedder.display_name,
            # --- transparency metadata (extra, non-breaking) -------------------
            "llm_status": generation.status,
            "llm_error": generation.error,
            "embedding_backend": self.embedder.backend,
            "top_k": len(retrieved),
            "latency_ms": latency_ms,
        }

    # ---------------------------------------------------------------- status
    def health(self) -> dict:
        return {
            "status": "ok",
            "corpus": chunker.corpus_stats(self.documents, self.chunks),
            **self.embedder.describe(),
            "llm_model": llm.MODEL_NAME,
            "llm_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
            "top_k": self.top_k,
        }
