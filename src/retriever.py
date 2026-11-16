"""Top-k retrieval with numpy cosine similarity.

No FAISS, no ChromaDB: the corpus is a handful of chunks, so a normalised matrix
multiply plus ``argsort`` is both simpler and faster than any index, and it has
zero native dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .chunker import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict:
        """Serialize a result, keeping its source, text, and similarity score."""
        return {
            "source": self.chunk.source_file,
            "chunk_id": self.chunk.chunk_id,
            "heading": self.chunk.heading_context,
            "text": self.chunk.text,
            "score": round(float(self.score), 4),
            "relevance": round(float(self.score), 4),
        }


class Retriever:
    """In-memory cosine-similarity index over pre-embedded chunks."""

    def __init__(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        """Validate and cache normalized document vectors for repeated searches."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk/embedding count mismatch: {len(chunks)} chunks vs {len(embeddings)} vectors"
            )
        if len(chunks) == 0:
            raise ValueError("Retriever requires at least one chunk")

        self.chunks = list(chunks)
        # Float32 keeps the index compact and is sufficient for similarity ranking.
        matrix = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        # Store unit vectors once; query vectors are unit-length too, so
        # cosine similarity collapses to a dot product.
        self.matrix = matrix / norms

    def __len__(self) -> int:
        """Return the number of indexed chunks (enables len(retriever))."""
        return len(self.chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list[RetrievedChunk]:
        """Rank chunks against one query vector and return the best matches."""
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                f"Query dim {query.shape[0]} does not match index dim {self.matrix.shape[1]}"
            )

        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            # Out-of-vocabulary query (TF-IDF fallback): nothing to rank on.
            return []

        # Both sides are unit vectors, so dot products are cosine similarities.
        similarities = self.matrix @ (query / norm)
        k = max(1, min(int(top_k), len(self.chunks)))
        # Sort by descending score, then source and chunk ID to break ties consistently.
        order = sorted(
            range(len(self.chunks)),
            key=lambda i: (-float(similarities[i]), self.chunks[i].source_file, self.chunks[i].chunk_id),
        )[:k]
        return [RetrievedChunk(chunk=self.chunks[i], score=float(similarities[i])) for i in order]
