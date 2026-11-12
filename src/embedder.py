"""Embeddings for the RAG pipeline.

Primary backend
---------------
``sentence-transformers`` with ``all-MiniLM-L6-v2`` (384-dim, CPU-friendly,
~80 MB). Vectors are L2-normalised so cosine similarity is a plain dot product.

Offline / constrained fallback
------------------------------
Downloading model weights needs huggingface.co. Builds and sandboxes that cannot
reach it (air-gapped CI, a locked-down review environment, a 512 MB container)
would otherwise fail at startup for no good reason. When sentence-transformers
is not installed, or the weights cannot be loaded, the embedder degrades to a
deterministic TF-IDF vectoriser (word unigrams + bigrams, sublinear TF,
L2-normalised) fitted on the chunk corpus. Retrieval quality on short, keyword-rich
SOP text stays workable and the whole pipeline remains runnable — the downgrade is
reported in every API response via ``embedding_backend`` instead of being hidden.

Select with the ``EMBEDDING_BACKEND`` environment variable: ``auto`` (default),
``sentence-transformers``, or ``tfidf``.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
TFIDF_BACKEND = "tfidf-fallback"


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


class Embedder:
    """Turns text into L2-normalised vectors with a documented fallback path."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, backend: str = "auto") -> None:
        self.requested_backend = (backend or "auto").strip().lower()
        self.model_name = model_name
        self.backend = "uninitialised"
        self.fallback_reason: str | None = None
        self._model = None
        self._vectorizer = None

    # ------------------------------------------------------------------ setup
    def fit(self, corpus: Sequence[str]) -> "Embedder":
        """Initialise the backend. ``corpus`` is only used by the TF-IDF fallback."""

        corpus = [text for text in corpus if text and text.strip()]
        if not corpus:
            raise ValueError("Cannot fit an embedder on an empty corpus")

        if self.requested_backend in {"auto", "sentence-transformers", "st"}:
            if self._try_sentence_transformers():
                return self

        if self.requested_backend == "sentence-transformers":
            # Explicitly requested: fail loudly instead of silently downgrading.
            raise RuntimeError(
                f"sentence-transformers backend requested but unavailable: {self.fallback_reason}"
            )

        self._fit_tfidf(corpus)
        return self

    def _try_sentence_transformers(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # ImportError, or a broken torch install
            self.fallback_reason = f"sentence-transformers not importable ({type(exc).__name__}: {exc})"
            LOGGER.warning("Embedding fallback: %s", self.fallback_reason)
            return False

        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:  # offline, no weights cached, OOM, ...
            self.fallback_reason = f"could not load '{self.model_name}' ({type(exc).__name__}: {exc})"
            LOGGER.warning("Embedding fallback: %s", self.fallback_reason)
            return False

        self.backend = "sentence-transformers"
        LOGGER.info("Embeddings: sentence-transformers/%s", self.model_name)
        return True

    def _fit_tfidf(self, corpus: Sequence[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            # English stopwords matter more than they look: without them a short
            # query is dominated by words like "be"/"should" that appear in every
            # SOP, which silently demotes the correct document.
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b",
        )
        self._vectorizer.fit(corpus)
        self.backend = TFIDF_BACKEND
        LOGGER.warning(
            "Embeddings: deterministic TF-IDF fallback (%s vocabulary terms). "
            "Install sentence-transformers and allow huggingface.co access for all-MiniLM-L6-v2.",
            len(self._vectorizer.vocabulary_),
        )

    # ------------------------------------------------------------------ encode
    def encode(self, texts: Sequence[str] | str) -> np.ndarray:
        """Return an ``(n, d)`` float32 matrix of unit-length vectors."""

        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        if not batch:
            return np.zeros((0, 1), dtype=np.float32)

        if self.backend == "sentence-transformers":
            vectors = self._model.encode(  # type: ignore[union-attr]
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            matrix = np.asarray(vectors, dtype=np.float32)
        elif self.backend == TFIDF_BACKEND:
            sparse = self._vectorizer.transform(batch)  # type: ignore[union-attr]
            matrix = np.asarray(sparse.todense(), dtype=np.float32)
            matrix = _l2_normalize(matrix)
        else:
            raise RuntimeError("Embedder.encode() called before fit()")

        return matrix[0] if single else matrix

    # ------------------------------------------------------------- reporting
    @property
    def dimension(self) -> int:
        if self.backend == "sentence-transformers":
            return int(self._model.get_sentence_embedding_dimension())  # type: ignore[union-attr]
        if self.backend == TFIDF_BACKEND:
            return len(self._vectorizer.vocabulary_)  # type: ignore[union-attr]
        return 0

    @property
    def display_name(self) -> str:
        if self.backend == "sentence-transformers":
            return self.model_name
        return TFIDF_BACKEND

    def describe(self) -> dict:
        return {
            "embedding_model": self.display_name,
            "embedding_backend": self.backend,
            "embedding_dim": self.dimension,
            "embedding_fallback_reason": self.fallback_reason,
            "is_fallback": self.backend != "sentence-transformers",
        }


def build_embedder(
    corpus: Sequence[str],
    model_name: str | None = None,
    backend: str | None = None,
) -> Embedder:
    """Convenience factory honouring the ``EMBEDDING_BACKEND`` env var."""

    selected = backend or os.getenv("EMBEDDING_BACKEND", "auto")
    embedder = Embedder(
        model_name=model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME),
        backend=selected,
    )
    return embedder.fit(corpus)
