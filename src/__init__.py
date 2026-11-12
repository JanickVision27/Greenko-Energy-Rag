"""Energy Operations RAG Assistant — core pipeline package.

Modules
-------
chunker             heading-aware word-window chunking with overlap
embedder            sentence-transformers (all-MiniLM-L6-v2) with offline fallback
retriever           numpy cosine-similarity top-k search
llm                 Groq chat completion + deterministic fallback
hallucination_check content-word support check over the retrieved context
pipeline            the RAGService that wires the above together
"""

__all__ = ["chunker", "embedder", "retriever", "llm", "hallucination_check", "pipeline"]

__version__ = "1.0.0"
