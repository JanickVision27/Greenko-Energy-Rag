"""Heading-aware chunking for the energy-operations SOP corpus.

Design
------
* Markdown headings (``# Title``) are treated as hard natural breaks: a chunk
  never spans two headings, and every chunk carries its heading as context.
* Inside a section the text is split on sentence boundaries and greedily packed
  into windows of ``CHUNK_WORDS`` words (default 250).
* Consecutive windows overlap by up to ``OVERLAP_WORDS`` words (default 25) so a
  procedure that is split across a boundary stays retrievable.
* Sentences longer than the window are hard-split (safety valve for tables /
  run-on text), and a trailing fragment shorter than ``MIN_CHUNK_WORDS`` is
  merged back into the previous chunk instead of becoming a useless 3-word chunk.
* Chunk ids are per-document and zero-based; ``heading_context`` is the nearest
  heading, which is what the UI shows on a citation chip.

On the bundled corpus (four documents, ~30-40 words each) every section fits in a
single window, so each file yields exactly one chunk. That is the correct
behaviour for a 4-document toy corpus, not a bug — the windowing logic only
starts splitting once a section exceeds 250 words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CHUNK_WORDS = 250
OVERLAP_WORDS = 25
MIN_CHUNK_WORDS = 25

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(r"^(#{1,6})\s+(?P<title>.+?)\s*$")


@dataclass(frozen=True)
class Document:
    """A raw source document loaded from ``data/``."""

    source_file: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text plus the metadata needed for a citation."""

    source_file: str
    chunk_id: int
    heading_context: str
    text: str
    n_words: int

    @property
    def citation(self) -> str:
        """Return the compact citation label used in generated answers."""
        return f"[{self.source_file}, chunk {self.chunk_id}]"

    def to_dict(self) -> dict:
        """Convert the chunk and its metadata to a JSON-friendly dictionary."""
        return {
            "source": self.source_file,
            "chunk_id": self.chunk_id,
            "heading": self.heading_context,
            "text": self.text,
            "n_words": self.n_words,
        }


def load_documents(data_dir: str | Path) -> list[Document]:
    """Load every ``*.md`` file in ``data_dir`` (sorted for determinism)."""

    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    documents: list[Document] = []
    for path in sorted(data_path.glob("*.md")):
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        title = _first_heading(raw) or path.stem.replace("_", " ").title()
        documents.append(Document(source_file=path.name, title=title, text=raw))

    if not documents:
        raise FileNotFoundError(f"No markdown documents found in {data_path}")
    return documents


def _first_heading(text: str) -> str | None:
    """Return the first Markdown heading, or None when the file has none."""
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            return match.group("title")
    return None


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into ``(heading, body)`` pairs.

    Text before the first heading is kept with an empty heading so nothing is
    silently dropped.
    """

    sections: list[tuple[str, str]] = []
    heading = ""
    body: list[str] = []

    def flush() -> None:
        """Append the accumulated body under its current heading, if nonempty."""
        block = "\n".join(body).strip()
        if block:
            sections.append((heading, block))

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            heading = match.group("title")
            body = []
        else:
            body.append(line)
    flush()
    return sections


def _split_sentences(block: str) -> list[str]:
    """Normalize whitespace and split a section body into sentences."""
    cleaned = re.sub(r"[ \t]+", " ", block)
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(cleaned) if p.strip()]
    return parts or [cleaned.strip()]


def _pack_sentences(
    sentences: Iterable[str],
    target_words: int = CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[str]:
    """Greedily pack sentences into windows of ~``target_words`` with overlap."""

    windows: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        words = sentence.split()
        n_words = len(words)

        # Oversized single sentence: hard-split into overlapping word windows.
        if n_words > target_words:
            if current:
                windows.append(" ".join(current))
                current, current_words = [], 0
            step = max(1, target_words - overlap_words)
            for start in range(0, n_words, step):
                piece = " ".join(words[start : start + target_words])
                if piece:
                    windows.append(piece)
                if start + target_words >= n_words:
                    break
            continue

        if current and current_words + n_words > target_words:
            windows.append(" ".join(current))
            tail: list[str] = []
            tail_words = 0
            for previous in reversed(current):
                previous_words = len(previous.split())
                if tail_words + previous_words > overlap_words:
                    break
                tail.insert(0, previous)
                tail_words += previous_words
            current, current_words = tail, tail_words

        current.append(sentence)
        current_words += n_words

    if current:
        text = " ".join(current)
        # Merge a stub tail into the previous window rather than emit a fragment.
        if windows and current_words < MIN_CHUNK_WORDS:
            windows[-1] = f"{windows[-1]} {text}".strip()
        else:
            windows.append(text)
    return [w for w in windows if w.strip()]


def chunk_document(
    document: Document,
    target_words: int = CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[Chunk]:
    """Chunk one document, keeping headings as boundaries and context."""

    chunks: list[Chunk] = []
    for heading, body in split_sections(document.text):
        for text in _pack_sentences(_split_sentences(body), target_words, overlap_words):
            body_text = text.strip()
            if not body_text:
                continue
            # The heading leads the chunk text so the embedded string and the
            # text shown in the UI both carry topical context.
            full_text = f"{heading}\n{body_text}" if heading else body_text
            chunks.append(
                Chunk(
                    source_file=document.source_file,
                    chunk_id=len(chunks),
                    heading_context=heading or document.title,
                    text=full_text,
                    n_words=len(full_text.split()),
                )
            )

    if not chunks:
        raise ValueError(f"Document produced no chunks: {document.source_file}")
    return chunks


def chunk_corpus( documents: Iterable[Document], target_words: int = CHUNK_WORDS, overlap_words: int = OVERLAP_WORDS,) -> list[Chunk]:
    """Chunk every document and return a flat, deterministic chunk list."""

    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, target_words, overlap_words))
    return chunks


def corpus_stats(documents: Iterable[Document], chunks: Iterable[Chunk]) -> dict:
    """Small summary used by ``/api/health`` and the evaluation report."""

    # Materialize once because inputs may be generators and are used more than once.
    docs = list(documents)
    chs = list(chunks)
    per_source: dict[str, int] = {}
    for chunk in chs:
        per_source[chunk.source_file] = per_source.get(chunk.source_file, 0) + 1
    return {
        "documents": len(docs),
        "chunks": len(chs),
        "words": sum(len(d.text.split()) for d in docs),
        "chunks_per_source": per_source,
        "chunk_words_target": CHUNK_WORDS,
        "chunk_words_overlap": OVERLAP_WORDS,
    }
