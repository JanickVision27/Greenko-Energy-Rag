"""Grounded answer generation with Groq (``openai/gpt-oss-120b``).

The prompt is deliberately strict: answer only from the excerpts, cite
``[source_file, chunk_id]`` after every factual claim, refuse when the excerpts do
not contain the answer. The API key is read from the environment at call time
(never hard-coded, never logged).

Failure handling is explicit rather than silent:
``ok``              a real model answer
``not_configured``  no ``GROQ_API_KEY`` -> deterministic, context-only fallback
``error``           the API call failed (network, auth, rate limit) -> same fallback
                    with the reason in ``error``
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Sequence

from .retriever import RetrievedChunk

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "openai/gpt-oss-120b"
TEMPERATURE = 0.2
MAX_TOKENS = 500
REQUEST_TIMEOUT_SECONDS = 30.0

SYSTEM_PROMPT = """You are an energy operations assistant. You answer questions ONLY using the provided document excerpts.
You must cite your source as [source_file, chunk_id] after every factual claim.
If the answer is not in the provided excerpts, say "I cannot find this in the provided documents."
Do not use outside knowledge. Do not invent procedures."""

NO_ANSWER_SENTENCE = "I cannot find this in the provided documents."


@dataclass
class AnswerResult:
    """Keep model output and its status together, including fallback cases."""
    text: str
    model: str = MODEL_NAME
    status: str = "ok"  # ok | not_configured | error
    error: str | None = None
    grounded: bool = True  # False when the text is a fallback, not a model answer
    usage: dict = field(default_factory=dict)

    @property
    def is_fallback(self) -> bool:
        """True when the answer did not come from a successful model response."""
        return self.status != "ok"


def format_context(retrieved: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks in the numbered, citable block the prompt expects."""

    # Numbered blocks make passages easy for the model to reference in citations.
    blocks = []
    for rank, item in enumerate(retrieved, start=1):
        blocks.append(
            f'[{rank}] ({item.chunk.source_file}, chunk {item.chunk.chunk_id}, '
            f'relevance: {item.score:.3f})\n"{item.chunk.text}"'
        )
    return "\n\n".join(blocks)


def build_messages(question: str, retrieved: Sequence[RetrievedChunk]) -> list[dict]:
    """Build chat messages that constrain the model to retrieved evidence."""
    user_prompt = (
        f"Question: {question.strip()}\n\n"
        f"Retrieved context:\n{format_context(retrieved)}\n\n"
        "Answer the question using ONLY the above context. Cite sources."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def fallback_text(retrieved: Sequence[RetrievedChunk]) -> str:
    """Return the best retrieved passage when model generation is unavailable."""
    top = retrieved[0].chunk.text if retrieved else "No chunks were retrieved."
    return f"LLM not configured. Retrieved context: {top}"


def _error_fallback(reason: str, retrieved: Sequence[RetrievedChunk]) -> AnswerResult:
    """Package a model failure with retrieved text so the UI can still respond."""
    top = retrieved[0].chunk.text if retrieved else "No chunks were retrieved."
    return AnswerResult(
        text=f"LLM call failed ({reason}). Retrieved context: {top}",
        status="error",
        error=reason,
        grounded=False,
    )


def generate_answer(
    question: str,
    retrieved: Sequence[RetrievedChunk],
    api_key: str | None = None,
    model: str = MODEL_NAME,
) -> AnswerResult:
    """Call Groq, or return a deterministic fallback when that is impossible."""

    # Resolve secrets at call time so credentials are never stored in source code.
    key = (api_key if api_key is not None else os.getenv("GROQ_API_KEY", "")).strip()
    if not key:
        LOGGER.warning("GROQ_API_KEY is not set - returning context-only fallback answer")
        return AnswerResult(
            text=fallback_text(retrieved),
            status="not_configured",
            error="GROQ_API_KEY is not set",
            grounded=False,
        )

    try:
        from groq import Groq  # imported lazily so the API can start without it
    except Exception as exc:  # pragma: no cover - depends on install
        return _error_fallback(f"groq SDK unavailable: {type(exc).__name__}", retrieved)

    try:
        client = Groq(api_key=key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=1)
        completion = client.chat.completions.create(
            model=model,
            messages=build_messages(question, retrieved),
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        choice = completion.choices[0]
        text = (choice.message.content or "").strip()
        if not text:
            return _error_fallback("empty response from model", retrieved)

        usage = {}
        if getattr(completion, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
                "completion_tokens": getattr(completion.usage, "completion_tokens", None),
                "total_tokens": getattr(completion.usage, "total_tokens", None),
            }
        return AnswerResult(
            text=text,
            model=getattr(completion, "model", model) or model,
            status="ok",
            usage=usage,
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        LOGGER.error("Groq call failed: %s", reason)
        return _error_fallback(reason[:200], retrieved)
