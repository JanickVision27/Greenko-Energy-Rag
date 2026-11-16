"""Cheap, deterministic hallucination check (no second LLM call).

Method
------
1. Split the answer into sentences and drop citation markers (``[...]``) plus the
   boilerplate a fallback answer adds, so only *claims* are scored.
2. Extract content words per sentence: tokens of 3+ letters that are not
   stopwords, plus every number. Words are crudely normalised (lowercase,
   possessive/plural/``-ed``/``-ing`` stripped) so ``derating`` matches
   ``derated`` and ``checks`` matches ``check``.
3. A content word is *supported* when the same normalised form appears in the
   retrieved context (the exact excerpts the model was shown).
4. Coverage = supported / total content words over the whole answer.
   ``>= 0.70`` -> PASS, otherwise WARNING. Per-sentence detail is returned so a
   reader can see which sentence is weak, not just that something is.

Limitations (documented on purpose): lexical overlap is a proxy, not entailment.
A true statement paraphrased with unseen vocabulary can be flagged, and a
sentence that reuses document words while inverting their meaning can pass.
It is a fast guardrail for a 4-document SOP corpus, not a factuality model.
"""

from __future__ import annotations

import re
from typing import Sequence

THRESHOLD = 0.70

STATUS_PASS = "PASS"
STATUS_WARNING = "WARNING: Some claims may not be supported by retrieved documents"

_CITATION = re.compile(r"\[[^\]]*\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z\-']+|\d+(?:\.\d+)?[A-Za-z]?")
_FALLBACK_PREFIX = re.compile(r"^LLM (?:not configured|call failed \(.*\))\.\s*", re.IGNORECASE)
_CONTEXT_LABEL = re.compile(r"^Retrieved context:\s*", re.IGNORECASE)
_REFUSAL = "cannot find this in the provided documents"

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "you",
    "your", "yours", "based", "above", "using", "used", "use", "per", "via", "may", "must",
    "should", "shall", "recommended", "provided", "following", "answer", "question", "context",
}


def _stem(word: str) -> str:
    """Crude, dependency-free normalisation (good enough for lexical overlap)."""

    word = word.lower().strip("-'")
    # This is lightweight normalization, not full linguistic stemming.
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    for suffix in ("ing", "ed"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word.endswith("es") and not word.endswith("ses"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def content_words(text: str) -> list[str]:
    """Return normalised content words (stopwords and citations removed)."""

    # Ignore citation labels so filenames and chunk numbers do not inflate support.
    lowered = _CITATION.sub(" ", text or "")
    words: list[str] = []
    for raw in _TOKEN.findall(lowered):
        if raw[0].isdigit():
            words.append(_stem(raw))
            continue
        if len(raw) < 3 or raw.lower() in STOPWORDS:
            continue
        words.append(_stem(raw))
    return words


def _strip_boilerplate(answer: str) -> str:
    """Remove fallback labels so the support score covers the actual answer."""
    text = (answer or "").strip()
    text = _FALLBACK_PREFIX.sub("", text)
    text = _CONTEXT_LABEL.sub("", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """Split answer text into sentences for per-sentence support details."""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def check_answer(
    answer: str,
    context_texts: Sequence[str],
    threshold: float = THRESHOLD,
) -> dict:
    """Score how well ``answer`` is supported by ``context_texts``."""

    # A set makes lookups fast and avoids counting repeated context words specially.
    context_vocab = {word for text in context_texts for word in content_words(text)}
    claim_text = _strip_boilerplate(answer or "")

    # A canonical refusal makes no factual claim, so it cannot hallucinate.
    if _REFUSAL in claim_text.lower() and len(content_words(_CITATION.sub(" ", claim_text))) <= 12:
        return {
            "status": STATUS_PASS,
            "label": STATUS_PASS,
            "coverage": 1.0,
            "threshold": threshold,
            "content_words": 0,
            "matched_words": 0,
            "unsupported_words": [],
            "sentences": [],
            "note": "Model declined to answer (no claims made).",
        }

    answer_words = content_words(claim_text)
    if not answer_words:
        return {
            "status": STATUS_PASS,
            "label": STATUS_PASS,
            "coverage": 1.0,
            "threshold": threshold,
            "content_words": 0,
            "matched_words": 0,
            "unsupported_words": [],
            "sentences": [],
            "note": "No content words to verify.",
        }

    # Coverage counts repeated answer terms; unsupported_words is a unique review list.
    supported = [word for word in answer_words if word in context_vocab]
    unsupported = sorted({word for word in answer_words if word not in context_vocab})
    coverage = len(supported) / len(answer_words)

    sentences = []
    for sentence in _split_sentences(_strip_boilerplate(claim_text)):
        words = content_words(sentence)
        if not words:
            continue
        matched = sum(1 for word in words if word in context_vocab)
        ratio = matched / len(words)
        sentences.append(
            {
                "text": sentence,
                "coverage": round(ratio, 3),
                "supported": ratio >= threshold,
                "unsupported_words": sorted({w for w in words if w not in context_vocab}),
            }
        )

    passed = coverage >= threshold
    return {
        "status": STATUS_PASS if passed else STATUS_WARNING,
        "label": STATUS_PASS if passed else "WARNING",
        "coverage": round(coverage, 3),
        "threshold": threshold,
        "content_words": len(answer_words),
        "matched_words": len(supported),
        "unsupported_words": unsupported,
        "sentences": sentences,
        "note": None,
    }
