"""FastAPI service for the Energy Operations Document RAG Assistant.

Startup (lifespan) loads the four SOP documents, chunks them, embeds every chunk
and builds the in-memory cosine-similarity index once. Each ``/api/run`` call is
then: embed query -> top-k retrieval -> grounded Groq answer -> hallucination
check -> JSON.

The same frontend is served by this app at ``/`` (FastAPI StaticFiles), so there is
exactly one deployable on Railway and no CORS/absolute-URL problems.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = REPO_ROOT / "frontend"

# Load .env once, before anything reads GROQ_API_KEY (never hard-coded).
load_dotenv(REPO_ROOT / ".env", override=True)

from src.llm import MODEL_NAME as LLM_MODEL  # noqa: E402
from src.pipeline import DEFAULT_DATA_DIR, RAGService  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("energy_rag")

SERVICE: RAGService | None = None
# FastAPI startup fills this shared service before request handlers use it.


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the RAG pipeline once, at startup."""

    global SERVICE
    LOGGER.info("Starting up: loading corpus from %s", DEFAULT_DATA_DIR)
    try:
        SERVICE = RAGService.build(DEFAULT_DATA_DIR)
    except Exception:
        LOGGER.exception("Failed to initialise the RAG pipeline")
        raise
    LOGGER.info("Startup complete: %d chunks indexed", len(SERVICE.chunks))
    yield
    SERVICE = None
    LOGGER.info("Shutdown complete")


app = FastAPI(
    title="Energy Operations Document RAG Assistant",
    description="Cited, grounded answers over renewable-energy operations, safety and maintenance SOPs.",
    version="1.0.0",
    lifespan=lifespan,
)


class Query(BaseModel):
    """JSON request schema for POST /api/run."""
    text: str = Field(default="", description="Operational question to answer from the SOP corpus")


@app.get("/api/health", tags=["ops"])
def health() -> JSONResponse:
    """Readiness probe: corpus size, embedding backend, whether the LLM is configured."""

    # 503 tells a browser or deployment platform that startup is not complete.
    if SERVICE is None:
        return JSONResponse(status_code=503, content={"status": "starting"})
    return JSONResponse(status_code=200, content=SERVICE.health())


@app.get("/api/demo", tags=["ops"])
def demo() -> dict:
    """Kept from the starter app; now reports the live pipeline configuration."""

    if SERVICE is None:
        raise HTTPException(status_code=503, detail="Service is still starting")
    health_payload = SERVICE.health()
    return {
        "case_study": "Energy Operations Document RAG Assistant",
        "status": "ready",
        "expected_outcome": (
            "A working RAG-based assistant that answers operational questions from provided "
            "documents with citations, retrieval evidence, and hallucination checks."
        ),
        "corpus": health_payload["corpus"],
        "embedding_model": health_payload["embedding_model"],
        "embedding_backend": health_payload["embedding_backend"],
        "llm_model": LLM_MODEL,
        "llm_configured": health_payload["llm_configured"],
        "top_k": health_payload["top_k"],
        "endpoints": {
            "POST /api/run": "ask a question; returns answer, citations, retrieved chunks, hallucination check",
            "GET /api/demo": "this configuration summary",
            "GET /api/health": "readiness probe incl. corpus statistics",
            "GET /docs": "interactive OpenAPI documentation",
        },
    }


@app.post("/api/run", tags=["rag"])
def run_case(query: Query) -> dict:
    """Answer a question with citations, retrieval evidence and a hallucination check."""

    if SERVICE is None:
        raise HTTPException(status_code=503, detail="Service is still starting")
    # Strip whitespace before validation so a spaces-only value is rejected.
    question = (query.text or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="Field 'text' must contain a question")
    if len(question) > 1000:
        raise HTTPException(status_code=422, detail="Question is too long (max 1000 characters)")

    # Log internal details, but return a concise error instead of a stack trace.
    try:
        response = SERVICE.answer(question)
    except Exception as exc:  # surface a clean error instead of a stack trace to the UI
        LOGGER.exception("RAG pipeline failed for question: %s", question)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {type(exc).__name__}") from exc

    LOGGER.info(
        "answered q=%r top=%s status=%s latency=%.0fms",
        question[:80],
        response["citations"][0]["source"] if response["citations"] else "none",
        response["llm_status"],
        response["latency_ms"],
    )
    return response


# Serve the single-page frontend from the same origin. Mounted last so /api/* wins.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    LOGGER.info("Serving frontend from %s", FRONTEND_DIR)
