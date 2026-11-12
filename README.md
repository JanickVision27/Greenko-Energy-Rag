# Energy Operations RAG Assistant

A small retrieval-augmented question-answering app for renewable-energy operations documents. A user asks a question; the app retrieves relevant passages from four bundled Markdown files and, when configured, asks a Groq-hosted language model to answer using those passages. The response includes citations, retrieved evidence, and a deterministic text-overlap support check.

This project is a **document Q&A demonstration**, not a renewable-generation forecasting model. The included corpus and evaluation set are intentionally small and do not establish production accuracy.

## What the app does

```text
Markdown SOPs -> heading-aware chunks -> text vectors -> cosine top-k retrieval
      question ------------------------------------------------------┘
retrieved passages -> Groq LLM (if configured) -> support check -> JSON/UI
```

At startup, the service loads and chunks the documents, builds the vectors, and creates an in-memory retrieval index. On each request it embeds the question, returns up to three matching chunks, and passes those chunks to the LLM. If no chunk matches, it refuses rather than sending unsupported context to the model.

| Component | Implementation |
|---|---|
| Corpus | Four renewable-energy operations Markdown documents in `data/` |
| Chunking | Heading-aware windows; target 250 words with 25-word overlap |
| Embeddings | `all-MiniLM-L6-v2` when available; TF-IDF fallback |
| Retrieval | In-memory NumPy cosine similarity; top-k defaults to 3 |
| Answer generation | Groq Chat Completions; code default is `openai/gpt-oss-120b` |
| Support check | Deterministic content-word overlap with retrieved passages; 70% threshold |
| API and UI | FastAPI backend and a single-page HTML frontend served by the same app |

The current four source files are shorter than the chunk target, so each produces one chunk. Consequently, this demo mostly retrieves among whole documents; it does not yet demonstrate chunk-level search over long manuals.

## Repository map

```text
backend/app.py                 FastAPI startup, endpoints, and frontend serving
src/chunker.py                 Markdown loading, chunking, and corpus statistics
src/embedder.py                Sentence-transformer embeddings and TF-IDF fallback
src/retriever.py               In-memory cosine-similarity search
src/llm.py                     Groq prompt, model call, and context-only fallbacks
src/hallucination_check.py     Lexical support check
src/pipeline.py                End-to-end RAG service used by API and evaluator
scripts/evaluate.py            Runs the labeled questions through the RAG service
data/                           Four source documents and questions.csv
outputs/evaluation_results.json Saved evaluation report
frontend/index.html             Browser UI
```

## Setup

Requires Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

The full requirements install `sentence-transformers` and PyTorch and may take time and disk space. For a lighter install without PyTorch, use:

```bash
python -m pip install -r requirements-minimal.txt
```

The embedder then uses TF-IDF. You can also select the backend with `EMBEDDING_BACKEND=auto`, `sentence-transformers`, or `tfidf`; `auto` is the default. The sentence-transformer backend may need to download its model weights the first time it runs.

### Configure the LLM (optional)

Copy `.env.example` to `.env` and set your Groq key in `.env`:

```dotenv
GROQ_API_KEY=your_key_here
```

Keep the key private; `.env` is git-ignored. Without a key, the application still starts and retrieves evidence, but returns a clearly labeled context-only fallback instead of an LLM-generated answer. A failed Groq request is also surfaced in the response status and falls back to retrieved context.

The model name above describes the **current code default** in `src/llm.py`. Older saved evaluation output or UI copy may name a different model; use the live API response and current source code when describing an actual run.

## Run the application

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. The API documentation is at <http://localhost:8000/docs>.

| Endpoint | Purpose |
|---|---|
| `POST /api/run` | Ask a question; returns answer, citations, retrieved chunks, support check, and runtime metadata |
| `GET /api/health` | Reports readiness, corpus statistics, embedding backend, and whether an LLM key is configured |
| `GET /api/demo` | Reports the live pipeline configuration and available endpoints |
| `GET /` | Serves the web interface |

Example request:

```bash
curl -X POST http://localhost:8000/api/run \
  -H "Content-Type: application/json" \
  -d '{"text":"What should be checked for inverter derating?"}'
```

The response includes fields such as `answer`, `citations`, `retrieved_chunks`, `hallucination_check`, `llm_status`, `embedding_backend`, and `latency_ms`. The `llm_status` field distinguishes generated answers (`ok`) from fallbacks (`not_configured` or `error`) and refusals (`skipped_no_context`).

## Evaluation

Run the included evaluation against the same `RAGService` used by the API:

```bash
python -m scripts.evaluate
```

This reads `data/questions.csv` and writes `outputs/evaluation_results.json`. The committed report records **2 correct top-source matches out of 2 questions (100% source attribution)**. In that saved run, the LLM key was absent, the embedding backend was TF-IDF, and **zero LLM answers were generated**. The reported answers were deterministic context-only fallbacks.

Interpret this as a small retrieval smoke test, not a reliable estimate of general performance. Four documents and two questions are not enough to measure answer quality, recall across a realistic library, robustness to distractors, or performance under load. The support check is lexical overlap rather than a semantic entailment test: it can flag valid paraphrases and can miss a claim that reuses source words but changes their meaning.

## Design choices and limitations

- **Retrieval task:** Questions are matched to document chunks by cosine similarity. Sentence embeddings are intended to capture related meaning; the TF-IDF fallback is keyword-based and may miss synonyms.
- **Generation:** The prompt asks the model to use only retrieved excerpts, cite factual claims with `[source_file, chunk_id]`, and say when the documents do not contain the answer. A prompt is a safeguard, not a guarantee of correctness.
- **No agent workflow:** This is a single retrieval-and-generation pipeline with one LLM call, not an agent that selects among tools or takes actions.
- **In-memory, stateless service:** The index is rebuilt at startup. Each question is handled independently; there is no chat history, persistence, authentication, or document-level access control.
- **Small, fixed corpus:** There is no document-upload interface. Add or edit Markdown files in `data/` and restart the service to rebuild the index.
- **No forecast or variance analysis:** The grid operations note mentions generation deviation checks, but the application does not train a forecasting model or calculate forecast errors.

Useful next steps are to evaluate on a larger labeled question set, add longer and more varied documents, assess answer faithfulness separately from source retrieval, and replace or supplement the lexical support check with claim-level verification.

## Deployment

`Procfile` and `railway.json` provide a Railway start configuration. Deployment still requires configuring `GROQ_API_KEY` as a Railway environment variable if model-generated answers are desired. After deployment, check `/api/health` and submit a question through `/api/run` or the web interface. A deployment URL or live-service result is not included in the saved evaluation report.

## Safe handling

Treat the controlled operational documents as the source of truth for field work. This demo's retrieved text, generated answer, and lexical support badge should not replace approved procedures or qualified operational judgment.
