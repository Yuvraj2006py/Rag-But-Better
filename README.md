# RAG but Better
## https://rag-but-better-1.onrender.com

Build a local semantic search + QA system over internal documents with proper chunking, hybrid retrieval (dense + sparse), reranking, citations, evaluation, and a feedback loop. Groq is used for query rewriting and answer generation so you do not need to run a local LLM.

## What this includes
- Smart semantic chunking (not naive fixed-size splits)
- Hybrid retrieval (dense + sparse + reranking)
- Query rewrite/expansion (Groq)
- Citation tracking per answer
- Answer quality scoring + hallucination detection
- User feedback storage
- A/B testing framework for retrieval settings
- Local React UI + FastAPI backend

## Architecture (high level)
- **React UI**: upload docs, ask questions, view citations/scores
- **FastAPI API**: ingestion, search, evaluation, feedback, experiments
- **Vector store**: Qdrant (Docker)
- **Sparse store**: OpenSearch (Docker)
- **LLM**: Groq API for rewrite + answer

## Quickstart

### Prereqs
- Python 3.11+ (recommended)
- Node.js 20+
- Docker Desktop (WSL2 backend enabled)
- Groq API key

### 1) Environment
Copy `.env.example` to `.env` and set your Groq key:

```
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

### 2) Start infrastructure (Qdrant + OpenSearch)
From repo root:

```bash
docker compose -f infra/docker-compose.yml up -d
```

Verify:
- Qdrant: http://localhost:6333
- OpenSearch: http://localhost:9200

### 3) Backend (FastAPI)
```bash
cd apps/api
python -m venv .venv
.venv/Scripts/activate
pip install -e .[test]
pytest
uvicorn app.main:app --reload --port 8000
```

### 4) Frontend (React)
```bash
cd apps/web
npm install
npm test
npm run dev
```

Open the UI at: http://localhost:5173

## External stores (optional toggle)
By default, the app uses in-memory stores to keep local dev simple. To use Qdrant/OpenSearch:

```
USE_EXTERNAL_STORES=true
QDRANT_URL=http://localhost:6333
OPENSEARCH_URL=http://localhost:9200
```

## Repository structure
- `apps/api` — FastAPI backend
- `apps/web` — React UI
- `infra` — Docker compose for Qdrant + OpenSearch
- `docs` — PLAN + PRD
- `data` — uploaded files (ignored by git)

## Notes & troubleshooting
- If `uvicorn` fails with missing deps, run `pip install -e .[test]` again.
- If OpenSearch fails to start, ensure Docker has enough memory (>= 4GB).
- On Windows, always run `npm` and `uvicorn` from their app directories.

## Security
- `.env` is ignored by git; only `.env.example` is tracked.
- Do not commit real API keys or internal documents.
