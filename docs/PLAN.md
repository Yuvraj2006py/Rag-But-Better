# Plan: NLP for Internal Company Documents (RAG but Better)

## Goals
- Build a local web UI (React) for semantic search + QA over internal docs.
- Use Groq for LLM tasks (query rewrite/answer); no local LLM.
- Provide robust retrieval, reranking, citations, evaluation, and feedback loops.
- Keep all infra free/self-hosted.

## Assumptions
- Windows host with Docker Desktop.
- Docs can be PDF, DOCX, HTML, TXT, Markdown, etc.
- Small-to-medium scale during development; no fixed corpus size.

## Phase 0 — Environment & Repo Setup (Detailed)
### 0.1 Prerequisites (your machine)
- Install Python 3.11+.
- Install Node.js 20+ (includes npm).
- Install Docker Desktop (WSL2 backend enabled).
- Create a Groq API key (free tier).

### 0.2 Local configuration (your steps)
- Create `.env` at repo root with:
  - `GROQ_API_KEY=your_key_here`
  - `GROQ_MODEL=llama-3.1-8b-instant` (or a Groq model you prefer)
- Ensure Docker is running before starting services.

### 0.3 Repo scaffolding (project layout)
- Establish directories:
  - `apps/web` (React UI)
  - `apps/api` (FastAPI)
  - `services/ingest` (parsers/chunking)
  - `services/retrieval` (hybrid search)
  - `services/eval` (RAGAS/TruLens)
  - `infra` (Docker compose for Qdrant + OpenSearch)
  - `data` (uploads)
  - `migrations` (SQLite → Postgres optional)

### 0.4 Initial infrastructure setup
- Add `docker-compose.yml` in `infra` for:
  - Qdrant
  - OpenSearch
- Add basic `README` instructions for running services.

## Phase 1 — Backend API (FastAPI)
- Endpoints:
  - `POST /documents/upload`
  - `GET /documents/status`
  - `POST /search/query`
  - `POST /feedback`
  - `GET /experiments/configs`
  - `POST /experiments/assign`
  - `POST /evaluation/run`
- Shared models:
  - `Document`, `Chunk`, `Citation`, `QueryTrace`, `Feedback`.

## Phase 2 — Ingestion + Normalization
- Parsers:
  - PDF: `pypdf`
  - DOCX: `python-docx`
  - HTML: `beautifulsoup4`
  - TXT/MD: builtin
- Normalize into a unified representation:
  - `doc_id`, `title`, `source_path`, `page`, `paragraph`, `text`, `metadata`
- Store raw text + doc metadata in SQLite.

## Phase 3 — Smart Semantic Chunking
- Sentence segmentation (`nltk` or `spacy` optional).
- Embed each sentence using `sentence-transformers`.
- Merge sentences until semantic similarity drops below threshold.
- Enforce min/max size constraints.
- Store:
  - `chunk_id`, `doc_id`, `start_para`, `end_para`, `text`, `embedding`.

## Phase 4 — Indexing
- Dense vectors: Qdrant (self-hosted via Docker).
- Sparse index: OpenSearch (self-hosted via Docker).
- Metadata: SQLite (upgrade path to Postgres).

## Phase 5 — Retrieval + Reranking
- Query rewrite/expansion via Groq.
- Dense retrieval (Qdrant top-K).
- Sparse retrieval (OpenSearch top-K).
- Merge results with Reciprocal Rank Fusion (RRF).
- Rerank with cross-encoder (sentence-transformers).

## Phase 6 — Answer Generation + Citations
- Build citation-aware prompt for Groq:
  - Answer only using provided chunks.
  - Provide citations per statement.
  - If insufficient evidence, respond with “Not enough information.”
- Return answer + chunk-level citations.
- Persist query trace for debugging and evaluation.

## Phase 7 — Answer Quality Scoring
- RAGAS metrics:
  - Faithfulness
  - Answer relevance
  - Context precision/recall
- Store per-query scores in SQLite.

## Phase 8 — Hallucination Detection
- Flag low faithfulness or unsupported statements.
- UI warning badge + confidence score.

## Phase 9 — Feedback Loop
- User thumbs up/down + optional notes.
- Store feedback linked to query and experiment config.

## Phase 10 — A/B Testing Framework
- Config objects:
  - Chunk size
  - Retrieval weights
  - Top-K values
  - Reranker on/off
- Assign configs per query.
- Compare metrics + feedback across configs.

## Phase 11 — Evaluation Suite
- Curated test queries + expected sources.
- Batch evaluation:
  - Retrieval recall@k
  - Rerank precision@k
  - RAGAS averages
- Export CSV reports.

## Deliverables
- React UI for upload/search/feedback.
- FastAPI backend for ingestion, search, and eval.
- Docker compose for Qdrant + OpenSearch.
- Local-only free pipeline with Groq for LLM tasks.

## Risks & Mitigations
- Lower accuracy vs paid LLMs → strong reranking + hybrid retrieval.
- Doc parsing noise → normalization/cleanup steps.
- Large docs → chunking thresholds and batching.

## Milestones
1) Backend + React UI skeleton
2) Ingestion + chunking
3) Hybrid retrieval + reranking
4) Citations + answer generation
5) Evaluation + hallucination detection
6) Feedback + A/B testing
