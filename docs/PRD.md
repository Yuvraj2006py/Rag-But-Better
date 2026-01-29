# PRD: NLP for Internal Company Documents (RAG but Better)

## 1. Overview
Build a local web application that enables semantic search and QA over internal documents with robust chunking, hybrid retrieval, reranking, citations, evaluation, and feedback. The system must use Groq for LLM tasks and remain fully free/self-hosted for all other components.

## 2. Goals
- Accurate, grounded answers with citations per statement.
- Hybrid retrieval (dense + sparse) with reranking for relevance.
- Meaningful evaluation metrics and hallucination detection.
- User feedback loop and A/B testing to improve retrieval and answer quality.
- Local React UI with upload, search, and analysis views.

## 3. Non-Goals
- Multi-tenant auth or enterprise identity integration.
- Paid managed vector DB or hosted LLMs besides Groq free tier.
- Advanced enterprise compliance features.

## 4. Users & Use Cases
### Primary Users
- Knowledge workers and internal stakeholders.

### Use Cases
- Upload internal docs (PDF/DOCX/HTML/TXT/MD).
- Ask questions across documents.
- Inspect citations and source passages.
- Provide feedback on answer quality.
- Compare experiments to improve retrieval and chunking.

## 5. Functional Requirements
### 5.1 Ingestion
- Accept multiple file types: PDF, DOCX, HTML, TXT, MD.
- Normalize extracted text to a structured document model.
- Store metadata: `doc_id`, title, source path, page, paragraph, timestamps.

### 5.2 Chunking
- Semantic chunking (sentence-level embeddings).
- Min/max chunk size bounds.
- Persist chunk metadata and mapping back to original docs.

### 5.3 Retrieval
- Dense search: Qdrant (embedding retrieval).
- Sparse search: OpenSearch (BM25).
- Hybrid merging via Reciprocal Rank Fusion.
- Reranking via cross-encoder.

### 5.4 Query Rewrite/Expansion (Groq)
- Rewrite or expand user query to improve recall.
- Preserve user intent and avoid semantic drift.

### 5.5 Answer Generation (Groq)
- Generate answers strictly from retrieved chunks.
- Provide citations for each answer segment.
- If evidence is insufficient, respond with “Not enough information.”

### 5.6 Citation Tracking
- Track `doc_id`, `chunk_id`, paragraph range.
- UI displays citations and source snippets.

### 5.7 Quality Scoring
- RAGAS metrics (faithfulness, relevance, context precision/recall).
- Store scores per query.

### 5.8 Hallucination Detection
- Flag low faithfulness or unsupported claims.
- Expose in UI as warning label.

### 5.9 Feedback Loop
- Thumbs up/down + optional comment.
- Persist feedback linked to query and experiment config.

### 5.10 A/B Testing
- Configurable experiment settings (chunk sizes, weights, top-K, reranker).
- Randomized assignment per query.
- Dashboard comparison of metrics and feedback.

### 5.11 Evaluation
- Batch evaluation with labeled queries.
- Metrics: recall@k, precision@k, RAGAS averages.
- Export CSV reports.

## 6. Non-Functional Requirements
- Local deployment on Windows with Docker.
- All dependencies free/self-hosted except Groq API usage.
- System should handle incremental ingestion without re-indexing all documents.
- Latency target: < 5 seconds for typical query on small datasets.

## 7. Tech Stack
- UI: React (local web app)
- API: FastAPI (Python)
- Embeddings + rerank: sentence-transformers
- Vector DB: Qdrant (Docker)
- Sparse retrieval: OpenSearch (Docker)
- Metadata: SQLite (upgrade path to Postgres)
- Evaluation: RAGAS / TruLens
- LLM: Groq API

## 8. Data Model (High Level)
- Document: `doc_id`, title, path, metadata
- Chunk: `chunk_id`, `doc_id`, text, start/end paragraph, embedding
- QueryTrace: query, rewritten query, retrieved chunks, answer
- Feedback: query_id, rating, note
- ExperimentConfig: weights, top-K, chunking params
- Evaluation: query_id, metrics

## 9. UX Requirements
- Upload screen with progress and status.
- Search screen with answer + citations.
- Source panel showing chunk text with highlights.
- Quality indicators (score + hallucination warnings).
- Feedback buttons and optional notes.
- Experiment comparison view.

## 10. Success Metrics
- Retrieval recall@k > 0.7 on labeled queries.
- RAGAS faithfulness > 0.7 average.
- Positive feedback rate increasing over time.

## 11. Risks & Mitigations
- Parsing errors → robust normalization + fallbacks.
- Low relevance → hybrid retrieval + reranking.
- Hallucinations → strict grounding + faithfulness scoring.
- Large docs → chunking thresholds and batching.

## 12. Milestones
1) Repo scaffold + React UI skeleton
2) Ingestion + semantic chunking
3) Hybrid retrieval + reranking
4) Citations + Groq answer generation
5) Evaluation + hallucination detection
6) Feedback + A/B testing
