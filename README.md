# RAG but Better

## Phase 0 Setup (local)

### Prereqs
- Python 3.11+
- Node.js 20+
- Docker Desktop (WSL2 backend enabled)
- Groq API key

### Environment
1) Copy `.env.example` to `.env` and set your Groq key.
2) Start Docker Desktop.

### Infra
From repo root:

```bash
docker compose -f infra/docker-compose.yml up -d
```

Verify:
- Qdrant: http://localhost:6333
- OpenSearch: http://localhost:9200

## Backend (FastAPI)

```bash
cd apps/api
python -m venv .venv
.venv/Scripts/activate
pip install -e .[test]
pytest
uvicorn app.main:app --reload --port 8000
```

## Frontend (React)

```bash
cd apps/web
npm install
npm test
npm run dev
```

## External Stores (Qdrant/OpenSearch)
Set in `.env`:

```
USE_EXTERNAL_STORES=true
QDRANT_URL=http://localhost:6333
OPENSEARCH_URL=http://localhost:9200
```
