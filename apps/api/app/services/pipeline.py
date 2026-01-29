from __future__ import annotations

from app.core.config import settings
from app.core.models import Chunk, Citation
from app.services.chunking import semantic_chunk
from app.services.embedding import Embedder
from app.services.ingest import parse_file
from app.services.stores import InMemoryVectorStore, InMemorySparseStore

try:
    from app.services.external_stores import QdrantVectorStore, OpenSearchStore
except Exception:  # pragma: no cover - optional external deps
    QdrantVectorStore = None
    OpenSearchStore = None


class Pipeline:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        if settings.use_external_stores and QdrantVectorStore and OpenSearchStore:
            self.vector_store = QdrantVectorStore()
            self.sparse_store = OpenSearchStore()
        else:
            self.vector_store = InMemoryVectorStore()
            self.sparse_store = InMemorySparseStore()
        self.chunks: dict[str, Chunk] = {}

    def ingest_file(self, path: str) -> int:
        doc = parse_file(path)
        chunks = semantic_chunk(doc.text, doc.doc_id, self.embedder)
        embeddings = self.embedder.embed([c.text for c in chunks])
        for chunk, vec in zip(chunks, embeddings, strict=False):
            self.chunks[chunk.chunk_id] = chunk
            self.vector_store.add(chunk, vec)
            self.sparse_store.add(chunk)
        return len(chunks)

    def citations_for(self, chunk_ids: list[str]) -> list[Citation]:
        citations = []
        for cid in chunk_ids:
            chunk = self.chunks.get(cid)
            if not chunk:
                continue
            snippet = chunk.text[:200]
            citations.append(Citation(doc_id=chunk.doc_id, chunk_id=chunk.chunk_id, snippet=snippet))
        return citations
