from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from opensearchpy import OpenSearch

from app.core.config import settings
from app.core.models import Chunk


@dataclass
class ExternalVectorResult:
    chunk: Chunk
    score: float


class QdrantVectorStore:
    def __init__(self) -> None:
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection for c in collections):
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=128, distance=Distance.COSINE),
            )

    def add(self, chunk: Chunk, vector: list[float]) -> None:
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload={
                        "doc_id": chunk.doc_id,
                        "text": chunk.text,
                        "start_paragraph": chunk.start_paragraph,
                        "end_paragraph": chunk.end_paragraph,
                    },
                )
            ],
        )

    def search(self, query_vector: list[float], top_k: int) -> list[ExternalVectorResult]:
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
        )
        results: list[ExternalVectorResult] = []
        for hit in hits:
            payload = hit.payload or {}
            chunk = Chunk(
                chunk_id=str(hit.id),
                doc_id=payload.get("doc_id", ""),
                text=payload.get("text", ""),
                start_paragraph=payload.get("start_paragraph", 0),
                end_paragraph=payload.get("end_paragraph", 0),
            )
            results.append(ExternalVectorResult(chunk=chunk, score=hit.score))
        return results


@dataclass
class ExternalSparseResult:
    chunk: Chunk
    score: float


class OpenSearchStore:
    def __init__(self) -> None:
        self.client = OpenSearch(settings.opensearch_url)
        self.index = settings.opensearch_index
        if not self.client.indices.exists(index=self.index):
            self.client.indices.create(index=self.index)

    def add(self, chunk: Chunk) -> None:
        doc = {
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            "start_paragraph": chunk.start_paragraph,
            "end_paragraph": chunk.end_paragraph,
        }
        self.client.index(index=self.index, id=chunk.chunk_id, body=doc, refresh=True)

    def search(self, query: str, top_k: int) -> list[ExternalSparseResult]:
        res = self.client.search(
            index=self.index,
            body={"query": {"match": {"text": query}}, "size": top_k},
        )
        hits = res.get("hits", {}).get("hits", [])
        results: list[ExternalSparseResult] = []
        for hit in hits:
            source = hit.get("_source", {})
            chunk = Chunk(
                chunk_id=str(hit.get("_id")),
                doc_id=source.get("doc_id", ""),
                text=source.get("text", ""),
                start_paragraph=source.get("start_paragraph", 0),
                end_paragraph=source.get("end_paragraph", 0),
            )
            results.append(ExternalSparseResult(chunk=chunk, score=float(hit.get("_score", 0))))
        return results
