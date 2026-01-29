from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.core.models import Chunk
from app.services.embedding import Embedder


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def search(self, query_vector: list[float], top_k: int) -> list:
        ...


class SparseStore(Protocol):
    def search(self, query: str, top_k: int) -> list:
        ...


def rrf_merge(
    dense: list[RetrievedChunk],
    sparse: list[RetrievedChunk],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}
    for rank, item in enumerate(dense, start=1):
        scores[item.chunk.chunk_id] = scores.get(item.chunk.chunk_id, 0.0) + dense_weight / (k + rank)
        chunk_map[item.chunk.chunk_id] = item.chunk
    for rank, item in enumerate(sparse, start=1):
        scores[item.chunk.chunk_id] = scores.get(item.chunk.chunk_id, 0.0) + sparse_weight / (k + rank)
        chunk_map[item.chunk.chunk_id] = item.chunk

    merged = [RetrievedChunk(chunk_map[cid], score) for cid, score in scores.items()]
    merged.sort(key=lambda r: r.score, reverse=True)
    return merged


def rerank(query: str, chunks: list[RetrievedChunk], embedder: Embedder) -> list[RetrievedChunk]:
    query_vec = embedder.embed([query])[0]
    ranked = []
    for item in chunks:
        c_vec = embedder.embed([item.chunk.text])[0]
        score = sum(a * b for a, b in zip(query_vec, c_vec))
        ranked.append(RetrievedChunk(item.chunk, score))
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        sparse_store: SparseStore,
        embedder: Embedder,
    ) -> None:
        self.vector_store = vector_store
        self.sparse_store = sparse_store
        self.embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        do_rerank: bool = True,
    ) -> list[RetrievedChunk]:
        q_vec = self.embedder.embed([query])[0]
        dense_results = [RetrievedChunk(r.chunk, r.score) for r in self.vector_store.search(q_vec, top_k)]
        sparse_results = [RetrievedChunk(r.chunk, r.score) for r in self.sparse_store.search(query, top_k)]
        merged = rrf_merge(
            dense_results,
            sparse_results,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        merged = merged[: max(top_k, len(merged))]
        return rerank(query, merged, self.embedder) if do_rerank else merged
