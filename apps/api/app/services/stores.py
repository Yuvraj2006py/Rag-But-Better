from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from app.core.models import Chunk


@dataclass
class VectorResult:
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}
        self._chunks: dict[str, Chunk] = {}

    def add(self, chunk: Chunk, vector: list[float]) -> None:
        self._chunks[chunk.chunk_id] = chunk
        self._vectors[chunk.chunk_id] = vector

    def search(self, query_vector: list[float], top_k: int) -> list[VectorResult]:
        q = np.array(query_vector)
        results: list[VectorResult] = []
        for chunk_id, vec in self._vectors.items():
            v = np.array(vec)
            denom = (np.linalg.norm(q) * np.linalg.norm(v))
            score = float(np.dot(q, v) / denom) if denom != 0 else 0.0
            results.append(VectorResult(self._chunks[chunk_id], score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


@dataclass
class SparseResult:
    chunk: Chunk
    score: float


class InMemorySparseStore:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def add(self, chunk: Chunk) -> None:
        self._chunks[chunk.chunk_id] = chunk

    def search(self, query: str, top_k: int) -> list[SparseResult]:
        tokens = set(query.lower().split())
        results: list[SparseResult] = []
        for chunk in self._chunks.values():
            score = sum(1 for t in chunk.text.lower().split() if t in tokens)
            results.append(SparseResult(chunk, float(score)))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
