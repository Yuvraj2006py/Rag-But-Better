from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    doc_id: str
    title: str
    source_path: str
    text: str
    metadata: dict[str, Any]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start_paragraph: int
    end_paragraph: int


@dataclass
class Citation:
    doc_id: str
    chunk_id: str
    snippet: str


@dataclass
class QueryTrace:
    query: str
    rewritten_query: str
    retrieved_chunk_ids: list[str]
    answer: str


@dataclass
class Feedback:
    query: str
    rating: int
    note: str | None


@dataclass
class ExperimentConfig:
    config_id: str
    dense_weight: float
    sparse_weight: float
    top_k: int
    rerank: bool


@dataclass
class EvaluationResult:
    query: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
