from __future__ import annotations

import re
from typing import Iterable

import numpy as np

from app.core.models import Chunk
from app.services.embedding import Embedder


_sentence_splitter = re.compile(r"(?<=[.!?])\s+")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def semantic_chunk(
    text: str,
    doc_id: str,
    embedder: Embedder,
    min_sentences: int = 2,
    max_sentences: int = 8,
    similarity_threshold: float = 0.6,
) -> list[Chunk]:
    sentences = [s.strip() for s in _sentence_splitter.split(text) if s.strip()]
    if not sentences:
        return []

    embeddings = embedder.embed(sentences)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_start = 0

    for idx, sentence in enumerate(sentences):
        current.append(sentence)
        if len(current) < min_sentences:
            continue

        if len(current) >= max_sentences:
            chunk_text = " ".join(current)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-{len(chunks)}",
                    doc_id=doc_id,
                    text=chunk_text,
                    start_paragraph=current_start,
                    end_paragraph=idx,
                )
            )
            current = []
            current_start = idx + 1
            continue

        prev_vec = np.array(embeddings[idx - 1])
        curr_vec = np.array(embeddings[idx])
        if _cosine(prev_vec, curr_vec) < similarity_threshold:
            chunk_text = " ".join(current)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-{len(chunks)}",
                    doc_id=doc_id,
                    text=chunk_text,
                    start_paragraph=current_start,
                    end_paragraph=idx,
                )
            )
            current = []
            current_start = idx + 1

    if current:
        chunk_text = " ".join(current)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}-{len(chunks)}",
                doc_id=doc_id,
                text=chunk_text,
                start_paragraph=current_start,
                end_paragraph=len(sentences) - 1,
            )
        )

    return chunks
