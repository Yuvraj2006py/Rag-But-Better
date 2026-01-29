from __future__ import annotations

import hashlib
import math
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


def _hash_token(token: str, dim: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % dim


class SimpleEmbedder:
    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=float)
            for token in text.lower().split():
                idx = _hash_token(token, self.dim)
                vec[idx] += 1.0
            norm = math.sqrt(float(np.dot(vec, vec)))
            if norm > 0:
                vec = vec / norm
            vectors.append(vec.tolist())
        return vectors
