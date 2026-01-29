from __future__ import annotations

import hashlib

from app.core.models import ExperimentConfig


def default_configs() -> list[ExperimentConfig]:
    return [
        ExperimentConfig("A", dense_weight=1.0, sparse_weight=1.0, top_k=5, rerank=True),
        ExperimentConfig("B", dense_weight=1.5, sparse_weight=0.5, top_k=8, rerank=True),
    ]


def assign_config(query: str, configs: list[ExperimentConfig]) -> ExperimentConfig:
    if not configs:
        raise ValueError("No experiment configs available")
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(configs)
    return configs[idx]
