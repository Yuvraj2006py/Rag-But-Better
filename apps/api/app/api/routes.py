from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, UploadFile

from app.api.models import (
    HealthResponse,
    UploadResponse,
    SearchRequest,
    SearchResponse,
    CitationModel,
    FeedbackRequest,
    FeedbackResponse,
    ExperimentConfigResponse,
    EvaluationResponse,
)
from app.core.models import Feedback
from app.services.embedding import SimpleEmbedder
from app.services.experiments import assign_config, default_configs
from app.services.feedback import FeedbackStore
from app.services.groq_client import GroqClient
from app.services.pipeline import Pipeline
from app.services.retrieval import HybridRetriever
from app.services.evaluation import evaluate_answer, is_hallucinated

router = APIRouter()

embedder = SimpleEmbedder()
pipeline = Pipeline(embedder=embedder)
feedback_store = FeedbackStore()
experiments = default_configs()

groq_client = GroqClient()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile]) -> UploadResponse:
    os.makedirs("data", exist_ok=True)
    uploaded = 0
    for file in files:
        if not file.filename:
            continue
        dest = Path("data") / file.filename
        contents = await file.read()
        dest.write_bytes(contents)
        uploaded += pipeline.ingest_file(str(dest))
    return UploadResponse(uploaded=uploaded)


@router.post("/search/query", response_model=SearchResponse)
def search_query(payload: SearchRequest) -> SearchResponse:
    config = assign_config(payload.query, experiments)
    rewritten = groq_client.rewrite_query(payload.query)

    retriever = HybridRetriever(
        vector_store=pipeline.vector_store,
        sparse_store=pipeline.sparse_store,
        embedder=embedder,
    )
    results = retriever.search(
        rewritten,
        top_k=config.top_k,
        dense_weight=config.dense_weight,
        sparse_weight=config.sparse_weight,
        do_rerank=config.rerank,
    )
    chunk_ids = [r.chunk.chunk_id for r in results]
    context = "\n\n".join(r.chunk.text for r in results)

    answer = groq_client.answer(payload.query, context)
    evaluation = evaluate_answer(payload.query, answer, context)
    citations = pipeline.citations_for(chunk_ids)

    return SearchResponse(
        answer=answer,
        citations=[
            CitationModel(doc_id=c.doc_id, chunk_id=c.chunk_id, snippet=c.snippet)
            for c in citations
        ],
        faithfulness=evaluation.faithfulness,
        hallucinated=is_hallucinated(evaluation.faithfulness),
    )


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    feedback_store.add(
        Feedback(
            query=payload.query,
            rating=payload.rating,
            note=payload.note,
        )
    )
    return FeedbackResponse(stored=True)


@router.get("/experiments/configs", response_model=list[ExperimentConfigResponse])
def list_experiments() -> list[ExperimentConfigResponse]:
    return [
        ExperimentConfigResponse(
            config_id=c.config_id,
            dense_weight=c.dense_weight,
            sparse_weight=c.sparse_weight,
            top_k=c.top_k,
            rerank=c.rerank,
        )
        for c in experiments
    ]


@router.post("/evaluation/run", response_model=EvaluationResponse)
def run_evaluation(payload: SearchRequest) -> EvaluationResponse:
    retriever = HybridRetriever(
        vector_store=pipeline.vector_store,
        sparse_store=pipeline.sparse_store,
        embedder=embedder,
    )
    results = retriever.search(payload.query, top_k=5)
    context = "\n\n".join(r.chunk.text for r in results)
    answer = groq_client.answer(payload.query, context)
    evaluation = evaluate_answer(payload.query, answer, context)
    return EvaluationResponse(
        faithfulness=evaluation.faithfulness,
        answer_relevance=evaluation.answer_relevance,
        context_precision=evaluation.context_precision,
        context_recall=evaluation.context_recall,
    )
