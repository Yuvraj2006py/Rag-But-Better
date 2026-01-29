from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class UploadResponse(BaseModel):
    uploaded: int


class SearchRequest(BaseModel):
    query: str


class CitationModel(BaseModel):
    doc_id: str
    chunk_id: str
    snippet: str


class SearchResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    faithfulness: float
    hallucinated: bool


class FeedbackRequest(BaseModel):
    query: str
    rating: int
    note: str | None = None


class FeedbackResponse(BaseModel):
    stored: bool


class ExperimentConfigResponse(BaseModel):
    config_id: str
    dense_weight: float
    sparse_weight: float
    top_k: int
    rerank: bool


class EvaluationResponse(BaseModel):
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
