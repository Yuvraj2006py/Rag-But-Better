from __future__ import annotations

from app.core.models import EvaluationResult


def _token_set(text: str) -> set[str]:
    return {t for t in text.lower().split() if t.strip()}


def evaluate_answer(query: str, answer: str, context: str) -> EvaluationResult:
    q_tokens = _token_set(query)
    a_tokens = _token_set(answer)
    c_tokens = _token_set(context)

    answer_relevance = len(a_tokens & q_tokens) / max(len(a_tokens), 1)
    faithfulness = len(a_tokens & c_tokens) / max(len(a_tokens), 1)
    context_precision = len(c_tokens & a_tokens) / max(len(c_tokens), 1)
    context_recall = len(c_tokens & a_tokens) / max(len(a_tokens), 1)

    return EvaluationResult(
        query=query,
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_precision=context_precision,
        context_recall=context_recall,
    )


def is_hallucinated(faithfulness: float, threshold: float = 0.6) -> bool:
    return faithfulness < threshold
