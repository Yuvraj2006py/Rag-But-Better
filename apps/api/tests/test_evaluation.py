from app.services.evaluation import evaluate_answer, is_hallucinated


def test_evaluation_scores():
    result = evaluate_answer("cat", "cat sits", "cat sits on mat")
    assert 0 <= result.faithfulness <= 1
    assert 0 <= result.answer_relevance <= 1


def test_hallucination_detection():
    assert is_hallucinated(0.2)
    assert not is_hallucinated(0.9)
