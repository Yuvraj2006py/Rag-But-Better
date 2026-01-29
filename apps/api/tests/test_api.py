from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_upload_documents(tmp_path):
    files = [
        ("files", ("a.txt", b"hello world")),
        ("files", ("b.txt", b"second file")),
    ]
    resp = client.post("/documents/upload", files=files)
    assert resp.status_code == 200
    assert resp.json()["uploaded"] >= 1


def test_search_query_stub():
    resp = client.post("/search/query", json={"query": "what is rag"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "citations" in data
    assert "faithfulness" in data
    assert "hallucinated" in data


def test_feedback():
    resp = client.post("/feedback", json={"query": "q", "rating": 1})
    assert resp.status_code == 200
    assert resp.json() == {"stored": True}


def test_experiments_configs():
    resp = client.get("/experiments/configs")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_evaluation_run():
    resp = client.post("/evaluation/run", json={"query": "test query"})
    assert resp.status_code == 200
    data = resp.json()
    assert "faithfulness" in data
