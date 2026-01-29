from app.services.embedding import SimpleEmbedder
from app.services.retrieval import HybridRetriever
from app.services.stores import InMemoryVectorStore, InMemorySparseStore
from app.core.models import Chunk


def test_hybrid_retrieval_returns_results():
    embedder = SimpleEmbedder()
    vector_store = InMemoryVectorStore()
    sparse_store = InMemorySparseStore()

    chunks = [
        Chunk("c1", "d1", "apple banana", 0, 0),
        Chunk("c2", "d1", "orange pear", 1, 1),
    ]

    for c in chunks:
        vector_store.add(c, embedder.embed([c.text])[0])
        sparse_store.add(c)

    retriever = HybridRetriever(vector_store, sparse_store, embedder)
    results = retriever.search("apple")
    assert results
    assert results[0].chunk.chunk_id in {"c1", "c2"}
