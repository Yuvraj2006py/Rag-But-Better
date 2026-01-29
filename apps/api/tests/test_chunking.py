from app.services.chunking import semantic_chunk
from app.services.embedding import SimpleEmbedder


def test_semantic_chunking_splits():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = semantic_chunk(text, "doc1", SimpleEmbedder(), min_sentences=1, max_sentences=2)
    assert len(chunks) >= 2
    assert all(c.doc_id == "doc1" for c in chunks)
