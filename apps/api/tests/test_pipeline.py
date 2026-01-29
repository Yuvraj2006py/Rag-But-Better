from pathlib import Path

from app.services.embedding import SimpleEmbedder
from app.services.pipeline import Pipeline


def test_pipeline_ingest_txt(tmp_path: Path):
    file_path = tmp_path / "doc.txt"
    file_path.write_text("alpha beta gamma", encoding="utf-8")
    pipeline = Pipeline(SimpleEmbedder())
    count = pipeline.ingest_file(str(file_path))
    assert count >= 1
