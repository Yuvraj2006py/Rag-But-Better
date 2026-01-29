from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    use_external_stores: bool = False
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_chunks"
    opensearch_url: str = "http://localhost:9200"
    opensearch_index: str = "rag_chunks"

    class Config:
        env_file = ".env"


settings = Settings()
