"""
Centralized application configuration.
Loads from environment variables / .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

    chunk_size: int = 1000
    chunk_overlap: int = 150

    vector_store_path: str = "data/vector_store"
    upload_dir: str = "data/uploads"
    retrieval_top_k: int = 4


settings = Settings()
