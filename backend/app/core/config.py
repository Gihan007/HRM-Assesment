"""Application settings, loaded from environment variables / .env.

See .env.example for the full list of supported variables and
PLAN.md / plan_recorrected.md for what each group controls.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.1-70b-instruct"
    llm_timeout_seconds: int = 20

    # --- Embeddings ---
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"

    # --- Reranker ---
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_threshold: float = 0.0

    # --- Retrieval pipeline ---
    top_k_wide: int = 25
    top_k: int = 5
    hybrid_alpha: float = 0.5
    query_expansion_variants: int = 2
    enable_query_expansion: bool = True

    # --- Chunking ---
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 100

    # --- Weaviate ---
    weaviate_url: str = "http://weaviate:8080"
    weaviate_api_key: str = ""

    # --- Postgres ---
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "ragdb"
    postgres_user: str = "raguser"
    postgres_password: str = ""
    database_url: str = ""

    # --- Auth ---
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # --- Storage ---
    upload_dir: str = "/data/uploads"
    max_upload_mb: int = 25

    # --- Memory ---
    long_term_summary_every_n_turns: int = 6

    # --- Service ---
    backend_port: int = 8000
    frontend_port: int = 8501
    backend_base_url: str = "http://backend:8000"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (reads .env once per process)."""
    return Settings()
