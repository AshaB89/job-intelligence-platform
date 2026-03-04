from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the project.

    Loads from environment variables and (optionally) a local `.env` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Logging
    log_level: str = "INFO"

    # Data paths
    postings_csv: Path = Path("src/data/csv/postings.csv")
    jobs_clean_csv: Path = Path("src/data/csv/jobs_clean.csv")

    # Retrieval / ranking defaults
    default_query: str = "machine learning engineer"
    candidate_k: int = 50
    final_k: int = 10

    # Evaluation defaults
    eval_corpus_rows: int = 5000
    eval_rows: int = 200
    eval_k_retrieve: int = 50
    eval_k: int = 10

    # RAG (local Ollama by default)
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "phi3"
    ollama_timeout_s: float = 60.0

    # Optional HF token (kept for future; not required for current Ollama flow)
    hf_api_key: Optional[str] = None


def get_settings() -> Settings:
    return Settings()

