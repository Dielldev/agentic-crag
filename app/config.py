"""Application configuration loaded from environment variables."""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings, populated from .env / environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (chat / grading / generation)
    groq_api_key: str = ""

    # Web search fallback
    tavily_api_key: str = ""

    # LangSmith observability
    langsmith_api_key: str = ""
    langsmith_project: str = "agentic-crag"

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "documents"


settings = Settings()

# Enable LangSmith tracing when an API key is configured. pydantic-settings reads
# .env into this Settings object but does NOT export to os.environ, which is where
# LangChain/LangGraph's tracer looks — so we bridge the relevant values across.
if settings.langsmith_api_key:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
