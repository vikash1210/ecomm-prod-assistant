
"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Settings shared across the application."""

    app_name: str = "E-commerce Product Assistant"
    app_env: str = "development"
    log_level: str = "INFO"

    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1-mini"

    openai_api_key: str | None = None
    google_api_key: str | None = None
    groq_api_key: str | None = None

    astra_db_api_endpoint: str | None = None
    astra_db_application_token: str | None = None
    astra_db_keyspace: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls(
            app_name=os.getenv(
                "APP_NAME", "E-commerce Product Assistant"
            ),
            app_env=os.getenv("APP_ENV", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            llm_provider=os.getenv("LLM_PROVIDER", "openai").lower(),
            llm_model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            astra_db_api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
            astra_db_application_token=os.getenv(
                "ASTRA_DB_APPLICATION_TOKEN"
            ),
            astra_db_keyspace=os.getenv("ASTRA_DB_KEYSPACE"),
        )


settings = Settings.from_env()
