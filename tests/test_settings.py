
from prod_assistant.config.settings import Settings


def test_settings_load_defaults(monkeypatch):
    """Settings should use defaults when environment variables are absent."""
    monkeypatch.delenv("APP_NAME", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    settings = Settings.from_env()

    assert settings.app_name == "E-commerce Product Assistant"
    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-4.1-mini"


def test_settings_reads_environment_variables(monkeypatch):
    """Settings should read overridden values from environment variables."""
    monkeypatch.setenv("APP_NAME", "Test Assistant")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    settings = Settings.from_env()

    assert settings.app_name == "Test Assistant"
    assert settings.llm_provider == "groq"
    assert settings.llm_model == "test-model"
