
import pytest

from prod_assistant.utils.model_loader import ModelLoader


def test_model_loader_reads_all_configured_providers():
    loader = ModelLoader()
    assert set(loader.config["llm"].keys()) == {"google", "groq", "openai"}


@pytest.mark.parametrize(
    ("provider", "api_key"),
    [
        ("google", "GOOGLE_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
    ],
)
def test_load_llm_rejects_missing_api_key(provider, api_key, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loader = ModelLoader()

    with pytest.raises(ValueError, match=api_key):
        loader.load_llm(provider)


def test_load_llm_rejects_unknown_provider():
    loader = ModelLoader()

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        loader.load_llm("unknown")