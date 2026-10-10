
from prod_assistant.utils.config_loader import load_config


def test_load_config_contains_expected_sections():
    config = load_config()

    assert "astra_db" in config
    assert "embedding_model" in config
    assert "retriever" in config
    assert "llm" in config


def test_load_config_contains_original_model_settings():
    config = load_config()

    assert config["astra_db"]["collection_name"] == "ecommercedata"
    assert config["embedding_model"]["provider"] == "google"
    assert config["retriever"]["top_k"] == 4
    assert config["llm"]["google"]["model_name"] == "gemini-2.0-flash"
    assert config["llm"]["groq"]["model_name"] == "deepseek-r1-distill-llama-70b"
    assert config["llm"]["openai"]["model_name"] == "gpt-4o"
