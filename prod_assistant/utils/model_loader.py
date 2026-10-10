
import os

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from prod_assistant.utils.config_loader import load_config


class ModelLoader:
    """Load configured chat models and embedding models."""

    def __init__(self) -> None:
        self.config = load_config()

    def load_embeddings(self):
        embedding_config = self.config["embedding_model"]

        if embedding_config["provider"].lower() == "google":
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("Missing required API key: GOOGLE_API_KEY")

            return GoogleGenerativeAIEmbeddings(
                model=embedding_config["model_name"]
            )

        raise ValueError(
            f"Unsupported embedding provider: "
            f"{embedding_config['provider']}"
        )

    def load_llm(self, provider: str | None = None):
        llm_config = self.config["llm"]

        if provider is None:
            provider = "google"

        provider = provider.lower()

        if provider not in llm_config:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        api_key_env_vars = {
            "google": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        api_key_name = api_key_env_vars[provider]

        if not os.getenv(api_key_name):
            raise ValueError(f"Missing required API key: {api_key_name}")

        model_config = llm_config[provider]
        common_args = {
            "model": model_config["model_name"],
            "temperature": model_config.get("temperature", 0),
        }

        if "max_output_tokens" in model_config:
            common_args["max_tokens"] = model_config["max_output_tokens"]

        if provider == "google":
            return ChatGoogleGenerativeAI(**common_args)

        if provider == "groq":
            return ChatGroq(**common_args)

        if provider == "openai":
            return ChatOpenAI(**common_args)

        raise ValueError(f"Unsupported LLM provider: {provider}")
