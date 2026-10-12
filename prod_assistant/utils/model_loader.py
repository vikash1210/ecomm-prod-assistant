
"""Load configured chat models and embedding models."""

import os

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from prod_assistant.utils.config_loader import load_config


class ModelLoader:
    """Load configured LLMs and embeddings with a Groq fallback."""

    def __init__(self) -> None:
        self.config = load_config()

    def load_embeddings(self):
        """Load the configured Google embedding model."""
        embedding_config = self.config["embedding_model"]
        provider = embedding_config["provider"].lower()

        if provider != "google":
            raise ValueError(f"Unsupported embedding provider: {provider}")
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("Missing required API key: GOOGLE_API_KEY")
        
        return GoogleGenerativeAIEmbeddings(
        model=embedding_config["model_name"],)
    
    def _load_provider_llm(self, provider: str):
        """Create a chat model for the selected provider."""
        llm_config = self.config["llm"]

        if provider not in llm_config:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        model_config = llm_config[provider]
        api_key_env = {
            "google": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
        }[provider]

        if not os.getenv(api_key_env):
            raise ValueError(f"Missing required API key: {api_key_env}")

        model_name = model_config["model_name"]
        temperature = model_config.get("temperature", 0)
        max_tokens = model_config.get("max_output_tokens", 1024)

        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

        if provider == "groq":
            return ChatGroq(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def load_llm(self, provider: str | None = None):
        """Load Google as primary and Groq as fallback by default."""
        selected_provider = (
            provider or os.getenv("LLM_PROVIDER", "google")
        ).lower()

        primary_llm = self._load_provider_llm(selected_provider)

        if selected_provider != "google":
            return primary_llm

        # If Google fails during a model call, try Groq.
        fallback_llm = self._load_provider_llm("groq")
        return primary_llm.with_fallbacks([fallback_llm])
