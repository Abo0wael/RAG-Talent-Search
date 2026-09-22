"""
RAG Talent Search Engine - LLM Provider Factory

Provides a unified interface for LLM instances across providers:
- Groq (Default: llama-3.3-70b-versatile)
- OpenAI (Alternative: gpt-4o-mini)
- Google Gemini (Alternative: gemini-1.5-flash)

Handles missing API keys gracefully with clear error guidance.
"""

import os
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_TEMPERATURE,
    GROQ_API_KEY,
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
)


class LLMConfigurationError(Exception):
    """Raised when an LLM provider is misconfigured or lacks required credentials."""
    pass


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None
) -> BaseChatModel:
    """
    Instantiate and return the configured LangChain chat model.

    Args:
        provider: Provider name ('groq', 'openai', 'google'). Defaults to config.LLM_PROVIDER.
        model: Specific model name. Defaults to provider default in config.
        temperature: Sampling temperature. Defaults to config.LLM_TEMPERATURE.

    Returns:
        BaseChatModel: Initialized LangChain chat model.

    Raises:
        LLMConfigurationError: If API keys are missing or provider unsupported.
    """
    selected_provider = (provider or LLM_PROVIDER).lower().strip()
    selected_temp = temperature if temperature is not None else LLM_TEMPERATURE

    if selected_provider == "groq":
        api_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
        if not api_key or api_key == "your_groq_api_key_here":
            raise LLMConfigurationError(
                "Groq API Key is missing. Please set GROQ_API_KEY in your .env file:\n"
                "  GROQ_API_KEY=gsk_...\n"
                "You can obtain a key at: https://console.groq.com/"
            )
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise LLMConfigurationError(
                "Package 'langchain-groq' is not installed. Run: pip install langchain-groq"
            )

        selected_model = model or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        return ChatGroq(
            model=selected_model,
            temperature=selected_temp,
            groq_api_key=api_key,
        )

    elif selected_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
        if not api_key or api_key == "your_openai_api_key_here":
            raise LLMConfigurationError(
                "OpenAI API Key is missing. Please set OPENAI_API_KEY in your .env file:\n"
                "  OPENAI_API_KEY=sk-...\n"
                "You can obtain a key at: https://platform.openai.com/"
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise LLMConfigurationError(
                "Package 'langchain-openai' is not installed. Run: pip install langchain-openai"
            )

        selected_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            model=selected_model,
            temperature=selected_temp,
            openai_api_key=api_key,
        )

    elif selected_provider in ("google", "gemini"):
        api_key = os.getenv("GOOGLE_API_KEY") or GOOGLE_API_KEY
        if not api_key or api_key == "your_google_api_key_here":
            raise LLMConfigurationError(
                "Google Gemini API Key is missing. Please set GOOGLE_API_KEY in your .env file:\n"
                "  GOOGLE_API_KEY=AIza...\n"
                "You can obtain a key at: https://aistudio.google.com/"
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise LLMConfigurationError(
                "Package 'langchain-google-genai' is not installed. Run: pip install langchain-google-genai"
            )

        selected_model = model or os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
        return ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=selected_temp,
            google_api_key=api_key,
        )

    else:
        raise LLMConfigurationError(
            f"Unsupported LLM provider: '{selected_provider}'. "
            f"Supported options are: 'groq', 'openai', 'google'."
        )


def check_llm_status() -> dict[str, Any]:
    """
    Check current LLM configuration and key availability.

    Returns:
        dict with provider, model, key_set boolean, and status message.
    """
    provider = LLM_PROVIDER
    has_key = False
    key_name = ""

    if provider == "groq":
        key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
        has_key = bool(key and key != "your_groq_api_key_here")
        key_name = "GROQ_API_KEY"
    elif provider == "openai":
        key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
        has_key = bool(key and key != "your_openai_api_key_here")
        key_name = "OPENAI_API_KEY"
    elif provider in ("google", "gemini"):
        key = os.getenv("GOOGLE_API_KEY") or GOOGLE_API_KEY
        has_key = bool(key and key != "your_google_api_key_here")
        key_name = "GOOGLE_API_KEY"

    return {
        "provider": provider,
        "model": LLM_MODEL,
        "key_set": has_key,
        "key_env_var": key_name,
        "status": "Ready" if has_key else f"Missing {key_name} in .env"
    }
