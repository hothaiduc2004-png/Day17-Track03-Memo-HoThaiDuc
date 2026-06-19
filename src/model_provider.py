from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Student TODO: map aliases like `anthorpic` -> `anthropic`."""
    if not value:
        return ""
    v = value.strip().lower()
    alias_map = {
        "anthorpic": "anthropic",
        "gpt": "openai",
        "openai": "openai",
        "custom": "custom",
        "gemini": "gemini",
        "ollama": "ollama",
        "openrouter": "openrouter",
    }
    return alias_map.get(v, v)


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider."""
    provider = normalize_provider(config.provider)
    if provider == "offline":
        return None

    try:
        if provider == "openai":
            from langchain.chat_models import ChatOpenAI

            return ChatOpenAI(
                model_name=config.model_name,
                temperature=config.temperature,
                openai_api_key=config.api_key,
            )

        if provider == "custom":
            from langchain.chat_models import ChatOpenAI

            return ChatOpenAI(
                model_name=config.model_name,
                temperature=config.temperature,
                openai_api_key=config.api_key,
                openai_api_base=config.base_url,
            )

        if provider == "gemini":
            from langchain.chat_models import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
                api_base=config.base_url,
            )

        if provider == "anthropic":
            from langchain.chat_models import ChatAnthropic

            return ChatAnthropic(
                model=config.model_name,
                temperature=config.temperature,
                anthropic_api_key=config.api_key,
            )

        if provider == "ollama":
            from langchain.chat_models import ChatOllama

            return ChatOllama(
                model=config.model_name,
                temperature=config.temperature,
                base_url=config.base_url,
            )

        if provider == "openrouter":
            from langchain.chat_models import ChatOpenRouter

            return ChatOpenRouter(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
                base_url=config.base_url,
            )
    except ImportError:
        return None

    return None
