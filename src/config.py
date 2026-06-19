from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / '.env')
except ImportError:
    pass


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Student TODO: load environment variables and return a LabConfig.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Basic defaults to allow offline/testing runs when env vars are not set
    state_dir = (root / "state")
    state_dir.mkdir(parents=True, exist_ok=True)

    data_dir = root / "data"

    # Sensible defaults for compact memory
    compact_threshold_tokens = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "2000"))
    compact_keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "8"))

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "offline"))
    model_name = os.getenv("LLM_MODEL", "offline-model")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CUSTOM_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OLLAMA_API_KEY")
    base_url = os.getenv("CUSTOM_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or os.getenv("OPENROUTER_BASE_URL") or os.getenv("GEMINI_BASE_URL")

    provider_config = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=provider_config,
        judge_model=provider_config,
    )
