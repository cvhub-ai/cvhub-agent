import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field, SecretStr


class LLMConfig(BaseModel):
    provider: Literal["deepseek", "local"]
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key: SecretStr

    temperature: float = Field(default=0.1, ge=0, le=2)
    max_output_tokens: int = Field(default=4096, gt=0)
    context_length: int = Field(default=4096, gt=0)

    timeout_seconds: float = Field(default=120, gt=0)
    max_retries: int = Field(default=1, ge=0)

    enable_thinking: bool = False


def load_llm_config(
    config_path: str | Path = "config/config.yaml",
    env_path: str | Path = ".env",
) -> LLMConfig:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or not isinstance(config.get("llm"), dict):
        raise TypeError("Missing or invalid 'llm' configuration")

    llm_config = config["llm"]

    # Load environment variables
    env_values = {**dotenv_values(env_path), **os.environ}

    # Select model provider
    provider = env_values.get("LLM_PROVIDER") or llm_config.get("provider")

    if provider not in ("deepseek", "local"):
        raise ValueError(f"Unsupported LLM provider: {provider!r}")

    # Load provider configuration
    providers = llm_config.get("providers")

    if not isinstance(providers, dict) or not isinstance(providers.get(provider), dict):
        raise TypeError(f"Missing configuration for LLM provider: {provider}")

    provider_config = providers[provider]

    # Resolve provider-specific environment variables
    env_prefix = "DEEPSEEK" if provider == "deepseek" else "LLM"

    api_key = env_values.get(f"{env_prefix}_API_KEY")

    if not api_key:
        raise ValueError(f"{env_prefix}_API_KEY is not configured")

    base_url = env_values.get(f"{env_prefix}_BASE_URL") or provider_config.get(
        "base_url"
    )

    model = env_values.get(f"{env_prefix}_MODEL") or provider_config.get("model")

    # Extract shared settings
    common_config = {
        key: value
        for key, value in llm_config.items()
        if key not in {"provider", "providers"}
    }

    return LLMConfig.model_validate(
        {
            **common_config,
            "provider": provider,
            "base_url": base_url,
            "model": model,
            "api_key": api_key,
        }
    )
