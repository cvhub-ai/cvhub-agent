import os
from pathlib import Path

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field, SecretStr


class LLMConfig(BaseModel):
    base_url: str
    model: str

    temperature: float = Field(default=0.1, ge=0, le=2)
    max_output_tokens: int = Field(default=512, gt=0)
    context_length: int = Field(default=4096, gt=0)

    timeout_seconds: float = Field(default=120, gt=0)
    max_retries: int = Field(default=1, ge=0)

    enable_thinking: bool = False

    api_key: SecretStr


def load_llm_config(
    config_path: str | Path = "config/config.yaml",
    env_path: str | Path = ".env",
) -> LLMConfig:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or not isinstance(config.get("llm"), dict):
        raise ValueError("Missing or invalid 'llm' configuration")

    api_key = os.environ.get("LLM_API_KEY") or dotenv_values(env_path).get(
        "LLM_API_KEY"
    )

    if not api_key:
        raise ValueError("LLM_API_KEY is not configured")

    return LLMConfig.model_validate(
        {
            **config["llm"],
            "api_key": api_key,
        }
    )
