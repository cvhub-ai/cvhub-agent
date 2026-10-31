import pytest
from pydantic import SecretStr, ValidationError

from cvhub_agent.config.settings import load_llm_config


@pytest.fixture
def config_files(tmp_path, monkeypatch):
    # Isolate tests from real environment variables
    for key in (
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"

    config_path.write_text(
        """
llm:
  provider: deepseek
  providers:
    deepseek:
      base_url: https://api.deepseek.com
      model: deepseek-flash
    local:
      base_url: http://localhost:8000/v1
      model: qwen3.5-9b

  temperature: 0.1
  max_output_tokens: 4096
  timeout_seconds: 120
  max_retries: 1
  enable_thinking: false
""",
        encoding="utf-8",
    )

    env_path.write_text(
        "DEEPSEEK_API_KEY=test-deepseek-key\nLLM_API_KEY=test-local-key\n",
        encoding="utf-8",
    )

    return config_path, env_path


def test_load_deepseek_config(config_files):
    config = load_llm_config(*config_files)

    assert config.provider == "deepseek"
    assert config.model == "deepseek-flash"
    assert config.base_url == "https://api.deepseek.com"
    assert config.max_output_tokens == 4096
    assert config.timeout_seconds == 120
    assert config.max_retries == 1


def test_switch_to_local_provider(config_files, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")

    config = load_llm_config(*config_files)

    assert config.provider == "local"
    assert config.model == "qwen3.5-9b"
    assert config.base_url == "http://localhost:8000/v1"
    assert config.api_key.get_secret_value() == "test-local-key"


def test_api_key_is_secret(config_files):
    config = load_llm_config(*config_files)

    assert isinstance(config.api_key, SecretStr)
    assert config.api_key.get_secret_value() == "test-deepseek-key"
    assert "test-deepseek-key" not in repr(config)


def test_environment_override(config_files, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL", "test-model")

    config = load_llm_config(*config_files)

    assert config.model == "test-model"


def test_invalid_provider(config_files, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="Unsupported"):
        load_llm_config(*config_files)


def test_missing_api_key(config_files):
    config_path, env_path = config_files
    env_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="API_KEY"):
        load_llm_config(config_path, env_path)


def test_invalid_temperature(config_files):
    config_path, env_path = config_files
    content = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        content.replace("temperature: 0.1", "temperature: 3.0"),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_llm_config(config_path, env_path)
