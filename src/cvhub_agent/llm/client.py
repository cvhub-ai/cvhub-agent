from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from cvhub_agent.config.settings import LLMConfig


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        extra_body = self._build_extra_body(config)

        self._model = ChatOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            temperature=(
                None
                if config.provider == "deepseek" and config.enable_thinking
                else config.temperature
            ),
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
            use_responses_api=False,
            extra_body=extra_body,
        )

    @staticmethod
    def _build_extra_body(config: LLMConfig) -> dict[str, Any]:
        extra_body: dict[str, Any] = {
            "max_tokens": config.max_output_tokens,
        }

        if config.provider == "deepseek":
            extra_body["thinking"] = {
                "type": "enabled" if config.enable_thinking else "disabled",
            }

        elif config.provider == "local":
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": config.enable_thinking,
            }

        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

        return extra_body

    @property
    def model(self) -> ChatOpenAI:
        return self._model

    def chat(self, messages: list[BaseMessage]) -> AIMessage:
        response = self._model.invoke(messages)

        if not isinstance(response, AIMessage):
            raise TypeError("Expected AIMessage from LLM")

        return response
