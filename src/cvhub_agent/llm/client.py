from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from cvhub_agent.config.settings import LLMConfig


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self._model = ChatOpenAI(
            base_url=config.base_url,
            api_key=config.api_key.get_secret_value(),
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
            use_responses_api=False,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": config.enable_thinking,
                }
            },
        )

    @property
    def model(self) -> ChatOpenAI:
        return self._model

    def chat(self, messages: list[BaseMessage]) -> AIMessage:
        response = self._model.invoke(messages)

        if not isinstance(response, AIMessage):
            raise TypeError("Expected AIMessage from LLM")

        return response
