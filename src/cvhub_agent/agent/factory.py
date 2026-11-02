from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph


class AgentFactory:
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    def build(
        self,
        tools: Sequence[BaseTool],
        system_prompt: str,
        middleware: Sequence[AgentMiddleware[Any, Any]] | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> CompiledStateGraph:
        return create_agent(
            model=self._model,
            tools=tools,
            system_prompt=system_prompt,
            middleware=middleware or (),
            checkpointer=checkpointer,
        )
