from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cvhub_agent.workflows.state import WorkflowState


class PlanOutput(BaseModel):
    steps: list[str] = Field(
        min_length=1,
        max_length=5,
        description="Ordered steps required to complete the task.",
    )


class PlannerNode:
    def __init__(
        self,
        model: BaseChatModel,
        tools: Sequence[BaseTool],
    ) -> None:
        self._model = model.with_structured_output(
            PlanOutput,
            method="json_mode",
        )

        self._tool_descriptions = (
            "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
            or "No external tools are available."
        )

    async def __call__(
        self,
        state: WorkflowState,
    ) -> dict:
        system_prompt = (
            "You are a task planner for an AI agent.\n"
            "Break the user's goal into clear, actionable steps.\n"
            "Use only capabilities provided by the available tools.\n"
            "Do not invent unavailable tools or capabilities.\n"
            "If a required capability is unavailable, include a "
            "step to report the limitation rather than fabricate results.\n"
            "Do not execute any tools.\n"
            "Return only a JSON object with a 'steps' field "
            "containing a list of 1 to 5 strings."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Goal:\n{state.goal}\n\n"
                    f"Available tools:\n{self._tool_descriptions}"
                )
            ),
        ]

        response = await self._model.ainvoke(messages)

        if not isinstance(response, PlanOutput):
            raise TypeError("Planner returned an invalid response.")

        return {
            "plan": response.steps,
            "step_index": 0,
            "status": "running",
            "error": None,
        }
