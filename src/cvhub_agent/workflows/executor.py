from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from cvhub_agent.workflows.state import StepResult, WorkflowState


class ExecutorNode:
    def __init__(self, agent: Runnable) -> None:
        self._agent = agent

    async def __call__(
        self,
        state: WorkflowState,
    ) -> dict:
        # Validate current step
        if not 0 <= state.step_index < len(state.plan):
            return {
                "status": "failed",
                "error": "No executable step at current index.",
            }

        # Get current step
        step = state.plan[state.step_index]

        # Collect previous step results
        previous_results = (
            "\n".join(
                f"Step: {result.step}\nResult: {result.output}"
                for result in state.results
            )
            or "No previous results."
        )

        # Build execution prompt
        prompt = (
            f"Original goal:\n{state.goal}\n\n"
            f"Current step:\n{step}\n\n"
            f"Previous results:\n{previous_results}\n\n"
            "Complete only the current step. "
            "Use available tools when necessary. "
            "Reuse previous results when possible. "
            "Do not repeat completed operations unnecessarily."
        )

        # Execute step using existing Agent
        response = await self._agent.ainvoke(
            {"messages": [HumanMessage(content=prompt)]}
        )

        # Extract final Agent response
        final_message = response["messages"][-1]

        if not isinstance(final_message, AIMessage):
            raise TypeError("Agent did not return a final AIMessage.")

        if final_message.tool_calls:
            raise ValueError("Agent returned unfinished tool calls.")

        output = final_message.content

        if not isinstance(output, str) or not output.strip():
            raise ValueError("Agent returned an empty or non-text result.")

        # Return partial Workflow State update
        return {
            "results": [
                StepResult(
                    step=step,
                    output=output,
                )
            ],
            "step_index": state.step_index + 1,
        }
