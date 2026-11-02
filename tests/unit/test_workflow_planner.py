from unittest.mock import AsyncMock, Mock

import pytest

from cvhub_agent.tools.loader import load_tools
from cvhub_agent.workflows.planner import (
    PlannerNode,
    PlanOutput,
)
from cvhub_agent.workflows.state import WorkflowState


@pytest.mark.asyncio
async def test_planner_generates_plan():
    # Mock structured LLM response
    model = Mock()

    structured_model = Mock()
    structured_model.ainvoke = AsyncMock(
        return_value=PlanOutput(
            steps=[
                "Calculate 125 + 375",
                "Explain the result",
            ]
        )
    )

    model.with_structured_output.return_value = structured_model

    # Create Planner
    planner = PlannerNode(
        model=model,
        tools=load_tools(),
    )

    state = WorkflowState(goal="Calculate 125 + 375 and explain the result.")

    # Execute Planner
    update = await planner(state)

    # Verify generated plan
    assert update["plan"] == [
        "Calculate 125 + 375",
        "Explain the result",
    ]

    assert update["step_index"] == 0
    assert update["status"] == "running"
    assert update["error"] is None

    # Verify LLM invocation
    structured_model.ainvoke.assert_awaited_once()

    # Verify initial state remains unchanged
    assert state.plan == []
    assert state.status == "pending"
