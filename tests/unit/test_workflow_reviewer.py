from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.language_models import BaseChatModel

from cvhub_agent.workflows.reviewer import (
    ReviewerNode,
    ReviewOutput,
)
from cvhub_agent.workflows.state import (
    ReviewResult,
    StepResult,
    WorkflowState,
)


def make_reviewer(
    approved: bool,
    feedback: str,
):
    model = Mock(spec=BaseChatModel)

    structured_model = Mock()
    structured_model.ainvoke = AsyncMock(
        return_value=ReviewOutput(
            approved=approved,
            feedback=feedback,
        )
    )

    model.with_structured_output.return_value = structured_model

    return ReviewerNode(model), structured_model


def make_completed_state() -> WorkflowState:
    return WorkflowState(
        goal="Calculate 125 + 375 and explain the result.",
        plan=[
            "Calculate the sum",
            "Explain the result",
        ],
        step_index=2,
        results=[
            StepResult(
                step="Calculate the sum",
                output="500",
            ),
            StepResult(
                step="Explain the result",
                output="125 + 375 = 500",
            ),
        ],
        status="running",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("approved", "feedback"),
    [
        (True, "The goal has been satisfied."),
        (False, "The explanation is insufficient."),
    ],
)
async def test_reviewer_returns_review_result(
    approved: bool,
    feedback: str,
):
    reviewer, structured_model = make_reviewer(
        approved=approved,
        feedback=feedback,
    )

    state = make_completed_state()

    update = await reviewer(state)

    review = update["review"]

    assert isinstance(review, ReviewResult)
    assert review.approved is approved
    assert review.feedback == feedback

    # Verify that the LLM was called once.
    structured_model.ainvoke.assert_awaited_once()

    # Verify the reviewer received the task context.
    messages = structured_model.ainvoke.await_args.args[0]

    assert state.goal in messages[1].content
    assert "Calculate the sum" in messages[1].content
    assert "500" in messages[1].content

    # Reviewer must not modify the input state.
    assert state.review is None
    assert state.status == "running"


@pytest.mark.asyncio
async def test_reviewer_rejects_incomplete_execution():
    reviewer, structured_model = make_reviewer(
        approved=True,
        feedback="Approved",
    )

    state = make_completed_state()
    state.step_index = 1

    with pytest.raises(
        ValueError,
        match="incomplete",
    ):
        await reviewer(state)

    # Incomplete execution should not trigger an LLM call.
    structured_model.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_reviewer_rejects_mismatched_steps():
    reviewer, structured_model = make_reviewer(
        approved=True,
        feedback="Approved",
    )

    state = make_completed_state()
    state.results[0].step = "Unexpected step"

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        await reviewer(state)

    structured_model.ainvoke.assert_not_awaited()
