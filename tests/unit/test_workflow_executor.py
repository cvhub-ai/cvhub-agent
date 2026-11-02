import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from cvhub_agent.workflows.executor import ExecutorNode
from cvhub_agent.workflows.state import StepResult, WorkflowState


@pytest.mark.asyncio
async def test_executor_runs_one_step():
    received_inputs = []

    async def fake_agent(agent_input):
        received_inputs.append(agent_input)
        return {"messages": [AIMessage(content="The sum is 500.")]}

    executor = ExecutorNode(agent=RunnableLambda(fake_agent))

    state = WorkflowState(
        goal="Calculate 125 + 375.",
        plan=["Calculate the sum"],
        step_index=0,
        status="running",
    )

    update = await executor(state)

    # Verify execution progress
    assert update["step_index"] == 1
    assert len(update["results"]) == 1

    # Verify StepResult
    result = update["results"][0]

    assert isinstance(result, StepResult)
    assert result.step == "Calculate the sum"
    assert result.output == "The sum is 500."

    # Verify Agent received the current step
    assert len(received_inputs) == 1

    prompt = received_inputs[0]["messages"][0].content
    assert "Calculate the sum" in prompt
    assert "Calculate 125 + 375" in prompt

    # Executor does not mutate the original State
    assert state.step_index == 0
    assert state.results == []


@pytest.mark.asyncio
async def test_executor_receives_previous_results():
    received_inputs = []

    async def fake_agent(agent_input):
        received_inputs.append(agent_input)
        return {"messages": [AIMessage(content="125 + 375 = 500.")]}

    executor = ExecutorNode(agent=RunnableLambda(fake_agent))

    state = WorkflowState(
        goal="Calculate and explain 125 + 375.",
        plan=[
            "Calculate 125 + 375",
            "Explain the result",
        ],
        step_index=1,
        results=[
            StepResult(
                step="Calculate 125 + 375",
                output="500",
            )
        ],
        status="running",
    )

    update = await executor(state)

    assert update["step_index"] == 2
    assert len(update["results"]) == 1

    prompt = received_inputs[0]["messages"][0].content

    assert "Explain the result" in prompt
    assert "Calculate 125 + 375" in prompt
    assert "Result: 500" in prompt


@pytest.mark.asyncio
async def test_executor_rejects_invalid_step():
    async def fake_agent(agent_input):
        pytest.fail("Agent must not be called")

    executor = ExecutorNode(agent=RunnableLambda(fake_agent))

    state = WorkflowState(
        goal="Test",
        plan=["Only step"],
        step_index=1,
        status="running",
    )

    update = await executor(state)

    assert update["status"] == "failed"
    assert update["error"] is not None
