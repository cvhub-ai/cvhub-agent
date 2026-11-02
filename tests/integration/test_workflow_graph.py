import os

import pytest

from cvhub_agent.agent.factory import AgentFactory
from cvhub_agent.agent.middleware import build_agent_middleware
from cvhub_agent.agent.prompts import DEFAULT_SYSTEM_PROMPT
from cvhub_agent.config.settings import load_llm_config
from cvhub_agent.llm.client import LLMClient
from cvhub_agent.tools.loader import load_tools
from cvhub_agent.workflows.executor import ExecutorNode
from cvhub_agent.workflows.planner import PlannerNode
from cvhub_agent.workflows.reviewer import ReviewerNode
from cvhub_agent.workflows.state import StepResult, WorkflowState
from cvhub_agent.workflows.workflow import TaskWorkflow


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_DEEPSEEK_TESTS") != "1",
    reason="Live DeepSeek tests are disabled",
)
async def test_task_workflow_with_deepseek():
    # --------------------------------------------------------
    # Load LLM configuration
    # --------------------------------------------------------
    config = load_llm_config()

    assert config.provider == "deepseek"
    assert not config.enable_thinking

    client = LLMClient(config)
    tools = load_tools()

    # --------------------------------------------------------
    # Initialize Planner
    # --------------------------------------------------------
    planner = PlannerNode(
        model=client.model,
        tools=tools,
    )

    # --------------------------------------------------------
    # Initialize LangChain Agent
    # --------------------------------------------------------
    agent = AgentFactory(client.model).build(
        tools=tools,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        middleware=build_agent_middleware(),
    )

    # --------------------------------------------------------
    # Initialize Executor and Reviewer
    # --------------------------------------------------------
    executor = ExecutorNode(agent=agent)

    reviewer = ReviewerNode(
        model=client.model,
    )

    # --------------------------------------------------------
    # Create Workflow
    # --------------------------------------------------------
    workflow = TaskWorkflow(
        planner=planner,
        executor=executor,
        reviewer=reviewer,
    )

    # --------------------------------------------------------
    # Execute Workflow
    # --------------------------------------------------------
    initial_state = WorkflowState(goal="Calculate 125 + 375 and explain the result.")

    result = await workflow.run(initial_state)

    # --------------------------------------------------------
    # Verify Workflow State
    # --------------------------------------------------------
    assert isinstance(result, WorkflowState)

    assert result.status == "completed"
    assert result.error is None

    assert len(result.plan) >= 1
    assert result.step_index == len(result.plan)
    assert len(result.results) == len(result.plan)

    assert result.step_retry_count == 0

    # --------------------------------------------------------
    # Verify Execution Results
    # --------------------------------------------------------
    for step, item in zip(
        result.plan,
        result.results,
        strict=True,
    ):
        assert isinstance(item, StepResult)
        assert item.step == step
        assert item.output.strip()

    outputs = "\n".join(item.output for item in result.results)

    assert "500" in outputs

    # --------------------------------------------------------
    # Verify Review
    # --------------------------------------------------------
    assert result.review is not None
    assert result.review.approved is True
    assert result.review.feedback.strip()

    # --------------------------------------------------------
    # Verify Original State
    # --------------------------------------------------------
    assert initial_state.status == "pending"
    assert initial_state.plan == []
    assert initial_state.results == []
    assert initial_state.review is None

    # --------------------------------------------------------
    # Print Execution Details
    # --------------------------------------------------------
    print("\n=== Task Workflow ===")
    print("Goal:", result.goal)
    print("Status:", result.status)

    print("\nPlan:")
    for index, step in enumerate(result.plan, start=1):
        print(f"{index}. {step}")

    print("\nExecution Results:")
    for index, item in enumerate(result.results, start=1):
        print(f"\nStep {index}: {item.step}")
        print(f"Output: {item.output}")

    print("\n=== Review ===")
    print("Approved:", result.review.approved)
    print("Feedback:", result.review.feedback)
