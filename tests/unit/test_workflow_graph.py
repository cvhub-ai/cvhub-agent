import pytest
from langchain_core.runnables import RunnableConfig

from cvhub_agent.persistence.checkpointer import open_checkpointer
from cvhub_agent.workflows.state import (
    ReviewResult,
    StepResult,
    WorkflowState,
)
from cvhub_agent.workflows.workflow import TaskWorkflow

# ============================================================
# Fake Nodes
# ============================================================


class FakePlanner:
    async def __call__(self, state: WorkflowState) -> dict:
        return {
            "plan": [
                "Calculate 125 + 375",
                "Explain the result",
            ],
            "step_index": 0,
            "status": "running",
            "error": None,
        }


class EmptyPlanner:
    async def __call__(self, state: WorkflowState) -> dict:
        return {
            "plan": [],
            "step_index": 0,
            "status": "running",
        }


class CrashingPlanner:
    async def __call__(self, state: WorkflowState) -> dict:
        raise ValueError("Simulated planner failure")


class FakeExecutor:
    def __init__(self):
        self.received_indices: list[int] = []

    async def __call__(self, state: WorkflowState) -> dict:
        self.received_indices.append(state.step_index)

        step = state.plan[state.step_index]

        if state.step_index == 0:
            output = "500"
        else:
            output = f"The calculated sum is {state.results[0].output}."

        return {
            "results": [
                StepResult(
                    step=step,
                    output=output,
                )
            ],
            "step_index": state.step_index + 1,
        }


class FailingExecutor:
    def __init__(self):
        self.calls = 0

    async def __call__(self, state: WorkflowState) -> dict:
        self.calls += 1

        return {
            "status": "failed",
            "error": "Simulated execution failure",
        }


class FlakyExecutor(FakeExecutor):
    def __init__(self, failures: int):
        super().__init__()
        self.remaining_failures = failures
        self.attempted_indices: list[int] = []

    async def __call__(self, state: WorkflowState) -> dict:
        self.attempted_indices.append(state.step_index)

        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise ConnectionError("Simulated connection failure")

        return await super().__call__(state)


class CrashingExecutor:
    def __init__(self):
        self.calls = 0

    async def __call__(self, state: WorkflowState) -> dict:
        self.calls += 1
        raise ValueError("Simulated unexpected executor failure")


class FakeReviewer:
    def __init__(
        self,
        approved: bool,
        feedback: str,
    ):
        self.approved = approved
        self.feedback = feedback
        self.calls = 0

    async def __call__(self, state: WorkflowState) -> dict:
        self.calls += 1

        assert state.step_index == len(state.plan)
        assert len(state.results) == len(state.plan)

        return {
            "review": ReviewResult(
                approved=self.approved,
                feedback=self.feedback,
            )
        }


class CrashingReviewer:
    def __init__(self):
        self.calls = 0

    async def __call__(self, state: WorkflowState) -> dict:
        self.calls += 1
        raise RuntimeError("Simulated reviewer failure")


# ============================================================
# Normal Execution
# ============================================================


@pytest.mark.asyncio
async def test_task_workflow_approved():
    executor = FakeExecutor()
    reviewer = FakeReviewer(
        approved=True,
        feedback="The goal has been satisfied.",
    )

    workflow = TaskWorkflow(
        planner=FakePlanner(),  # type: ignore
        executor=executor,  # type: ignore
        reviewer=reviewer,  # type: ignore
    )

    initial_state = WorkflowState(goal="Calculate 125 + 375 and explain the result.")

    result = await workflow.run(initial_state)

    assert isinstance(result, WorkflowState)
    assert result.status == "completed"
    assert result.error is None

    assert result.step_index == 2
    assert len(result.plan) == 2
    assert len(result.results) == 2

    assert executor.received_indices == [0, 1]

    assert result.results[0].step == result.plan[0]
    assert result.results[1].step == result.plan[1]

    assert result.results[0].output == "500"
    assert "500" in result.results[1].output

    assert reviewer.calls == 1
    assert result.review is not None
    assert result.review.approved is True

    assert result.step_retry_count == 0

    # The original input state must not be modified.
    assert initial_state.status == "pending"
    assert initial_state.plan == []
    assert initial_state.results == []
    assert initial_state.review is None


# ============================================================
# Execution Failure
# ============================================================


@pytest.mark.asyncio
async def test_task_workflow_executor_failure():
    executor = FailingExecutor()
    reviewer = FakeReviewer(
        approved=True,
        feedback="Approved",
    )

    workflow = TaskWorkflow(
        planner=FakePlanner(),  # type: ignore
        executor=executor,  # type: ignore
        reviewer=reviewer,  # type: ignore
        max_step_retries=2,
    )

    result = await workflow.run(WorkflowState(goal="Test execution failure"))

    assert result.status == "failed"
    assert result.error == "Simulated execution failure"

    assert result.step_index == 0
    assert result.results == []
    assert result.step_retry_count == 0

    # Explicit failure must not trigger automatic retry.
    assert executor.calls == 1

    # Reviewer must not run after execution failure.
    assert reviewer.calls == 0
    assert result.review is None


# ============================================================
# Review Rejection
# ============================================================


@pytest.mark.asyncio
async def test_task_workflow_review_rejected():
    executor = FakeExecutor()
    reviewer = FakeReviewer(
        approved=False,
        feedback="The result does not satisfy the goal.",
    )

    workflow = TaskWorkflow(
        planner=FakePlanner(),  # type: ignore
        executor=executor,  # type: ignore
        reviewer=reviewer,  # type: ignore
    )

    result = await workflow.run(WorkflowState(goal="Test review rejection"))

    # All steps are executed before review.
    assert executor.received_indices == [0, 1]
    assert result.step_index == len(result.plan)
    assert len(result.results) == len(result.plan)

    assert reviewer.calls == 1
    assert result.review is not None
    assert result.review.approved is False
    assert result.review.feedback == ("The result does not satisfy the goal.")

    assert result.status == "failed"
    assert result.error == result.review.feedback


# ============================================================
# Retry Success
# ============================================================


@pytest.mark.asyncio
async def test_task_workflow_retry_success():
    executor = FlakyExecutor(failures=1)
    reviewer = FakeReviewer(
        approved=True,
        feedback="Approved",
    )

    workflow = TaskWorkflow(
        planner=FakePlanner(),  # type: ignore
        executor=executor,  # type: ignore
        reviewer=reviewer,  # type: ignore
        max_step_retries=2,
    )

    result = await workflow.run(WorkflowState(goal="Calculate 125 + 375"))

    assert result.status == "completed"
    assert result.error is None

    # Step 0 fails once, then succeeds.
    # Step 1 is executed normally.
    assert executor.attempted_indices == [0, 0, 1]
    assert executor.received_indices == [0, 1]

    assert result.step_index == 2
    assert len(result.results) == 2

    # Retry count resets after successful execution.
    assert result.step_retry_count == 0

    assert reviewer.calls == 1
    assert result.review is not None
    assert result.review.approved is True


# ============================================================
# Retry Exhausted
# ============================================================


@pytest.mark.asyncio
async def test_task_workflow_retry_exhausted():
    executor = FlakyExecutor(failures=10)
    reviewer = FakeReviewer(
        approved=True,
        feedback="Approved",
    )

    workflow = TaskWorkflow(
        planner=FakePlanner(),  # type: ignore
        executor=executor,  # type: ignore
        reviewer=reviewer,  # type: ignore
        max_step_retries=2,
    )

    result = await workflow.run(WorkflowState(goal="Test retry exhaustion"))

    # Initial attempt + 2 retries.
    assert executor.attempted_indices == [0, 0, 0]
    assert executor.received_indices == []

    assert result.status == "failed"
    assert result.error is not None
    assert "ConnectionError" in result.error

    assert result.step_index == 0
    assert result.results == []
    assert result.step_retry_count == 2

    assert reviewer.calls == 0
    assert result.review is None


# ============================================================
# Retry Disabled by Default
# ============================================================


@pytest.mark.asyncio
async def test_task_workflow_retry_disabled_by_default():
    executor = FlakyExecutor(failures=1)
    reviewer = FakeReviewer(
        approved=True,
        feedback="Approved",
    )

    workflow = TaskWorkflow(
        planner=FakePlanner(),  # type: ignore
        executor=executor,  # type: ignore
        reviewer=reviewer,  # type: ignore
    )

    result = await workflow.run(WorkflowState(goal="Test retry disabled"))

    # No retry when max_step_retries = 0.
    assert executor.attempted_indices == [0]

    assert result.status == "failed"
    assert result.step_index == 0
    assert result.step_retry_count == 0
    assert result.results == []

    assert reviewer.calls == 0
    assert result.review is None


# ============================================================
# Planner Exception
# ============================================================


@pytest.mark.asyncio
async def test_task_workflow_planner_exception():
    executor = FakeExecutor()
    reviewer = FakeReviewer(
        approved=True,
        feedback="Approved",
    )

    workflow = TaskWorkflow(
        planner=CrashingPlanner(),  # type: ignore
        executor=executor,  # type: ignore
        reviewer=reviewer,  # type: ignore
    )

    result = await workflow.run(WorkflowState(goal="Test planner failure"))

    assert result.status == "failed"
    assert result.error == "Planner error: ValueError"

    assert result.plan == []
    assert result.step_index == 0
    assert result.results == []

    assert executor.received_indices == []
    assert reviewer.calls == 0
    assert result.review is None


# ============================================================
# Empty Plan
# ============================================================


@pytest.mark.asyncio
async def test_task_workflow_empty_plan():
    executor = FakeExecutor()
    reviewer = FakeReviewer(
        approved=True,
        feedback="Approved",
    )

    workflow = TaskWorkflow(
        planner=EmptyPlanner(),  # type: ignore
        executor=executor,  # type: ignore
        reviewer=reviewer,  # type: ignore
    )

    result = await workflow.run(WorkflowState(goal="Test empty plan"))

    assert result.status == "failed"
    assert result.error == "Planner returned an empty plan."

    assert result.plan == []
    assert result.results == []

    assert executor.received_indices == []
    assert reviewer.calls == 0


# ============================================================
# Reviewer Exception
# ============================================================


@pytest.mark.asyncio
async def test_task_workflow_reviewer_exception():
    executor = FakeExecutor()
    reviewer = CrashingReviewer()

    workflow = TaskWorkflow(
        planner=FakePlanner(),  # type: ignore
        executor=executor,  # type: ignore
        reviewer=reviewer,  # type: ignore
    )

    result = await workflow.run(WorkflowState(goal="Test reviewer failure"))

    assert result.status == "failed"
    assert result.error == "Reviewer error: RuntimeError"

    # Execution finishes before reviewer fails.
    assert executor.received_indices == [0, 1]
    assert result.step_index == len(result.plan)
    assert len(result.results) == len(result.plan)

    assert reviewer.calls == 1
    assert result.review is None


# ============================================================
# Unexpected Executor Exception
# ============================================================


@pytest.mark.asyncio
async def test_task_workflow_executor_unexpected_exception():
    executor = CrashingExecutor()
    reviewer = FakeReviewer(
        approved=True,
        feedback="Approved",
    )

    workflow = TaskWorkflow(
        planner=FakePlanner(),  # type: ignore
        executor=executor,  # type: ignore
        reviewer=reviewer,  # type: ignore
        max_step_retries=2,
    )

    result = await workflow.run(WorkflowState(goal="Test unexpected executor failure"))

    # ValueError is not retryable.
    assert executor.calls == 1

    assert result.status == "failed"
    assert result.error is not None
    assert "Executor error" in result.error
    assert "ValueError" in result.error

    assert result.step_index == 0
    assert result.results == []

    assert reviewer.calls == 0
    assert result.review is None


# ============================================================
# Invalid Retry Configuration
# ============================================================


def test_task_workflow_negative_retries():
    with pytest.raises(
        ValueError,
        match="max_step_retries must not be negative",
    ):
        TaskWorkflow(
            planner=FakePlanner(),  # type: ignore
            executor=FakeExecutor(),  # type: ignore
            reviewer=FakeReviewer(
                approved=True,
                feedback="Approved",
            ),  # type: ignore
            max_step_retries=-1,
        )


# ============================================================
# Workflow Checkpoint Persistence
# ============================================================


@pytest.mark.asyncio
async def test_task_workflow_inmemory_checkpoint():
    executor = FakeExecutor()
    reviewer = FakeReviewer(
        approved=True,
        feedback="Approved",
    )

    thread_id = "test-workflow-checkpoint"

    thread_config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    async with open_checkpointer("memory") as checkpointer:
        workflow = TaskWorkflow(
            planner=FakePlanner(),  # type: ignore
            executor=executor,  # type: ignore
            reviewer=reviewer,  # type: ignore
            checkpointer=checkpointer,
        )

        # Execute workflow
        result = await workflow.run(
            WorkflowState(goal="Calculate 125 + 375 and explain the result."),
            thread_id=thread_id,
        )

        assert result.status == "completed"
        assert result.step_index == 2
        assert len(result.results) == 2

        # Read the latest persisted checkpoint
        snapshot = await workflow._graph.aget_state(thread_config)

        assert snapshot.values["status"] == "completed"
        assert snapshot.values["goal"] == result.goal
        assert snapshot.values["plan"] == result.plan
        assert snapshot.values["step_index"] == 2
        assert len(snapshot.values["results"]) == 2
        assert snapshot.next == ()

        # Read checkpoint history
        history = [
            item async for item in workflow._graph.aget_state_history(thread_config)
        ]

        assert len(history) > 1

        # Verify an intermediate execution state was persisted
        assert any(
            item.values.get("step_index") == 1
            and len(item.values.get("results", [])) == 1
            for item in history
        )

        # Verify execution order
        assert executor.received_indices == [0, 1]
        assert reviewer.calls == 1


@pytest.mark.asyncio
async def test_task_workflow_checkpointer_requires_thread_id():
    async with open_checkpointer("memory") as checkpointer:
        workflow = TaskWorkflow(
            planner=FakePlanner(),  # type: ignore
            executor=FakeExecutor(),  # type: ignore
            reviewer=FakeReviewer(
                approved=True,
                feedback="Approved",
            ),  # type: ignore
            checkpointer=checkpointer,
        )

        with pytest.raises(
            ValueError,
            match="thread_id is required",
        ):
            await workflow.run(WorkflowState(goal="Test missing thread ID"))
