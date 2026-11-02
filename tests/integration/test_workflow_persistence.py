import os
from uuid import uuid4

import pytest
from dotenv import load_dotenv
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
    def __init__(self, events: list[str]):
        self.events = events

    async def __call__(self, state: WorkflowState) -> dict:
        self.events.append("planner")

        return {
            "plan": [
                "Calculate 125 + 375",
                "Explain the result",
            ],
            "step_index": 0,
            "status": "running",
            "error": None,
        }


class FakeExecutor:
    def __init__(self, events: list[str]):
        self.events = events

    async def __call__(self, state: WorkflowState) -> dict:
        self.events.append(f"executor:{state.step_index}")

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


class FakeReviewer:
    def __init__(self, events: list[str]):
        self.events = events

    async def __call__(self, state: WorkflowState) -> dict:
        self.events.append("reviewer")

        assert state.step_index == len(state.plan)
        assert len(state.results) == len(state.plan)

        return {
            "review": ReviewResult(
                approved=True,
                feedback="The goal has been satisfied.",
            )
        }


def create_workflow(
    checkpointer,
    events: list[str],
) -> TaskWorkflow:
    return TaskWorkflow(
        planner=FakePlanner(events),  # type: ignore[arg-type]
        executor=FakeExecutor(events),  # type: ignore[arg-type]
        reviewer=FakeReviewer(events),  # type: ignore[arg-type]
        checkpointer=checkpointer,
    )


# ============================================================
# PostgreSQL Persistence + Resume
# ============================================================


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_TESTS") != "1",
    reason="PostgreSQL integration tests are disabled",
)
async def test_workflow_postgres_persistence_and_resume():
    load_dotenv()

    database_url = os.getenv("AGENT_DATABASE_URL")

    if not database_url:
        raise ValueError("AGENT_DATABASE_URL is not configured.")

    thread_id = f"test-workflow-{uuid4().hex}"

    thread_config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    events: list[str] = []

    try:
        # ----------------------------------------------------
        # Session 1: Execute and pause after the first step
        # ----------------------------------------------------
        async with open_checkpointer(
            "postgres",
            database_url=database_url,
        ) as checkpointer:
            workflow = create_workflow(checkpointer, events)

            partial_result = await workflow.run(
                WorkflowState(goal="Calculate 125 + 375 and explain the result."),
                thread_id=thread_id,
                pause_after_executor=True,
            )

            assert partial_result.status == "running"
            assert partial_result.step_index == 1
            assert len(partial_result.results) == 1
            assert partial_result.results[0].output == "500"

            assert partial_result.review is None

            assert events == [
                "planner",
                "executor:0",
            ]

            # Verify the checkpoint is paused
            snapshot = await workflow._graph.aget_state(thread_config)

            assert snapshot.values["step_index"] == 1
            assert len(snapshot.values["results"]) == 1
            assert "executor" in snapshot.next

            print("\n=== Session 1: Paused ===")
            print("Thread ID:", thread_id)
            print("Step Index:", partial_result.step_index)
            print("Next Nodes:", snapshot.next)

        # ----------------------------------------------------
        # Session 2: Reconnect and resume the workflow
        # ----------------------------------------------------
        async with open_checkpointer(
            "postgres",
            database_url=database_url,
        ) as checkpointer:
            # Create a new workflow instance
            workflow = create_workflow(checkpointer, events)

            # Verify state before resuming
            snapshot = await workflow._graph.aget_state(thread_config)

            assert snapshot.values["step_index"] == 1
            assert len(snapshot.values["results"]) == 1
            assert "executor" in snapshot.next

            # Resume from the saved checkpoint
            result = await workflow.resume(thread_id=thread_id)

            assert result.status == "completed"
            assert result.error is None

            assert result.step_index == 2
            assert len(result.plan) == 2
            assert len(result.results) == 2

            assert result.results[0].output == "500"
            assert "500" in result.results[1].output

            assert result.review is not None
            assert result.review.approved is True

            # No previously completed node was re-executed.
            assert events == [
                "planner",
                "executor:0",
                "executor:1",
                "reviewer",
            ]

            # A completed workflow cannot be resumed again.
            with pytest.raises(
                ValueError,
                match="already finished",
            ):
                await workflow.resume(thread_id=thread_id)

            # The same thread cannot start a new workflow.
            with pytest.raises(
                ValueError,
                match="already has a workflow",
            ):
                await workflow.run(
                    WorkflowState(goal="Another task"),
                    thread_id=thread_id,
                )

            print("\n=== Session 2: Resumed ===")
            print("Status:", result.status)
            print("Step Index:", result.step_index)
            print("Execution Events:", events)

        # ----------------------------------------------------
        # Session 3: Verify final persisted state
        # ----------------------------------------------------
        async with open_checkpointer(
            "postgres",
            database_url=database_url,
        ) as checkpointer:
            workflow = create_workflow(checkpointer, events)

            snapshot = await workflow._graph.aget_state(thread_config)

            assert snapshot.values["status"] == "completed"
            assert snapshot.values["step_index"] == 2
            assert len(snapshot.values["results"]) == 2
            assert snapshot.next == ()

            # Read checkpoint history
            history = [
                item async for item in workflow._graph.aget_state_history(thread_config)
            ]

            assert len(history) > 1

            # Verify intermediate checkpoint was preserved.
            assert any(
                item.values.get("step_index") == 1
                and len(item.values.get("results", [])) == 1
                for item in history
            )

            print("\n=== Session 3: Persistence Verified ===")
            print("Final Status:", snapshot.values["status"])
            print("Checkpoint Count:", len(history))

    finally:
        # Remove only the test-created thread.
        async with open_checkpointer(
            "postgres",
            database_url=database_url,
        ) as checkpointer:
            await checkpointer.adelete_thread(thread_id)
