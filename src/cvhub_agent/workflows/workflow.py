import logging
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from cvhub_agent.workflows.executor import ExecutorNode
from cvhub_agent.workflows.planner import PlannerNode
from cvhub_agent.workflows.reviewer import ReviewerNode
from cvhub_agent.workflows.state import ReviewResult, WorkflowState

logger = logging.getLogger(__name__)


class TaskWorkflow:
    def __init__(
        self,
        *,
        planner: PlannerNode,
        executor: ExecutorNode,
        reviewer: ReviewerNode,
        max_step_retries: int = 0,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        if max_step_retries < 0:
            raise ValueError("max_step_retries must not be negative.")
        self._planner = planner
        self._executor = executor
        self._reviewer = reviewer
        self._max_step_retries = max_step_retries
        self._checkpointer = checkpointer

        self._graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(WorkflowState)

        # Register workflow nodes
        builder.add_node("planner", self._run_planner)
        builder.add_node("executor", self._run_executor)
        builder.add_node("reviewer", self._run_reviewer)
        builder.add_node("finish", self._finish)
        builder.add_node("reject", self._reject)

        # Define execution flow
        builder.add_edge(START, "planner")

        builder.add_conditional_edges(
            "planner",
            self._route_after_planning,
            {"execute": "executor", "failed": END},
        )

        # Route according to execution progress
        builder.add_conditional_edges(
            "executor",
            self._route_after_execution,
            {
                "continue": "executor",
                "retry": "executor",
                "review": "reviewer",
                "failed": END,
            },
        )

        # Route according to review result
        builder.add_conditional_edges(
            "reviewer",
            self._route_after_review,
            {
                "approved": "finish",
                "rejected": "reject",
                "failed": END,
            },
        )

        builder.add_edge("finish", END)
        builder.add_edge("reject", END)

        return builder.compile(checkpointer=self._checkpointer)

    async def _run_planner(self, state: WorkflowState) -> dict:
        try:
            update = await self._planner(state)
        except Exception as exc:
            logger.exception("Planner execution failed")
            return {
                "status": "failed",
                "error": f"Planner error: {type(exc).__name__}",
            }

        if not update.get("plan"):
            return {
                "status": "failed",
                "error": "Planner returned an empty plan.",
            }

        return update

    async def _run_executor(
        self,
        state: WorkflowState,
    ) -> dict:
        try:
            update = await self._executor(state)

        except (TimeoutError, ConnectionError) as exc:
            if state.step_retry_count < self._max_step_retries:
                retry_count = state.step_retry_count + 1

                logger.warning(
                    "Retrying workflow step %s: %s (%s/%s)",
                    state.step_index,
                    type(exc).__name__,
                    retry_count,
                    self._max_step_retries,
                )

                return {
                    "status": "retrying",
                    "step_retry_count": retry_count,
                    "error": (
                        f"Temporary failure at step "
                        f"{state.step_index}: {type(exc).__name__}"
                    ),
                }

            logger.warning(
                "Workflow step %s failed after retries",
                state.step_index,
            )

            return {
                "status": "failed",
                "error": (
                    f"Step {state.step_index} failed after "
                    f"{state.step_retry_count + 1} attempts: "
                    f"{type(exc).__name__}"
                ),
            }

        except Exception as exc:
            logger.exception(
                "Unexpected error at workflow step %s",
                state.step_index,
            )

            return {
                "status": "failed",
                "error": (
                    f"Executor error at step {state.step_index}: {type(exc).__name__}"
                ),
            }

        # Explicit execution failures are not retried.
        if update.get("status") == "failed":
            return {
                **update,
                "step_retry_count": 0,
            }

        # A successful execution must advance one step
        # and provide at least one result.
        if update.get("step_index") != state.step_index + 1 or not update.get(
            "results"
        ):
            return {
                "status": "failed",
                "error": "Executor did not produce a valid step update.",
            }

        return {
            **update,
            "status": "running",
            "error": None,
            "step_retry_count": 0,
        }

    async def _run_reviewer(
        self,
        state: WorkflowState,
    ) -> dict:
        try:
            update = await self._reviewer(state)
        except Exception as exc:
            logger.exception("Reviewer execution failed")
            return {
                "status": "failed",
                "error": f"Reviewer error: {type(exc).__name__}",
            }

        if not isinstance(update.get("review"), ReviewResult):
            return {
                "status": "failed",
                "error": "Reviewer returned an invalid result.",
            }

        return update

    def _route_after_planning(
        self,
        state: WorkflowState,
    ) -> Literal["execute", "failed"]:
        if state.status == "failed":
            return "failed"

        return "execute"

    def _route_after_execution(
        self,
        state: WorkflowState,
    ) -> Literal["continue", "retry", "review", "failed"]:
        if state.status == "failed":
            return "failed"

        if state.status == "retrying":
            return "retry"

        if state.step_index >= len(state.plan):
            return "review"

        return "continue"

    def _route_after_review(
        self, state: WorkflowState
    ) -> Literal["failed", "approved", "rejected"]:
        if state.status == "failed":
            return "failed"

        if state.review is None:
            return "failed"

        if state.review.approved:
            return "approved"

        return "rejected"

    def _finish(self, state: WorkflowState) -> dict:
        return {"status": "completed"}

    def _reject(self, state: WorkflowState) -> dict:
        if state.review is None:
            raise ValueError("Missing review result.")

        return {"status": "failed", "error": state.review.feedback}

    def _build_run_config(
        self,
        thread_id: str | None,
    ) -> RunnableConfig:
        recursion_limit = max(
            30,
            5 * (self._max_step_retries + 1) + 10,
        )

        config: RunnableConfig = {
            "recursion_limit": recursion_limit,
        }

        if self._checkpointer is not None:
            if not thread_id:
                raise ValueError(
                    "thread_id is required when a checkpointer is enabled."
                )

            config["configurable"] = {
                "thread_id": thread_id,
            }

        return config

    async def run(
        self,
        state: WorkflowState,
        *,
        thread_id: str | None = None,
        pause_after_executor: bool = False,
    ) -> WorkflowState:
        if pause_after_executor and self._checkpointer is None:
            raise ValueError("A checkpointer is required to pause a workflow.")

        config = self._build_run_config(thread_id)

        if self._checkpointer is not None:
            snapshot = await self._graph.aget_state(config)

            if snapshot.values or snapshot.next:
                raise ValueError(
                    "This thread already has a workflow. "
                    "Use a new thread_id or resume()."
                )

        result = await self._graph.ainvoke(
            state,
            config=config,
            interrupt_after=(["executor"] if pause_after_executor else None),
        )

        return WorkflowState(**result)

    async def resume(
        self,
        *,
        thread_id: str,
    ) -> WorkflowState:
        if self._checkpointer is None:
            raise ValueError("Cannot resume a workflow without a checkpointer.")

        config = self._build_run_config(thread_id)

        snapshot = await self._graph.aget_state(config)

        if not snapshot.values:
            raise ValueError(f"No checkpoint found for thread: {thread_id}")

        if not snapshot.next:
            raise ValueError(
                "Workflow has already finished. There are no pending nodes to resume."
            )

        result = await self._graph.ainvoke(
            None,
            config=config,
        )

        return WorkflowState(**result)
