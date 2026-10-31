from collections.abc import Sequence

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.tools import ToolException
from langgraph.prebuilt.tool_node import ToolCallRequest


def _handle_tool_error(
    exc: Exception,
    request: ToolCallRequest,
) -> str | None:
    """Convert expected tool errors into safe agent messages."""

    if isinstance(exc, ToolException):
        return str(exc)

    return None


def build_agent_middleware(
    max_model_calls: int = 8,
    max_tool_calls: int = 12,
    retryable_tools: Sequence[str] = (),
    max_tool_retries: int = 1,
) -> list[
    ModelCallLimitMiddleware
    | ToolCallLimitMiddleware
    | ToolErrorMiddleware
    | ToolRetryMiddleware
]:
    """Build execution limits, error handling, and retry middleware."""

    if max_model_calls < 1:
        raise ValueError("max_model_calls must be positive")

    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be positive")

    if max_tool_retries < 0:
        raise ValueError("max_tool_retries cannot be negative")

    middleware: list[
        ModelCallLimitMiddleware
        | ToolCallLimitMiddleware
        | ToolErrorMiddleware
        | ToolRetryMiddleware
    ] = [
        ModelCallLimitMiddleware(
            run_limit=max_model_calls,
            exit_behavior="end",
        ),
        ToolCallLimitMiddleware(
            run_limit=max_tool_calls,
            exit_behavior="error",
        ),
        ToolErrorMiddleware(
            on_error=_handle_tool_error,
        ),
    ]

    if retryable_tools and max_tool_retries > 0:
        middleware.append(
            ToolRetryMiddleware(
                tools=list(retryable_tools),
                max_retries=max_tool_retries,
                retry_on=(TimeoutError, ConnectionError),
                on_failure="error",
            )
        )

    return middleware
