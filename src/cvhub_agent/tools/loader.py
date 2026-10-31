from collections.abc import Sequence

from langchain_core.tools import BaseTool

from cvhub_agent.tools.native_tools import (
    calculate,
    get_current_date,
)


def load_tools(
    external_tools: Sequence[BaseTool] = (),
) -> list[BaseTool]:
    """Collect native and external tools for the agent."""
    native_tools = [
        calculate,
        get_current_date,
    ]

    return [*native_tools, *external_tools]
