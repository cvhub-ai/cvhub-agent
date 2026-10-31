from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import ToolException, tool


@tool
def calculate(
    a: float,
    b: float,
    operation: Literal["add", "subtract", "multiply", "divide"],
) -> float:
    """Perform a basic arithmetic operation on two numbers.

    Use this tool for addition, subtraction, multiplication,
    or division when an exact calculation is needed.
    """
    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":
        if b == 0:
            raise ToolException("Division by zero is not allowed.")
        return a / b

    raise ToolException("Unsupported arithmetic operation.")


@tool
def get_current_date(timezone: str = "UTC") -> str:
    """Get the current date in a specified timezone.

    Use this tool when the user asks for today's date.
    The timezone must be a valid IANA timezone name,
    such as UTC, Europe/Berlin, or Asia/Shanghai.
    """
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ToolException(f"Invalid timezone: {timezone}") from exc

    return datetime.now(tz).date().isoformat()
