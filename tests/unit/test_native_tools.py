from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest
from langchain_core.tools import ToolException
from pydantic import ValidationError

from cvhub_agent.tools.native_tools import (
    calculate,
    get_current_date,
)


@pytest.mark.parametrize(
    ("a", "b", "operation", "expected"),
    [
        (10, 5, "add", 15.0),
        (10, 5, "subtract", 5.0),
        (10, 5, "multiply", 50.0),
        (10, 5, "divide", 2.0),
        (-3, 2, "multiply", -6.0),
        (5, 2, "divide", 2.5),
    ],
)
def test_calculate(a, b, operation, expected):
    result = calculate.invoke(
        {
            "a": a,
            "b": b,
            "operation": operation,
        }
    )

    assert result == pytest.approx(expected)


def test_calculate_division_by_zero():
    with pytest.raises(
        ToolException,
        match="Division by zero",
    ):
        calculate.invoke(
            {
                "a": 10,
                "b": 0,
                "operation": "divide",
            }
        )


def test_calculate_invalid_operation():
    with pytest.raises(ValidationError):
        calculate.invoke(
            {
                "a": 10,
                "b": 5,
                "operation": "power",
            }
        )


@pytest.mark.parametrize(
    "timezone",
    [
        "UTC",
        "Europe/Berlin",
        "Asia/Shanghai",
    ],
)
def test_get_current_date(timezone):
    tz = ZoneInfo(timezone)

    before = datetime.now(tz).date()

    result = get_current_date.invoke({"timezone": timezone})

    after = datetime.now(tz).date()

    assert isinstance(result, str)
    assert date.fromisoformat(result) in {before, after}


def test_get_current_date_default_timezone():
    result = get_current_date.invoke({})

    assert isinstance(result, str)
    assert date.fromisoformat(result)


def test_get_current_date_invalid_timezone():
    with pytest.raises(
        ToolException,
        match="Invalid timezone",
    ):
        get_current_date.invoke({"timezone": "Invalid/Timezone"})
