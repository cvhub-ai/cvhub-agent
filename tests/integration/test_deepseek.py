import os
from decimal import Decimal

import pytest
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from cvhub_agent.config.settings import load_llm_config
from cvhub_agent.llm.client import LLMClient

if os.getenv("RUN_DEEPSEEK_TESTS") != "1":
    pytest.skip(
        "Live DeepSeek tests are disabled",
        allow_module_level=True,
    )


class CityInfo(BaseModel):
    city: str = Field(description="City name")
    country_code: str = Field(description="ISO country code")


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integer numbers."""
    return a + b


@pytest.fixture(scope="module")
def client() -> LLMClient:
    config = load_llm_config()

    if config.provider != "deepseek":
        pytest.fail("Active LLM provider must be deepseek")

    if config.enable_thinking:
        pytest.fail("Run initial tests with thinking disabled")

    return LLMClient(config)


def report_usage(label: str, response) -> None:
    usage = response.usage_metadata
    assert usage is not None, "Missing Token Usage"

    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    total_tokens = usage["total_tokens"]

    print(
        f"\n[{label}] "
        f"input={input_tokens}, "
        f"output={output_tokens}, "
        f"total={total_tokens}"
    )

    # Optional USD cost estimation.
    input_price = os.getenv("DEEPSEEK_INPUT_USD_PER_1M")
    output_price = os.getenv("DEEPSEEK_OUTPUT_USD_PER_1M")

    if input_price is not None and output_price is not None:
        estimated_cost = (
            Decimal(input_tokens) * Decimal(input_price)
            + Decimal(output_tokens) * Decimal(output_price)
        ) / Decimal(1_000_000)

        print(f"[{label}] estimated_cost_usd={estimated_cost:.6f}")


def test_chat(client: LLMClient):
    response = client.chat([HumanMessage(content="Reply with exactly: Hello CVHub")])

    assert isinstance(response.content, str)
    assert response.content.strip()

    print(f"\nChat response: {response.content}")
    report_usage("chat", response)


def test_tool_calling(client: LLMClient):
    model = client.model.bind_tools(
        [add_numbers],
        tool_choice="required",
    )

    response = model.invoke(
        [
            HumanMessage(
                content=(
                    "Call add_numbers with a=7 and b=5. Do not calculate it yourself."
                )
            )
        ]
    )

    assert response.tool_calls, "No tool call returned"

    tool_call = response.tool_calls[0]

    assert tool_call["name"] == "add_numbers"
    assert tool_call["args"]["a"] == 7
    assert tool_call["args"]["b"] == 5

    print(f"\nTool call: {tool_call}")
    report_usage("tool_calling", response)


def test_structured_output(client: LLMClient):
    model = client.model.with_structured_output(
        CityInfo,
        method="json_mode",
        include_raw=True,
    )

    result = model.invoke(
        [
            SystemMessage(
                content=(
                    "Return valid JSON only. "
                    "Use exactly two fields: "
                    "city and country_code."
                )
            ),
            HumanMessage(
                content=(
                    "Return JSON for Berlin, Germany. "
                    "Set city to Berlin and country_code to DE."
                )
            ),
        ]
    )

    assert result["parsing_error"] is None

    parsed = result["parsed"]

    assert isinstance(parsed, CityInfo)
    assert parsed.city == "Berlin"
    assert parsed.country_code == "DE"

    print(f"\nStructured output: {parsed}")
    report_usage("structured_output", result["raw"])


def test_streaming(client: LLMClient):
    chunks = list(
        client.model.stream([HumanMessage(content="Say hello in one short sentence.")])
    )

    assert chunks, "No streaming chunks received"

    content = "".join(
        chunk.content for chunk in chunks if isinstance(chunk.content, str)
    )

    assert content.strip(), "Empty streaming response"

    print(f"\nStreaming response: {content}")
    print(f"Streaming chunks: {len(chunks)}")
