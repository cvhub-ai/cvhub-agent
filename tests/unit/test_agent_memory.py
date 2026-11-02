import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from cvhub_agent.agent.factory import AgentFactory
from cvhub_agent.agent.prompts import DEFAULT_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_agent_thread_isolation():
    # Initialize fake LLM
    model = FakeListChatModel(
        responses=[
            "Acknowledged ALPHA-42.",
            "Acknowledged BETA-99.",
            "Continuing conversation A.",
        ]
    )

    # Shared checkpointer
    checkpointer = InMemorySaver()

    # Build a single agent
    agent = AgentFactory(model).build(
        tools=[],
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        middleware=[],
        checkpointer=checkpointer,
    )

    # Independent conversation configurations
    config_a: RunnableConfig = {"configurable": {"thread_id": "conversation-A"}}

    config_b: RunnableConfig = {"configurable": {"thread_id": "conversation-B"}}

    # --------------------------------------------------------
    # Thread A: First turn
    # --------------------------------------------------------
    await agent.ainvoke(
        {"messages": [HumanMessage(content="My test code is ALPHA-42.")]},
        config=config_a,
    )

    # --------------------------------------------------------
    # Thread B: First turn
    # --------------------------------------------------------
    await agent.ainvoke(
        {"messages": [HumanMessage(content="My test code is BETA-99.")]},
        config=config_b,
    )

    # --------------------------------------------------------
    # Verify independent states
    # --------------------------------------------------------
    state_a = await agent.aget_state(config_a)
    state_b = await agent.aget_state(config_b)

    messages_a = state_a.values["messages"]
    messages_b = state_b.values["messages"]

    assert len(messages_a) == 2
    assert len(messages_b) == 2

    assert messages_a[0].content == "My test code is ALPHA-42."
    assert messages_b[0].content == "My test code is BETA-99."

    # --------------------------------------------------------
    # Thread A: Second turn
    # --------------------------------------------------------
    await agent.ainvoke(
        {"messages": [HumanMessage(content="Continue our previous conversation.")]},
        config=config_a,
    )

    # --------------------------------------------------------
    # Verify conversation persistence
    # --------------------------------------------------------
    state_a = await agent.aget_state(config_a)
    state_b = await agent.aget_state(config_b)

    messages_a = state_a.values["messages"]
    messages_b = state_b.values["messages"]

    assert len(messages_a) == 4
    assert len(messages_b) == 2

    # Thread A retains its original messages.
    assert messages_a[0].content == "My test code is ALPHA-42."
    assert messages_a[2].content == ("Continue our previous conversation.")

    # Thread B remains unchanged.
    assert messages_b[0].content == "My test code is BETA-99."

    # No cross-thread message leakage.
    assert not any("BETA-99" in str(message.content) for message in messages_a)

    assert not any("ALPHA-42" in str(message.content) for message in messages_b)
