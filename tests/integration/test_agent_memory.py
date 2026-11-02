import os
from uuid import uuid4

import pytest
from dotenv import load_dotenv
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from cvhub_agent.agent.factory import AgentFactory
from cvhub_agent.agent.middleware import build_agent_middleware
from cvhub_agent.agent.prompts import DEFAULT_SYSTEM_PROMPT
from cvhub_agent.config.settings import load_llm_config
from cvhub_agent.llm.client import LLMClient
from cvhub_agent.persistence.checkpointer import open_checkpointer
from cvhub_agent.tools.loader import load_tools


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_DEEPSEEK_TESTS") != "1",
    reason="Live DeepSeek tests are disabled",
)
async def test_agent_inmemorysaver():
    llm_config = load_llm_config()

    assert llm_config.provider == "deepseek"
    assert not llm_config.enable_thinking

    client = LLMClient(llm_config)

    thread_config: RunnableConfig = {
        "configurable": {"thread_id": "test-conversation-001"}
    }

    async with open_checkpointer("memory") as checkpointer:
        agent = AgentFactory(client.model).build(
            tools=load_tools(),
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            middleware=build_agent_middleware(),
            checkpointer=checkpointer,
        )

        # First turn
        await agent.ainvoke(
            {"messages": [HumanMessage(content="Remember this test code: ALPHA-42.")]},
            config=thread_config,
        )

        # Second turn
        response = await agent.ainvoke(
            {"messages": [HumanMessage(content="What was my test code?")]},
            config=thread_config,
        )

        # Verify memory
        final_message = response["messages"][-1]

        assert isinstance(final_message.content, str)
        assert "ALPHA-42" in final_message.content

        # Verify stored conversation
        snapshot = await agent.aget_state(thread_config)
        assert len(snapshot.values["messages"]) >= 4

        print("\n=== Agent Memory Test ===")
        print("Response:", final_message.content)
        print("Stored Messages:", len(snapshot.values["messages"]))


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_TESTS") != "1",
    reason="PostgreSQL integration tests are disabled",
)
async def test_agent_postgres_persistence():
    load_dotenv()

    database_url = os.getenv("AGENT_DATABASE_URL")

    if not database_url:
        raise ValueError("AGENT_DATABASE_URL is not configured.")

    thread_id = f"test-postgres-{uuid4().hex}"

    thread_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    try:
        # Session 1: Save conversation
        async with open_checkpointer(
            "postgres",
            database_url=database_url,
        ) as checkpointer:
            model = FakeListChatModel(responses=["Acknowledged ALPHA-42."])

            agent = AgentFactory(model).build(
                tools=[],
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                middleware=[],
                checkpointer=checkpointer,
            )

            await agent.ainvoke(
                {"messages": [HumanMessage(content="My test code is ALPHA-42.")]},
                config=thread_config,
            )

            snapshot = await agent.aget_state(thread_config)
            messages = snapshot.values["messages"]

            assert len(messages) == 2
            assert messages[0].content == ("My test code is ALPHA-42.")

        # Session 2: Reconnect and restore
        async with open_checkpointer(
            "postgres",
            database_url=database_url,
        ) as checkpointer:
            model = FakeListChatModel(responses=["Continuing conversation."])

            agent = AgentFactory(model).build(
                tools=[],
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                middleware=[],
                checkpointer=checkpointer,
            )

            # Verify restored history
            snapshot = await agent.aget_state(thread_config)
            messages = snapshot.values["messages"]

            assert len(messages) == 2
            assert messages[0].content == ("My test code is ALPHA-42.")

            # Continue conversation
            await agent.ainvoke(
                {"messages": [HumanMessage(content="Continue our conversation.")]},
                config=thread_config,
            )

            snapshot = await agent.aget_state(thread_config)
            messages = snapshot.values["messages"]

            assert len(messages) == 4
            assert messages[0].content == ("My test code is ALPHA-42.")
            assert messages[2].content == ("Continue our conversation.")

            print("\n=== PostgreSQL Memory Test ===")
            print("Thread ID:", thread_id)
            print("Stored Messages:", len(messages))

    finally:
        # Clean up only this test's checkpoint data
        async with open_checkpointer(
            "postgres",
            database_url=database_url,
        ) as checkpointer:
            await checkpointer.adelete_thread(thread_id)
