import asyncio
import os

from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


async def main() -> None:
    # Load environment variables
    load_dotenv()

    database_url = os.getenv("AGENT_DATABASE_URL")

    if not database_url:
        raise ValueError("AGENT_DATABASE_URL is not configured.")

    # Initialize PostgreSQL checkpointer
    async with AsyncPostgresSaver.from_conn_string(database_url) as checkpointer:
        await checkpointer.setup()

    print("LangGraph checkpoint database initialized successfully.")


if __name__ == "__main__":
    asyncio.run(main())
