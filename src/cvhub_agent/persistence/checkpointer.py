from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

CheckpointBackend = Literal["memory", "postgres"]


@asynccontextmanager
async def open_checkpointer(
    backend: CheckpointBackend,
    *,
    database_url: str | None = None,
) -> AsyncGenerator[BaseCheckpointSaver]:
    if backend == "memory":
        yield InMemorySaver()
        return

    if backend == "postgres":
        if not database_url or not database_url.strip():
            raise ValueError("database_url is required for PostgreSQL checkpointer.")

        async with AsyncPostgresSaver.from_conn_string(database_url) as checkpointer:
            yield checkpointer

        return

    raise ValueError(f"Unsupported checkpoint backend: {backend}")
