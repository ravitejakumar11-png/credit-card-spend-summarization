from src.api.v1.agents.agents import (
    run_search_agent,
    run_search_agent_stream,
)


def query_documents(
    query: str,
    thread_id: str,
):
    print(query)
    print(f"thread_id: {thread_id}")

    return run_search_agent(
        query=query,
        thread_id=thread_id,
    )


async def query_documents_stream(
    query: str,
    thread_id: str | None = None,
):
    print(query)
    print(f"thread_id: {thread_id}")

    async for event in run_search_agent_stream(
        query=query,
        thread_id=thread_id,
    ):
        yield event
