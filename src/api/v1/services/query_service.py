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


def query_documents_stream(
    query: str,
    thread_id: str,
):

    print(query)
    print(f"thread_id: {thread_id}")

    return run_search_agent_stream(
        query=query,
        thread_id=thread_id,
    )
