from src.api.v1.agents.agents import run_search_agent


def query_documents(query: str, thread_id: str | None = None):
    print(query)
    print(f"thread_id: {thread_id}")
    return run_search_agent(
        query=query,
        thread_id=thread_id,
    )
