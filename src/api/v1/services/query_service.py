from src.api.v1.agents.agents import run_search_agent


def query_documents(query: str):
    print(query)
    return run_search_agent(query)
