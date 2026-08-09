from typing import TypedDict, List
from langchain_core.documents import Document


# state for the agent
class RAGState(TypedDict):
    # the original user query text
    query: str

    route: str

    retrieved_docs: list

    reranked_docs: list

    generated_sql: str

    sql_result: str

    response: dict
