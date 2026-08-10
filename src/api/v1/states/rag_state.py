from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


# state for the agent
class RAGState(TypedDict):
    # the original user query text
    messages: Annotated[list[AnyMessage], add_messages]
    query: str

    route: str

    retrieved_docs: list

    reranked_docs: list

    generated_sql: str

    sql_result: str

    response: dict
