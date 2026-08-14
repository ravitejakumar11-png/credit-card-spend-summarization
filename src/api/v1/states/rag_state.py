from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    thread_id: str
    knowledge_strategy: str
    query: str
    route: str
    retrieval_query: str
    fts_query: str
    vector_docs: list
    fts_docs: list
    rrf_docs: list
    retrieval_attempt: int
    retrieved_docs: list
    reranked_docs: list
    generated_sql: str
    sql_result: str
    sql_context: str
    vector_context: str
    final_context: str
    response: dict
    evaluation: str
    evaluation_feedback: str
    evaluate_count: int
    user_preferences: str
    user_id: str
