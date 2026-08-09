from src.api.v1.states.rag_state import RAGState
from src.core.db import search_vector_store


def vector_search_node(state: RAGState):
    print("====== INSIDE vector_search_node: searching the vector db")
    # search_vector_store(query, k) returns a list of LangChain Documents
    docs = search_vector_store(state["query"], k=20)
    print(
        "======= INSIDE vector_search_node: Searched the Vector DB - Retrieved Docs Count:",
        len(docs),
    )
    return {"retrieved_docs": docs}
