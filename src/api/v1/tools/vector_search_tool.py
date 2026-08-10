from src.api.v1.states.rag_state import RAGState
from src.core.db import search_vector_store


def vector_search_node(state: RAGState) -> RAGState:

    print("====== INSIDE VECTOR SEARCH NODE ======")

    search_query = state.get("retrieval_query") or state["query"]

    retrieval_attempt = state.get("retrieval_attempt", 0) + 1

    docs = search_vector_store(
        query=search_query,
        k=20,
    )

    print(f"Original Query      : {state['query']}")
    print(f"Retrieval Query     : {search_query}")
    print(f"Retrieval Attempt   : {retrieval_attempt}")
    print(f"Retrieved Documents : {len(docs)}")

    if docs:
        print(f"Top Match Similarity: " f"{docs[0].metadata.get('similarity')}")

        for i, doc in enumerate(docs[:20], start=1):
            print(f"  {i}. " f"Similarity: {doc.metadata.get('similarity')}")

    return {
        **state,
        "retrieved_docs": docs,
        "retrieval_attempt": retrieval_attempt,
    }
