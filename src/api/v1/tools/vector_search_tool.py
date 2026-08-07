from src.api.v1.states.rag_state import RAGState
from src.core.db import search_vector_store


def vector_search_node(state: RAGState):

    print("====== INSIDE vector_search_node ======")

    docs = search_vector_store(
        query=state["query"],
        k=20,
    )

    print(f"Query                : {state['query']}")
    print(f"Retrieved Documents  : {len(docs)}")

    if docs:
        print(f"Top Match Similarity : {docs[0].metadata.get('similarity')}")
        print(f"Top Match Source     : {docs[0].metadata.get('source')}")
        print(f"Top Match Page       : {docs[0].metadata.get('page')}")

    return {
        **state,
        "retrieved_docs": docs,
    }
