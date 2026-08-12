from langchain_core.documents import Document
from src.api.v1.states.rag_state import RAGState
from src.core.db import (
    search_vector_store,
    fts_search,
)
from src.api.v1.services.query_cancellation import (
    raise_if_query_cancelled,
)


def vector_search_node(state: RAGState) -> RAGState:

    print("====== INSIDE VECTOR SEARCH NODE ======")

    thread_id = state.get("thread_id")

    # Check before starting the expensive operation.
    raise_if_query_cancelled(thread_id)

    search_query = state.get("retrieval_query") or state["query"]

    # retrieval_attempt = state.get("retrieval_attempt", 0) + 1

    docs = search_vector_store(
        query=search_query,
        k=20,
    )

    # Check again after vector search completes.
    raise_if_query_cancelled(thread_id)

    print(f"Original Query      : {state['query']}")

    print(f"Retrieval Query     : {search_query}")

    # print(f"Retrieval Attempt   : {retrieval_attempt}")

    print(f"Retrieved Documents : {len(docs)}")

    if docs:

        print("Top Match Similarity: " f"{docs[0].metadata.get('similarity')}")

        for i, doc in enumerate(
            docs[:20],
            start=1,
        ):
            print(f"  {i}. " f"Similarity: " f"{doc.metadata.get('similarity')}")

    return {"retrieved_docs": docs}


def search_fts_store(
    query: str,
    k: int = 20,
) -> list[Document]:
    """
    Perform PostgreSQL Full Text Search and return
    LangChain Documents.
    """

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    rows = fts_search(
        query=query,
        k=k,
    )

    documents = []

    for row in rows:

        documents.append(
            Document(
                page_content=row["content"],
                metadata={
                    "chunk_id": row.get("id"),
                    "source": row.get("source_file"),
                    "page": row.get("page_number"),
                    "section": row.get("section"),
                    "chunk_type": row.get("chunk_type"),
                    "element_type": row.get("element_type"),
                    "fts_rank": row.get("fts_rank"),
                    "image_base64": row.get("image_base64"),
                    "mime_type": row.get("mime_type"),
                    "position": row.get("position"),
                    "metadata": row.get("metadata"),
                },
            )
        )

    return documents


def fts_search_node(state: RAGState) -> RAGState:

    print("====== INSIDE FTS SEARCH NODE ======")

    thread_id = state.get("thread_id")

    # Check before starting the expensive operation.
    raise_if_query_cancelled(thread_id)

    search_query = (
        state.get("fts_query") or state.get("retrieval_query") or state["query"]
    )

    docs = search_fts_store(
        query=search_query,
        k=20,
    )

    # Check again after FTS search completes.
    raise_if_query_cancelled(thread_id)

    print(f"Original Query      : {state['query']}")

    print(f"FTS Query           : {search_query}")

    print(f"Retrieved Documents : {len(docs)}")

    if docs:

        print("Top FTS Rank: " f"{docs[0].metadata.get('fts_rank')}")

        for i, doc in enumerate(
            docs[:20],
            start=1,
        ):
            print(f"  {i}. " f"FTS Rank: " f"{doc.metadata.get('fts_rank')}")

    return {
        "fts_docs": docs,
        "retrieved_docs": docs,
    }


# ---------------------------------------------------------------------------
# Hybrid Vector + FTS Search using Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def search_vector_fts(
    vector_query: str,
    fts_query: str,
    k: int = 20,
) -> list[Document]:
    """
    Perform hybrid document retrieval:

    1. Vector semantic search
    2. FTS lexical search
    3. Reciprocal Rank Fusion (RRF)

    Returns LangChain Documents.
    """

    if not vector_query or not vector_query.strip():
        raise ValueError("Vector query cannot be empty.")

    if not fts_query or not fts_query.strip():
        raise ValueError("FTS query cannot be empty.")

    print("====== INSIDE VECTOR + FTS HYBRID SEARCH ======")

    # ---------------------------------------------------------
    # Execute both searches
    # ---------------------------------------------------------

    vector_docs = search_vector_store(
        query=vector_query,
        k=k,
    )

    fts_docs = search_fts_store(
        query=fts_query,
        k=k,
    )

    print(f"Vector Results : {len(vector_docs)}")

    print(f"FTS Results    : {len(fts_docs)}")

    # ---------------------------------------------------------
    # RRF fusion
    # ---------------------------------------------------------

    rrf_scores = {}

    document_map = {}

    def get_document_key(doc: Document):

        source = doc.metadata.get("source")

        page = doc.metadata.get("page")

        position = doc.metadata.get("position")

        if isinstance(position, dict):
            position = str(position)

        key = (
            source,
            page,
            position,
        )

        # fallback if metadata is insufficient
        if not any(key):
            return doc.page_content[:120]

        return key

    # ---------------------------------------------------------
    # Vector contribution
    # ---------------------------------------------------------

    for rank, doc in enumerate(
        vector_docs,
        start=1,
    ):

        key = get_document_key(doc)

        rrf_scores[key] = rrf_scores.get(key, 0) + (1 / (60 + rank))

        document_map[key] = doc

    # ---------------------------------------------------------
    # FTS contribution
    # ---------------------------------------------------------

    for rank, doc in enumerate(
        fts_docs,
        start=1,
    ):

        key = get_document_key(doc)

        rrf_scores[key] = rrf_scores.get(key, 0) + (1 / (60 + rank))

        document_map[key] = doc

    # ---------------------------------------------------------
    # Sort by RRF score
    # ---------------------------------------------------------

    ranked_documents = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    results = []

    for key, score in ranked_documents[:k]:

        doc = document_map[key]

        metadata = dict(doc.metadata)

        metadata.update(
            {
                "rrf_score": round(
                    score,
                    6,
                ),
                "score_type": "VECTOR_FTS",
            }
        )

        results.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
        )

    print(f"Hybrid Results : {len(results)}")

    if results:

        print("Top RRF Score: " f"{results[0].metadata.get('rrf_score')}")

    return results


def vector_fts_search_node(state: RAGState) -> RAGState:

    print("====== INSIDE VECTOR + FTS SEARCH NODE ======")

    thread_id = state.get("thread_id")

    #_check_cancelled(state)

    vector_query = state.get("retrieval_query") or state["query"]

    fts_query = state.get("fts_query") or state["query"]

    docs = search_vector_fts(
        vector_query=vector_query,
        fts_query=fts_query,
        k=20,
    )

    #_check_cancelled(state)

    print(f"Original Query : {state['query']}")

    print(f"Vector Query  : {vector_query}")

    print(f"FTS Query     : {fts_query}")

    print(f"Hybrid Documents Retrieved : {len(docs)}")

    if docs:

        print("Top RRF Score : " f"{docs[0].metadata.get('rrf_score')}")

        for i, doc in enumerate(
            docs[:10],
            start=1,
        ):

            print(f"{i}. " f"RRF Score: " f"{doc.metadata.get('rrf_score')}")

    return {
        "retrieved_docs": docs,
        "rrf_docs": docs,
    }
