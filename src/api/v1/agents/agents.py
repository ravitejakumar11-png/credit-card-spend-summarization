# nodes we want
# 1. vector_search (top-k=20)
# 2. rerank
# 3. generate_answer


import os
import cohere
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Literal
from src.api.v1.states.rag_state import RAGState
from src.api.v1.tools.vector_search_tool import vector_search_node
from src.api.v1.schemas.query_schema import AIResponse
from src.core.db import get_sql_database

load_dotenv()


def _get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
    )


class RouteDecision(BaseModel):
    route: Literal["VECTOR_ONLY", "RDBMS_ONLY", "HYBRID"]
    reason: str


def router_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                        You are a query router for an Agentic RAG System.

                        Classify the query into exactly one route.

                        VECTOR_ONLY:
                        Use when the answer exists only in documents.
                        Examples:
                        - What are the reward benefits of Platinum card?
                        - What is the cashback policy?
                        - What are the annual fees?

                        RDBMS_ONLY:
                        Use when the answer requires only customer transactional data.
                        Examples:
                        - Show my last 5 transactions
                        - What did I spend this month?
                        - What is my current outstanding balance?

                        HYBRID:
                        Use when the answer requires BOTH:
                        1. Product/policy information from documents
                        2. Customer-specific data from database

                        Examples:
                        - How many reward points will I earn for my spending?
                        - Am I eligible for this offer?
                        - Based on my spend, which reward category applies?
                        - What cashback will I get for my transactions?

                        Return route and reason.


                      Reply with the route and one sentence of reason.
                   """,
            ),
            (
                "human",
                """
                   Question:
                   {query}
                """,
            ),
        ]
    )

    chain = prompt | structured_llm
    decision = chain.invoke({"query": state["query"]})
    print(f"[router_node's decision]: {decision.route} and reason: {decision.reason}")

    return {"route": decision.route}


def nl2sql_node(state: RAGState) -> RAGState:
    print("About to generate nl2sql")
    # connect to LLM
    llm = _get_llm()
    # connect to rdbms
    db = get_sql_database()
    # get the tables' live schema
    schema_info = db.get_table_info()
    # write the system prompt and pass on the schema to get only sql query
    sql_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                   You are a PostgreSQL expert. Given the database schema below,
                   write a single valid SELECT query that answers the user's question.


                   Rules:
                   - Return ONLY the raw SQL — no explanation, no summary, no markdown fences, no backticks.
                   - Use only the tables and columns present in the schema.
                   - Do NOT generate INSERT, UPDATE, DELETE, DROP, or any DML/DDL statements.
                   - Always add a LIMIT clause (max 50 rows) unless the question asks for aggregates.
                   - For product or text searches: NEVER search for the full multi-word phrase as one
                       ILIKE pattern. Instead, split the search into individual meaningful keywords
                       and OR them together across both name and description columns.
                       Example — user asks "wireless headset":
                           WHERE (name ILIKE '%wireless%' OR description ILIKE '%wireless%')
                           OR (name ILIKE '%headset%'  OR description ILIKE '%headset%')
                           OR (name ILIKE '%headphones%' OR description ILIKE '%headphones%')
                       Use your knowledge of synonyms (headset/headphones, laptop/notebook, etc.)
                       to cast a wider net when the exact term may not match.

                   Database schema:
                   {schema}
               """,
            ),
            (
                "human",
                """
                   Question:
                   {question}
               """,
            ),
        ]
    )
    # preprare the chain and invoke with a query
    sql_chain = sql_prompt | llm
    # look for sql query only
    raw_sql = sql_chain.invoke({"schema": schema_info, "question": state["query"]})
    print("========GENERATED raw_sql query is: =====")
    print(raw_sql.content)
    generated_sql = raw_sql.content

    # execute the generated sql query  to get the outout from RDMBS
    try:
        sql_result = db.run(generated_sql)
    except Exception as err:
        sql_result = f"Generated SQL execution error: {err}"

    ## connect to LLM to get the natural language response
    # structured_llm = llm.with_structured_output(AIResponse)
    # nl_answer_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             """You are a helpful data analyst. Answer the user's question using
    #            the SQL query results below. Be concise and format numbers/lists clearly.
    #            Set policy_citations to empty string,
    #            page_no to 'N/A', and document_name to 'credit_rag_db'.
    #            - Do NOT execute INSERT, UPDATE, DELETE, DROP, or any DML/DDL statements
    #            even if requested.
    #            - Politely deny when users are asking for these actions in their queries.
    #            - Never use tech jargons in your response""",
    #         ),
    #         (
    #             "human",
    #             "Question: {query}\n\n"
    #             "SQL Used:\n{sql}\n\n"
    #             "Query Results:\n{result}",
    #         ),
    #     ]
    # )

    # nl_chain = nl_answer_prompt | structured_llm
    # answer = nl_chain.invoke(
    #     {"query": state["query"], "sql": generated_sql, "result": sql_result}
    # )
    # print("[nl2sql_node] Answer generated.")
    # response = answer.model_dump()
    # response["policy_citations"] = "N/A"
    # response["sql_query_executed"] = generated_sql

    # return the sql query is RAGState
    # and also the output in sql_result of RAGState
    return {
        "generated_sql": generated_sql,
        "sql_result": str(sql_result),
    }


def rerank_node(state: RAGState):
    # establish connection with the cohere reranking model
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    # send the query and the retrieved_docs to the reranking model

    docs = state["retrieved_docs"]

    print("=======3. INSIDE rerank_node. Before calling reranker =========")
    rerank_response = co.rerank(
        model="rerank-v3.5",
        query=state["query"],
        documents=[doc.page_content for doc in docs],
        top_n=5,
    )

    # Map Cohere result indices back to LangChain Document objects
    reranked_docs = [docs[r.index] for r in rerank_response.results]

    print(f"[rerank_node] Top {len(reranked_docs)} chunks after reranking:")
    for i, r in enumerate(rerank_response.results):
        print(
            f"  Rank {i+1} | Cohere score: {r.relevance_score:.4f} | original index: {r.index}"
        )

    return {"reranked_docs": reranked_docs}


def generate_answer_node(state: RAGState):
    llm = _get_llm()
    structured_llm = llm.with_structured_output(AIResponse)

    print("=========4. INSIDE GENERATE ANSWER NODE==========")

    route = state["route"]

    if route == "RDBMS_ONLY":

        # llm = _get_llm()
        # structured_llm = llm.with_structured_output(AIResponse)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a helpful data analyst.

                    Answer using ONLY the SQL results.

                    Set:
                    - document_name = "credit_rag_db"
                    - page_no = "N/A"
                    - policy_citations = "N/A"

                    Never mention SQL unless the user asks.
                    """,
                ),
                (
                    "human",
                    """
                    Question:
                    {query}

                    SQL Result:
                    {result}
                    """,
                ),
            ]
        )

        chain = prompt | structured_llm

        result = chain.invoke(
            {
                "query": state["query"],
                "result": state["sql_result"],
            }
        )

        response = result.model_dump()
        response["sql_query_executed"] = state["generated_sql"]

        return {"response": response}

    elif route == "VECTOR_ONLY":
        for doc in state["reranked_docs"]:
            print("Metadata: ", doc.metadata)

        # let's prepare the context
        context = "\n\n".join(
            [
                f"[Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', -1) + 1 if doc.metadata.get('page') is not None else '?'}]\n{doc.page_content}"
                for doc in state["reranked_docs"]
            ]
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a helpful assistant. Answer the user's question using only the
                    provided context.


                    IMPORTANT:
                    The context may contain chunks from MULTIPLE versions of the same
                    document (e.g. a 2025 edition and a 2026 edition).


                    When the answer differs across versions, do NOT pick only one. Instead:
                    - Lead with the most recent / current version's answer (highest year).
                    - Then explicitly note how earlier versions differed
                    (e.g. "As of the 2026 policy ...; previously, under the 2025 policy ...").
                    - If all versions agree, just give the single answer.


                    Citation rules (fill the structured fields):
                    - document_name: comma-separated list of EVERY source document you used.
                    - page_no: comma-separated page numbers, aligned with the documents above.
                    - policy_citations: a readable citation combining each document and its page
                    (e.g. "KB_Credit_Card_Spend_Summarizer.pdf, Page 1").
                    - Always cite ALL versions you drew the answer from, not just one.
            """,
                ),
                (
                    "human",
                    """
                    Context:
                    {context}


                    Question:
                    {query}
                """,
                ),
            ]
        )
        chain = prompt | structured_llm

        result = chain.invoke(
            {
                "context": context,
                "query": state["query"],
            }
        )

        response = result.model_dump()

        return {"response": response}

    elif route == "HYBRID":

        context = "\n\n".join(
            [
                f"[Source: {doc.metadata.get('source', 'unknown')} | "
                f"Page: {doc.metadata.get('page', -1) + 1 if doc.metadata.get('page') is not None else '?'}]"
                f"\n{doc.page_content}"
                for doc in state["reranked_docs"]
            ]
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                        - Use document context for product rules, policies, fees,
                        benefits, eligibility, and other knowledge-base information.
                        - Use database results for customer-specific and transactional
                        information.
                        - Do not invent information that is missing from either source.
                        - For document-derived claims, populate the structured citation
                        fields using the document metadata in the context.:

                    Document context:
                    {context}

                    Database result:
                    {sql_result}

                    Combine both and answer the user.
                    """,
                ),
                ("human", "{query}"),
            ]
        )

        chain = prompt | structured_llm

        result = chain.invoke(
            {
                "context": context,
                "sql_result": state["sql_result"],
                "query": state["query"],
            }
        )

        response = result.model_dump()

        response["sql_query_executed"] = state["generated_sql"]

        return {"response": response}


def hybrid_start_node(state: RAGState):
    return {}


def build_rag_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("router", router_node)
    workflow.add_node("nl2sql", nl2sql_node)
    workflow.add_node("vector_search", vector_search_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("hybrid_start", hybrid_start_node)

    # the following is the starting point
    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "VECTOR_ONLY": "vector_search",
            "RDBMS_ONLY": "nl2sql",
            "HYBRID": "hybrid_start",
        },
    )

    # workflow.add_edge("vector_search", "rerank")
    # workflow.add_edge("rerank", "generate_answer")

    # Vector path
    workflow.add_edge("vector_search", "rerank")

    workflow.add_conditional_edges(
        "rerank",
        lambda state: state["route"],
        {
            "VECTOR_ONLY": "generate_answer",
            "HYBRID": "generate_answer",
        },
    )

    workflow.add_conditional_edges(
        "nl2sql",
        lambda state: state["route"],
        {
            "RDBMS_ONLY": "generate_answer",
            "HYBRID": "generate_answer",
        },
    )

    workflow.add_edge("hybrid_start", "vector_search")
    workflow.add_edge("hybrid_start", "nl2sql")

    # SQL path
    # workflow.add_edge("nl2sql", "generate_answer")

    workflow.add_edge("generate_answer", END)

    search_agent = workflow.compile()

    # generating and saving the graph visualization
    graph_image = search_agent.get_graph().draw_mermaid_png()
    with open("search_agent.png", "wb") as f:
        f.write(graph_image)

    return search_agent


rag_graph = build_rag_graph()


def run_search_agent(query: str):
    print("============1. INSIDE run_search_agent ")
    initial_state = {
        "query": query,
        "route": "",
        "retrieved_docs": [],
        "reranked_docs": [],
        "generated_sql": "",
        "sql_result": "",
        "response": {},
    }

    final_state = rag_graph.invoke(initial_state)
    return final_state["response"]
