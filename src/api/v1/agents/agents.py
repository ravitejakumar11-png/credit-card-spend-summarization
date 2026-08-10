# ============================================================================
# agent.py
# ============================================================================

import os
from typing import Literal, Optional

import cohere
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from pydantic import BaseModel

from src.api.v1.states.rag_state import RAGState
from src.api.v1.tools.vector_search_tool import vector_search_node
from src.api.v1.schemas.query_schema import AIResponse
from src.core.db import get_sql_database

# ============================================================================
# Environment
# ============================================================================

load_dotenv()


# ============================================================================
# LLM
# ============================================================================


def _get_llm():

    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ============================================================================
# Router Decision
# ============================================================================


class RouteDecision(BaseModel):

    route: Literal[
        "CHITCHAT",
        "VECTOR_ONLY",
        "RDBMS_ONLY",
        "HYBRID",
    ]

    reason: str

    # Used only when route == CHITCHAT
    answer: Optional[str] = None


# ============================================================================
# Router Node
# ============================================================================
#
# The router now performs TWO responsibilities:
#
# 1. Detect CHITCHAT
# 2. Route RAG questions to:
#       VECTOR_ONLY
#       RDBMS_ONLY
#       HYBRID
#
# If CHITCHAT is detected:
#
#       router -> END
#
# No other node is executed.
# ============================================================================


def router_node(state: RAGState):

    print("========== INSIDE ROUTER NODE ==========")

    llm = _get_llm()

    structured_llm = llm.with_structured_output(RouteDecision)

    query = state["query"]

    print(
        "[router_node] Current query:",
        query,
    )

    # ------------------------------------------------------------------------
    # Router Prompt
    # ------------------------------------------------------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the first-stage router for a Credit Card
Agentic RAG System.

Your job is to classify the user's CURRENT message.

You MUST select exactly ONE of these routes:

1. CHITCHAT
2. VECTOR_ONLY
3. RDBMS_ONLY
4. HYBRID


======================================================================
CHITCHAT
======================================================================

Use CHITCHAT when the user is having normal casual conversation
and the request does NOT require documents or customer database
information.

Examples:

- Hi
- Hello
- Hey
- Good morning
- Good afternoon
- How are you?
- How are you doing?
- What can you do?
- Who are you?
- Thanks
- Thank you
- Thank you so much
- Okay
- Great
- Nice
- Bye
- Goodbye
- See you later
- Have a nice day

CHITCHAT also includes simple conversational acknowledgements
that do not require document or database information.

Examples:

User: Thanks
User: That's helpful
User: Okay
User: Got it

For CHITCHAT:

- Do NOT use vector search.
- Do NOT use the database.
- Do NOT use SQL.
- Do NOT use document retrieval.
- Do NOT use reranking.
- Do NOT use generate_answer.
- Return a short, natural conversational response in `answer`.

The `answer` field MUST contain the response that should be
returned directly to the user.

Examples:

User:
Hi

Route:
CHITCHAT

Answer:
Hello! How can I help you today?


User:
How are you?

Route:
CHITCHAT

Answer:
I'm doing well, thank you! How can I help you with your
credit card today?


User:
Thanks

Route:
CHITCHAT

Answer:
You're welcome! Let me know if you need anything else.


======================================================================
VECTOR_ONLY
======================================================================

Use VECTOR_ONLY when the answer exists only in the uploaded
documents / knowledge base.

Examples:

- What are the reward benefits of Platinum card?
- What is the cashback policy?
- What are the annual fees?
- What is the late payment fee?
- What are the eligibility requirements?
- What are the benefits of this card?
- What is the reward points policy?
- What is the card's annual fee?


======================================================================
RDBMS_ONLY
======================================================================

Use RDBMS_ONLY when the answer requires only
customer-specific transactional/database information.

Examples:

- Show my last 5 transactions
- What did I spend this month?
- What is my current outstanding balance?
- Show my recent payments
- How much did I spend last month?
- What is my current balance?
- Show my transactions
- How much have I spent?


======================================================================
HYBRID
======================================================================

Use HYBRID when the answer requires BOTH:

1. Product/policy information from documents
2. Customer-specific information from the database

Examples:

- How many reward points will I earn for my spending?
- Am I eligible for this offer?
- Based on my spend, which reward category applies?
- What cashback will I get for my transactions?
- Based on my spending, which card benefit applies?
- Which reward benefit applies to my transactions?


======================================================================
IMPORTANT ROUTING RULE
======================================================================

First determine whether the message is CHITCHAT.

If it is normal casual conversation, return CHITCHAT.

If it is a question about credit card policies, benefits,
fees, eligibility, rewards, or other knowledge-base information,
use VECTOR_ONLY.

If it requires customer-specific transaction or account data,
use RDBMS_ONLY.

If it requires BOTH document information and customer-specific
database information, use HYBRID.

Do not classify a genuine credit-card information request
as CHITCHAT merely because it is phrased conversationally.

For example:

"Hi, what is the annual fee for my Platinum card?"

This is NOT CHITCHAT.

It should be:

VECTOR_ONLY

Similarly:

"Hey, how much did I spend this month?"

This is NOT CHITCHAT.

It should be:

RDBMS_ONLY.


======================================================================
CHITCHAT ANSWER
======================================================================

When route == CHITCHAT:

Generate a short, helpful response directly in the `answer`
field.

Do not answer any credit-card policy, transaction, balance,
reward, eligibility, or database question in the CHITCHAT answer.

For RAG questions, set `answer` to null.


Return:

- route
- reason
- answer
""",
            ),
            (
                "human",
                """
Current user question:

{query}
""",
            ),
        ]
    )

    chain = prompt | structured_llm

    decision = chain.invoke(
        {
            "query": query,
        }
    )

    print(f"[router_node] Decision: {decision.route}")

    print(f"[router_node] Reason: {decision.reason}")

    # ------------------------------------------------------------------------
    # CHITCHAT
    # ------------------------------------------------------------------------
    #
    # The router itself generates the response.
    #
    # The graph will then route directly to END.
    #
    # No other RAG node will execute.
    # ------------------------------------------------------------------------

    if decision.route == "CHITCHAT":

        answer = decision.answer or "Hello! How can I help you today?"

        print("[router_node] CHITCHAT detected.")

        print(
            "[router_node] Direct answer:",
            answer,
        )

        return {
            "route": "CHITCHAT",
            "response": {
                "query": query,
                "answer": answer,
                "document_name": "N/A",
                "page_no": "N/A",
                "policy_citations": "N/A",
                "sql_query_executed": None,
            },
        }

    # ------------------------------------------------------------------------
    # RAG routes
    # ------------------------------------------------------------------------

    return {
        "route": decision.route,
    }


# ============================================================================
# NL2SQL Node
# ============================================================================


def nl2sql_node(state: RAGState) -> RAGState:

    print("========== INSIDE NL2SQL NODE ==========")

    llm = _get_llm()

    db = get_sql_database()

    # ------------------------------------------------------------------------
    # Get live database schema
    # ------------------------------------------------------------------------

    schema_info = db.get_table_info()

    # ------------------------------------------------------------------------
    # SQL generation prompt
    # ------------------------------------------------------------------------

    sql_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a PostgreSQL expert.

Given the database schema below, write a single valid
SELECT query that answers the user's question.

Rules:

- Return ONLY the raw SQL.

- No explanation.

- No summary.

- No markdown fences.

- No backticks.

- Use only the tables and columns present in the schema.

- Do NOT generate:
  INSERT
  UPDATE
  DELETE
  DROP
  ALTER
  TRUNCATE
  or any other DML/DDL.

- Always add a LIMIT clause with a maximum of 50 rows
  unless the question asks for an aggregate.

- For product or text searches, NEVER search for the
  entire multi-word phrase as one ILIKE pattern.

- Split the search into individual meaningful keywords.

- Search across relevant name and description columns.

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

    sql_chain = sql_prompt | llm

    raw_sql = sql_chain.invoke(
        {
            "schema": schema_info,
            "question": state["query"],
        }
    )

    generated_sql = raw_sql.content

    print("========== GENERATED SQL ==========")

    print(generated_sql)

    # ------------------------------------------------------------------------
    # Execute SQL
    # ------------------------------------------------------------------------

    try:

        sql_result = db.run(generated_sql)

    except Exception as err:

        sql_result = f"Generated SQL execution error: {err}"

    print("========== SQL RESULT ==========")

    print(sql_result)

    return {
        "generated_sql": generated_sql,
        "sql_result": str(sql_result),
    }


# ============================================================================
# Rerank Node
# ============================================================================


def rerank_node(state: RAGState):

    print("========== INSIDE RERANK NODE ==========")

    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    docs = state["retrieved_docs"]

    print(f"[rerank_node] Received {len(docs)} documents")

    if not docs:

        print("[rerank_node] No documents to rerank.")

        return {"reranked_docs": []}

    rerank_response = co.rerank(
        model="rerank-v3.5",
        query=state["query"],
        documents=[doc.page_content for doc in docs],
        top_n=5,
    )

    reranked_docs = [docs[r.index] for r in rerank_response.results]

    print(f"[rerank_node] Top " f"{len(reranked_docs)} chunks after reranking:")

    for i, r in enumerate(rerank_response.results):

        print(
            f"Rank {i + 1} | "
            f"Cohere score: "
            f"{r.relevance_score:.4f} | "
            f"Original index: {r.index}"
        )

    return {"reranked_docs": reranked_docs}


# ============================================================================
# Generate Answer Node
# ============================================================================


def generate_answer_node(state: RAGState):

    print("========== INSIDE GENERATE ANSWER NODE ==========")

    llm = _get_llm()

    structured_llm = llm.with_structured_output(AIResponse)

    route = state["route"]

    # ========================================================================
    # RDBMS ONLY
    # ========================================================================

    if route == "RDBMS_ONLY":

        print("[generate_answer_node] RDBMS_ONLY")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a helpful data analyst.

Answer using ONLY the SQL results.

Set:

document_name = "credit_rag_db"

page_no = "N/A"

policy_citations = "N/A"

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

    # ========================================================================
    # VECTOR ONLY
    # ========================================================================

    elif route == "VECTOR_ONLY":

        print("[generate_answer_node] VECTOR_ONLY")

        for doc in state["reranked_docs"]:

            print("Metadata:", doc.metadata)

        context = "\n\n".join(
            [
                (
                    f"[Source: "
                    f"{doc.metadata.get('source', 'unknown')} "
                    f"| Page: "
                    f"{doc.metadata.get('page', -1) + 1 if doc.metadata.get('page') is not None else '?'}]"
                    f"\n{doc.page_content}"
                )
                for doc in state["reranked_docs"]
            ]
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a helpful assistant.

Answer the user's question using ONLY the
provided context.

The context may contain chunks from MULTIPLE
versions of the same document.

When the answer differs across versions:

- Lead with the most recent/current version.
- Explicitly note how earlier versions differed.
- If all versions agree, provide the single answer.

Citation rules:

- document_name:
  comma-separated list of EVERY source document used.

- page_no:
  comma-separated page numbers aligned with the documents.

- policy_citations:
  readable citation combining each document and page.

- Always cite ALL versions used.
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

    # ========================================================================
    # HYBRID
    # ========================================================================

    elif route == "HYBRID":

        print("[generate_answer_node] HYBRID")

        context = "\n\n".join(
            [
                (
                    f"[Source: "
                    f"{doc.metadata.get('source', 'unknown')} "
                    f"| Page: "
                    f"{doc.metadata.get('page', -1) + 1 if doc.metadata.get('page') is not None else '?'}]"
                    f"\n{doc.page_content}"
                )
                for doc in state["reranked_docs"]
            ]
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the NorthStar Credit Card Assistant.

Use document context for:

- product rules
- policies
- fees
- benefits
- eligibility
- knowledge-base information

Use database results for:

- customer-specific information
- transactional information

Do not invent information missing from either source.

For document-derived claims, populate the structured
citation fields using the document metadata.

Document context:

{context}

Database result:

{sql_result}

Combine both sources and answer the user.
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

    # ========================================================================
    # Unexpected route
    # ========================================================================

    else:

        raise ValueError(f"Unsupported route: {route}")


# ============================================================================
# Hybrid Start Node
# ============================================================================


def hybrid_start_node(state: RAGState):

    print("========== INSIDE HYBRID START NODE ==========")

    return {}


# ============================================================================
# Build RAG Graph
# ============================================================================


def build_rag_graph():

    workflow = StateGraph(RAGState)

    # ------------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------------

    workflow.add_node("router", router_node)

    workflow.add_node("nl2sql", nl2sql_node)

    workflow.add_node("vector_search", vector_search_node)

    workflow.add_node("rerank", rerank_node)

    workflow.add_node("generate_answer", generate_answer_node)

    workflow.add_node("hybrid_start", hybrid_start_node)

    # ------------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------------

    workflow.set_entry_point("router")

    # ------------------------------------------------------------------------
    # Router -> Route
    # ------------------------------------------------------------------------
    #
    # CHITCHAT goes directly to END.
    #
    # It does NOT execute:
    #
    # vector_search
    # rerank
    # nl2sql
    # hybrid_start
    # generate_answer
    #
    # ------------------------------------------------------------------------

    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "CHITCHAT": END,
            "VECTOR_ONLY": "vector_search",
            "RDBMS_ONLY": "nl2sql",
            "HYBRID": "hybrid_start",
        },
    )

    # ------------------------------------------------------------------------
    # Vector path
    # ------------------------------------------------------------------------

    workflow.add_edge("vector_search", "rerank")

    workflow.add_conditional_edges(
        "rerank",
        lambda state: state["route"],
        {
            "VECTOR_ONLY": "generate_answer",
            "HYBRID": "generate_answer",
        },
    )

    # ------------------------------------------------------------------------
    # SQL path
    # ------------------------------------------------------------------------

    workflow.add_conditional_edges(
        "nl2sql",
        lambda state: state["route"],
        {
            "RDBMS_ONLY": "generate_answer",
            "HYBRID": "generate_answer",
        },
    )

    # ------------------------------------------------------------------------
    # Hybrid path
    # ------------------------------------------------------------------------

    workflow.add_edge("hybrid_start", "vector_search")

    workflow.add_edge("hybrid_start", "nl2sql")

    # ------------------------------------------------------------------------
    # Final node
    # ------------------------------------------------------------------------

    workflow.add_edge("generate_answer", END)

    # ------------------------------------------------------------------------
    # Checkpointer
    # ------------------------------------------------------------------------

    memory = InMemorySaver()

    search_agent = workflow.compile(checkpointer=memory)

    # ------------------------------------------------------------------------
    # Generate graph visualization
    # ------------------------------------------------------------------------

    try:

        graph_image = search_agent.get_graph().draw_mermaid_png()

        with open("search_agent.png", "wb") as f:

            f.write(graph_image)

    except Exception as exc:

        print("[build_rag_graph] " f"Could not generate graph image: {exc}")

    return search_agent


# ============================================================================
# Build graph once
# ============================================================================

rag_graph = build_rag_graph()


# ============================================================================
# Run Search Agent
# ============================================================================


def run_search_agent(query: str, thread_id: str | None = None):

    print("============ INSIDE run_search_agent ============")

    normalized_thread_id = thread_id or "default-thread"

    print(f"[run_search_agent] " f"thread_id={normalized_thread_id}")

    # ------------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------------
    #
    # No conversational state is required.
    #
    # The router itself handles CHITCHAT.
    # ------------------------------------------------------------------------

    initial_state = {
        "query": query,
        "route": "",
        "retrieved_docs": [],
        "reranked_docs": [],
        "generated_sql": "",
        "sql_result": "",
        "response": {},
    }

    # ------------------------------------------------------------------------
    # Checkpoint configuration
    # ------------------------------------------------------------------------

    config = {"configurable": {"thread_id": normalized_thread_id}}

    # ------------------------------------------------------------------------
    # Invoke graph
    # ------------------------------------------------------------------------

    final_state = rag_graph.invoke(
        initial_state,
        config=config,
    )

    print("============ AGENT COMPLETED ============")

    print("[run_search_agent] Final route:", final_state.get("route"))

    return final_state["response"]
