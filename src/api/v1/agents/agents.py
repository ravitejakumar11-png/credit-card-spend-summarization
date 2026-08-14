import asyncio
import json
import os

import cohere
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.config import get_stream_writer
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Literal
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from src.api.v1.states.rag_state import RAGState
from src.api.v1.tools.tools import (
    vector_search_node,
    fts_search_node,
    vector_fts_search_node,
)
from src.api.v1.schemas.query_schema import AIResponse
from src.core.db import get_sql_database
from src.core.db import get_cached_schema
from src.api.v1.services.query_cancellation import (
    raise_if_query_cancelled,
    clear_query_cancellation,
    QueryCancelled,
)
from mem0 import MemoryClient

load_dotenv()


def _check_cancelled(state: RAGState) -> None:
    thread_id = state.get("thread_id")

    raise_if_query_cancelled(thread_id)


# ============================================================================
# USER-FACING STREAMING
# ============================================================================


def _emit_progress(message: str) -> None:
    """
    Emit a small, user-facing progress update.

    IMPORTANT:
        This intentionally exposes only high-level progress.

    It does NOT expose:
        - routing decisions
        - retrieval queries
        - similarity scores
        - retrieval attempts
        - reranking
        - query reformulation
        - SQL
        - evaluation
        - regeneration
        - internal node names
    """

    try:

        writer = get_stream_writer()

        writer(
            {
                "event": "progress",
                "message": message,
            }
        )

    except Exception:
        # Streaming is an optional UI feature.
        #
        # If the graph is being executed using normal invoke()
        # rather than stream(), progress events should never break
        # the actual RAG pipeline.
        pass


def _emit_stream_event(event: dict) -> None:
    """
    Emit a user-facing streaming event through LangGraph's custom stream.

    This is intentionally best-effort so the normal non-streaming graph
    execution is completely unaffected.
    """
    try:
        writer = get_stream_writer()
        writer(event)
    except Exception:
        pass


def _emit_token(content: str) -> None:
    if content:
        _emit_stream_event(
            {
                "event": "token",
                "content": content,
            }
        )


def _emit_stream_reset() -> None:
    _emit_stream_event({"event": "reset"})


def _get_router_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )


def _get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )


def _get_evaluator_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )


client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))


# ============================================================================
# STRUCTURED OUTPUT MODELS
# ============================================================================


class RouteDecision(BaseModel):
    route: Literal[
        "VECTOR_DB",
        "RDBMS",
        "HYBRID",
        "DIRECT",
    ]
    reason: str
    direct_response: str


class KnowledgeStrategyDecision(BaseModel):
    knowledge_strategy: Literal[
        "VECTOR",
        "FTS",
        "VECTOR_FTS",
    ]

    reason: str

    retrieval_query: str

    fts_query: str


class RetrievalQueryDecision(BaseModel):
    retrieval_query: str

    fts_query: str


class EvaluationDecision(BaseModel):
    evaluation: Literal["PASS", "REGENERATE"]
    feedback: str


class UserPreferenceDecision(BaseModel):
    is_preference: bool
    preference: str


def save_user_preference_to_mem0(
    preference: str,
    user_id: str,
) -> None:

    if not preference:
        return

    try:

        messages = [
            {
                "role": "user",
                "content": preference,
            }
        ]

        client.add(
            messages,
            user_id=user_id,
        )

        print("[Mem0] Preference saved successfully: " f"{preference}")

    except Exception as exc:

        # Memory failure should NOT break the RAG pipeline.
        print("[Mem0] Failed to save preference: " f"{exc}")


def retrieve_user_preferences_from_mem0(
    query: str,
    user_id: str,
    top_k: int = 5,
) -> str:
    """
    Retrieve the user's previously stored preferences from Mem0.

    The search is always scoped to the authenticated user_id so memories
    from another user can never be used as personalization context.

    Returns a compact newline-separated string suitable for an LLM prompt.
    Memory retrieval is best-effort and must never break the RAG pipeline.
    """

    if not user_id:
        return ""

    try:
        results = client.search(
            query=(
                "User preferences, likes, dislikes, habits, choices, "
                "defaults, and constraints relevant to this request. "
                f"Current request: {query}"
            ),
            filters={"user_id": user_id},
            top_k=top_k,
        )

        # Mem0 Platform responses can be returned either as a list or as an
        # envelope containing a `results` list depending on SDK/API version.
        if isinstance(results, dict):
            memories = results.get("results", [])
        else:
            memories = results or []

        preferences = []
        seen = set()

        for item in memories:
            if isinstance(item, dict):
                memory = item.get("memory", "")
            else:
                memory = getattr(item, "memory", "")

            if not memory:
                continue

            memory = str(memory).strip()
            if not memory or memory in seen:
                continue

            seen.add(memory)
            preferences.append(memory)

        if not preferences:
            print(f"[Mem0] No stored preferences found for user_id={user_id}")
            return ""

        formatted = "\n".join(f"- {preference}" for preference in preferences)

        print(
            "[Mem0] Retrieved "
            f"{len(preferences)} preference(s) for user_id={user_id}"
        )

        return formatted

    except Exception as exc:
        print("[Mem0] Failed to retrieve preferences: " f"{exc}")
        return ""


def detect_user_preference(
    query: str,
    history: list,
) -> UserPreferenceDecision:

    llm = _get_router_llm()

    structured_llm = llm.with_structured_output(UserPreferenceDecision)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a user-preference detection component for the
NorthStar Credit Card Assistant.

Your task is to determine whether the user's current message
contains information that should be remembered as a long-term
user preference, like, dislike, habit, choice, or personal
preference.

A preference should be stored when the user explicitly or
implicitly tells the assistant something they like, dislike,
prefer, usually do, or want the assistant to remember.

Examples that SHOULD be remembered:

- "I prefer cashback cards."
- "I like earning travel rewards."
- "I don't like annual-fee cards."
- "I prefer Visa cards."
- "I usually use my card for dining."
- "I prefer to redeem my points for flights."
- "I don't want recommendations involving annual fees."
- "I prefer concise answers."
- "I like airport lounge benefits."

Examples that SHOULD NOT be remembered:

- "What are the benefits of this card?"
- "How many points did I earn?"
- "What is my balance?"
- "What is the annual fee?"
- "Tell me about lounge access."
- "Hello."
- "What can you do?"
- Temporary instructions that only apply to the current request.

Important rules:

1. Only extract information actually stated or clearly implied
   by the user.
2. Do not invent preferences.
3. Do not store ordinary questions as preferences.
4. Do not store sensitive personal information.
5. Keep the preference concise and written as a statement
   about the user.
6. If multiple preferences are expressed, combine them into
   one concise preference string.
7. If there is no preference, return an empty string.
8. Do not answer the user's question.

Return:

- is_preference
- preference
""",
            ),
            (
                "human",
                """
Conversation History:

{history}

Current User Message:

{query}
""",
            ),
        ]
    )

    chain = prompt | structured_llm

    return chain.invoke(
        {
            "history": history,
            "query": query,
        }
    )


def memory_node(state: RAGState) -> RAGState:
    """
    Manage long-term user memory before route selection.

    Flow:
        1. Detect a new preference in the current user message.
        2. Save the new preference to Mem0.
        3. Search Mem0 for the authenticated user's existing preferences.
        4. Put the retrieved preferences into RAGState.UserPreference.

    The retrieved preference string is later supplied separately to the
    answer generator and evaluator as personalization context.
    """

    _check_cancelled(state)

    print("========= INSIDE MEMORY NODE =========")

    query = state["query"]
    history = state.get("messages", [])
    user_id = state.get("user_id")

    if not user_id:
        print(
            "[memory_node] No authenticated user_id found. " "Skipping Mem0 operations."
        )
        return {"UserPreference": ""}

    # ------------------------------------------------------------
    # 1. Detect a newly expressed preference.
    # ------------------------------------------------------------

    preference_decision = detect_user_preference(
        query=query,
        history=history,
    )

    _check_cancelled(state)

    if (
        preference_decision.is_preference
        and preference_decision.preference
        and preference_decision.preference.strip()
    ):
        new_preference = preference_decision.preference.strip()

        print("[memory_node] New user preference detected: " f"{new_preference}")

        save_user_preference_to_mem0(
            preference=new_preference,
            user_id=user_id,
        )
    else:
        print("[memory_node] No new user preference detected.")

    _check_cancelled(state)

    # ------------------------------------------------------------
    # 2. Retrieve existing preferences for this authenticated user.
    # ------------------------------------------------------------

    user_preferences = retrieve_user_preferences_from_mem0(
        query=query,
        user_id=user_id,
        top_k=5,
    )

    _check_cancelled(state)

    return {
        "UserPreference": user_preferences,
    }


# ============================================================================
# ROUTER
# ============================================================================


def router_node(state: RAGState) -> RAGState:

    _check_cancelled(state)

    print("========= INSIDE ROUTER NODE =========")

    _emit_progress("Understanding your question...")

    history = state.get("messages", [])
    query = state["query"]

    llm = _get_router_llm()

    structured_llm = llm.with_structured_output(RouteDecision)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the Query Router for the NorthStar Credit Card Agentic RAG System.

Determine which source, if any, is required to answer the user's question.

Choose EXACTLY ONE route:

1. VECTOR_DB

Use when the answer requires information from the credit-card knowledge base, including:

- Card features, benefits, rewards, cashback, lounge access
- Fees, interest, billing, EMI, eligibility
- Policies, FAQs, terms & conditions
- Product documentation or bank rules

2. RDBMS

Use when the answer requires only structured data from the relational database, including:

- Customer, card, account or transaction data
- Spending, payments, statements, balances or credit limits
- Reward points or transaction history
- Counts, summaries or other customer-specific data

RDBMS is for actual customer-specific data, not generic questions about
how the product or reward program works.

Use RDBMS when the question asks for actual or historical customer data,
such as:

- what the customer spent
- what the customer earned
- transaction history
- balances
- payments
- statements
- customer/card details

A generic question about earning rates, benefits, eligibility, fees,
rules, or policies is VECTOR_DB even if phrased using "I", "my", or "me".

Examples:

- "How many points do I earn for dining?" → VECTOR_DB
- "How many points did I earn for dining last month?" → RDBMS
- "What is the dining reward rate?" → VECTOR_DB

3. HYBRID

Use when BOTH the relational database and knowledge base are required.

Examples include questions requiring customer-specific data together with:

- Reward rules
- Fee or interest policies
- Billing rules
- Spend categorization rules
- Other product or policy information

Use HYBRID only when the answer genuinely requires BOTH:

1. actual customer/transaction data
2. knowledge-base rules or policies

Do not use HYBRID merely because the question mentions rewards,
spending, fees, or another topic that exists in both sources.

4. DIRECT

Use when no database or knowledge-base retrieval is required.

This includes:

- Greetings and simple conversation
- Questions about the assistant's capabilities
- General chit-chat
- Questions unrelated to banking, credit cards, transactions or rewards
- Conversational follow-ups that require no new information from the
  database or knowledge base.

For DIRECT:

- Provide a brief natural response in `direct_response` for greetings,
  capabilities and simple conversation.
- For unrelated questions, provide a brief polite refusal explaining
  that the assistant is designed for NorthStar credit-card and related
  topics.
- Do not retrieve from the RDBMS or VECTOR_DB.

Routing rules:

- Return exactly ONE route.
- Use HYBRID whenever both structured data and knowledge-base information
  are required.
- Use retrieval routes only when retrieval is necessary.
- Use conversation history to resolve references and determine whether
  the current question is related to the ongoing conversation.
- If unsure between a single retrieval source and HYBRID, choose HYBRID.
- For VECTOR_DB, RDBMS and HYBRID, set `direct_response` to an empty string.

Return:

- route
- reason
- direct_response
""",
            ),
            (
                "human",
                """
Conversation History:

{history}

Current User Question:

{query}
""",
            ),
        ]
    )

    chain = prompt | structured_llm

    decision = chain.invoke(
        {
            "query": state["query"],
            "history": state.get("messages", []),
        }
    )

    _check_cancelled(state)

    if decision.route == "RDBMS":

        _emit_progress("Checking your account information...")

    elif decision.route == "VECTOR_DB":

        _emit_progress("Checking the relevant card information...")

    elif decision.route == "HYBRID":

        _emit_progress("Checking your account information...")

        _emit_progress("Checking the relevant card information...")

    print(f"[router_node] Route : {decision.route}")

    print(f"[router_node] Reason: {decision.reason}")

    return {
        "route": decision.route,
        "response": {
            "query": state["query"],
            "answer": decision.direct_response,
            "policy_citations": "",
            "page_no": "",
            "document_name": "",
            "sql_query_executed": None,
        },
    }


# ============================================================================
# HYBRID START / JOIN
# ============================================================================


def hybrid_start_node(state: RAGState) -> RAGState:

    print("========= INSIDE HYBRID START NODE =========")

    return {}


def hybrid_join_node(state: RAGState) -> RAGState:

    print("========= INSIDE HYBRID JOIN NODE =========")

    return {}


# ============================================================================
# KNOWLEDGE STRATEGY ROUTER
# ============================================================================


def knowledge_strategy_router_node(state: RAGState) -> RAGState:

    print("====== INSIDE KNOWLEDGE STRATEGY ROUTER ======")

    _check_cancelled(state)

    _emit_progress("Understanding the question...")

    llm = _get_router_llm()

    structured_llm = llm.with_structured_output(KnowledgeStrategyDecision)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a retrieval strategy classifier for a banking knowledge base.

Your task is to decide the best document retrieval strategy.

Available strategies:

1. VECTOR

Use VECTOR when:

- The user is asking for explanation, reasoning, summary, or conceptual understanding.
- The answer requires semantic understanding.
- The wording may not exactly match the document.

Examples:
"What are the benefits of this credit card?"
"Explain reward redemption rules"
"How does billing work?"

2. FTS

Use FTS when:

- Exact words, phrases, identifiers, names, codes, or specific terms matter.
- The user is looking for an exact mention in documents.

Examples:
"Find the section mentioning NEFT"
"Where is Statement Credit mentioned?"
"Find Platinum card"

3. VECTOR_FTS

Use VECTOR_FTS when:

- The query needs both semantic understanding and exact terminology matching.
- The question is complex or policy/rule based.
- Missing either semantic or keyword retrieval may reduce recall.

Examples:
"What are the eligibility rules for accelerated dining rewards?"
"Explain reward points conversion and redemption options"
"What is MCC category and how does NorthStar classify transactions?"
"Explain the dining reward eligibility rules for accelerated points"
"What are the redemption options and their point conversion values?"
"Tell me about Platinum card lounge benefits"

After selecting the retrieval strategy, create the appropriate search queries.

The queries serve different purposes.

================================================

retrieval_query:

Used for vector similarity search.

Rules:

- Write a standalone semantic search query.
- Preserve the user's intent.
- Add useful banking or credit-card terminology.
- Expand concepts where helpful.
- Do not answer the question.

Example:

User:
"What are reward redemption options?"

retrieval_query:
"NorthStar credit card reward points redemption options,
conversion values, and redemption methods"

================================================

fts_query:

Used for PostgreSQL full-text search.

Rules:

- Extract important lexical terms.
- Preserve exact phrases, acronyms, product names,
  codes, and identifiers.
- Remove conversational words.
- Do not create explanations.
- Do not answer the question.

Example:

User:
"What is MCC?"

fts_query:
"MCC Merchant Category Code"

================================================

For VECTOR strategy:

Generate:

- retrieval_query

Set:

- fts_query as empty string

For FTS strategy:

Generate:

- fts_query

Set:

- retrieval_query as empty string

For VECTOR_FTS strategy:

Generate BOTH.

Example:

User:
"Explain reward points conversion and redemption options"

knowledge_strategy:
VECTOR_FTS

retrieval_query:
"NorthStar credit card reward points conversion,
redemption options, redemption values and methods"

fts_query:
"reward points conversion redemption options
Statement Credit Partner Vouchers Airline Miles"

Important:
FTS performs lexical matching.
Use FTS when the user likely expects exact terminology,
names, phrases, identifiers, or direct references.

Do not choose FTS only because the question contains
common words like "credit", "card", "reward", or "payment".

Rules:

- Return ONLY JSON.
- Do not add explanations.

Output format:

Return ONLY JSON.

Output format:

{{
"knowledge_strategy": "VECTOR | FTS | VECTOR_FTS",
"reason": "short explanation",
"retrieval_query": "semantic search query for vector retrieval",
"fts_query": "lexical search query for full text search"
}}
""",
            ),
            (
                "human",
                """
Conversation History:

{history}

Current User Question:

{query}
""",
            ),
        ]
    )

    chain = prompt | structured_llm

    decision = chain.invoke(
        {
            "query": state["query"],
            "history": state.get("messages", []),
        }
    )

    _check_cancelled(state)

    strategy = decision.knowledge_strategy

    if strategy not in (
        "VECTOR",
        "FTS",
        "VECTOR_FTS",
    ):
        strategy = "VECTOR_FTS"

    print(f"[knowledge_strategy_router_node] Route : {strategy}")

    print(f"Retrieval Query : {decision.retrieval_query}")

    print(f"FTS Query       : {decision.fts_query}")

    return {
        "knowledge_strategy": strategy,
        "retrieval_query": decision.retrieval_query,
        "fts_query": decision.fts_query,
    }


# ============================================================================
# QUERY OPTIMIZATION / REFORMULATION
# ============================================================================


def query_optimization_node(state: RAGState) -> RAGState:

    _check_cancelled(state)

    print("========= INSIDE QUERY REFORMULATION NODE =========")

    llm = _get_router_llm()

    structured_llm = llm.with_structured_output(RetrievalQueryDecision)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You rewrite user questions into effective retrieval queries
for the NorthStar Credit Card knowledge base.

The retrieval strategy is:

{knowledge_strategy}

Generate search queries according to the strategy.

Rules:

- Do not answer the question.
- Preserve the user's intent.
- Do not invent facts, policies, names, amounts, dates, or rules.
- Only improve retrieval effectiveness.

================================================

VECTOR strategy:

Create only a semantic retrieval query.

The query should:

- capture the meaning of the question
- add useful banking or credit-card terminology
- expand concepts when helpful

Example:

Question:
"What are reward redemption options?"

retrieval_query:
"NorthStar credit card reward points redemption options,
conversion values, and redemption methods"

================================================

FTS strategy:

Create only a lexical search query.

The query should:

- preserve exact terms
- preserve acronyms
- preserve product names
- remove conversational words

Remove words like:
what, why, how, explain, find, section, mentioned

Example:

Question:
"What is MCC?"

fts_query:
"MCC Merchant Category Code"

Question:
"Where is Statement Credit mentioned?"

fts_query:
"Statement Credit"

================================================

VECTOR_FTS strategy:

Create BOTH queries.

retrieval_query:

- semantic query for vector similarity search
- include context and meaning

fts_query:

- keyword-focused query for PostgreSQL FTS
- preserve exact document terminology

Example:

Question:
"Explain reward points conversion and redemption options"

retrieval_query:
"NorthStar credit card reward points conversion,
redemption options, redemption values and methods"

fts_query:
"reward points conversion redemption options
Statement Credit Partner Vouchers Airline Miles"

================================================

Return only the structured output.
""",
            ),
            (
                "human",
                """
Conversation History:

{history}

Current Question:

{query}

Retrieval Strategy:

{knowledge_strategy}
""",
            ),
        ]
    )

    chain = prompt | structured_llm

    result = chain.invoke(
        {
            "history": state.get("messages", []),
            "query": state["query"],
            "knowledge_strategy": state.get(
                "knowledge_strategy",
                "VECTOR",
            ),
        }
    )

    _check_cancelled(state)

    print("========= QUERY REFORMULATION =========")

    print(f"Original Query       : {state['query']}")

    print(f"Retrieval Query      : {result.retrieval_query}")

    print(f"FTS Query            : {result.fts_query}")

    return {
        "retrieval_query": result.retrieval_query,
        "fts_query": result.fts_query,
    }


# ============================================================================
# NL2SQL
# ============================================================================


def nl2sql_node(state: RAGState) -> RAGState:

    _check_cancelled(state)

    print("========= INSIDE NL2SQL NODE =========")

    llm = _get_llm()

    db = get_sql_database()

    schema_info = get_cached_schema()

    sql_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert PostgreSQL SQL generator for a conversational
NorthStar Credit Card Assistant.

Generate ONE valid PostgreSQL SELECT query that answers the current
user question using the database schema, business rules and
conversation history.

## Conversation and Entity Resolution

- Resolve references such as he, she, they, this customer, this card,
  this transaction, the same one, or the previous one using conversation
  history.
- Preserve entities, identifiers and filters established in previous
  turns.
- If conversation history identifies the target entity, do not broaden
  the query to unrelated entities.
- Do not ask for clarification when the target can be resolved from the
  available history.

## SQL Rules

- Use only tables and columns present in the supplied schema.

- Generate only the SQL required to answer the question.

- Return ONLY raw SQL. No explanation, markdown or code fences.

- SELECT statements only.

- Never generate INSERT, UPDATE, DELETE, DROP or other DML/DDL.

- Add LIMIT 50 unless the query is an aggregate.

- For multi-word text searches, search meaningful keywords separately
  rather than requiring the entire phrase to match.

- Use appropriate synonyms when useful for text searches.

Reward points have multiple meanings in the database:

- credit_cards.reward_points:
  Current reward-point balance on the card.

- card_transactions.reward_pts_earned:
  Reward points attributed to individual card transactions.

- reward_transactions.points_earned:
  Reward points posted through the reward ledger.

Do not treat these fields as interchangeable.

When aggregating data from multiple one-to-many tables, never directly
join the detail tables and aggregate both sides in the same SELECT.
Aggregate each detail table separately first, then join the aggregated
results.

Avoid double-counting caused by joining multiple transaction-level tables.

## Final Validation

Before returning SQL, verify:

1. It answers the current question.
2. Conversation references are resolved correctly.
3. Required filters are present.
4. The query does not unnecessarily broaden the dataset.
5. The SQL is valid PostgreSQL.

Database schema:

{schema}
""",
            ),
            (
                "human",
                """
Conversation History:

{history}

Current User Question:

{question}
""",
            ),
        ]
    )

    sql_chain = sql_prompt | llm

    history = state.get("messages", [])

    raw_sql = sql_chain.invoke(
        {
            "schema": schema_info,
            "history": history,
            "question": state["query"],
        }
    )

    _check_cancelled(state)

    generated_sql = raw_sql.content.strip()

    print("======== GENERATED SQL QUERY ========")

    print(generated_sql)

    _check_cancelled(state)

    try:

        sql_result = db.run(generated_sql)

        _check_cancelled(state)

    except Exception as err:

        sql_result = f"Generated SQL execution error: {err}"

    print("========= NL2SQL NODE OUTPUT =========")

    print("\nSQL Result:")

    print(str(sql_result))

    print("======================================")

    # IMPORTANT:
    #
    # Do not return **state here.
    #
    # This node can execute in parallel with vector retrieval for HYBRID.
    # Returning only fields produced by this node avoids unnecessarily
    # overwriting state produced by the vector branch.

    return {
        "generated_sql": generated_sql,
        "sql_result": str(sql_result),
    }


def route_after_nl2sql(state: RAGState) -> str:

    if state["route"] == "HYBRID":

        return "HYBRID"

    return "RDBMS"


# ============================================================================
# RERANK
# ============================================================================


def rerank_node(state: RAGState) -> RAGState:

    _check_cancelled(state)

    print("========= INSIDE RERANK NODE =========")

    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    docs = state.get("retrieved_docs", [])

    if not docs:

        print("[rerank_node] No documents to rerank.")

        return {
            "reranked_docs": [],
        }

    search_query = state.get("retrieval_query") or state["query"]

    print("======= BEFORE CALLING RERANKER =======")

    print(f"Search query: {search_query}")

    print(f"Documents   : {len(docs)}")

    rerank_response = co.rerank(
        model="rerank-v3.5",
        query=search_query,
        documents=[doc.page_content for doc in docs],
        top_n=5,
    )

    _check_cancelled(state)

    reranked_docs = [docs[r.index] for r in rerank_response.results]

    print(f"[rerank_node] Top {len(reranked_docs)} " "chunks after reranking:")

    for i, r in enumerate(rerank_response.results):

        print(
            f"  Rank {i + 1} | "
            f"Cohere score: {r.relevance_score:.4f} | "
            f"original index: {r.index}"
        )

    return {
        "reranked_docs": reranked_docs,
    }


def route_after_rerank(state: RAGState) -> str:

    if state["route"] == "HYBRID":

        return "HYBRID"

    return "VECTOR_DB"


# ============================================================================
# MERGE CONTEXT
# ============================================================================


def merge_context_node(
    state: RAGState,
) -> RAGState:
    """
    Merge SQL results and Vector search results into one context
    for answer generation.

    This node does NOT call an LLM.
    """

    _check_cancelled(state)

    print("========= INSIDE MERGE CONTEXT NODE =========")

    _emit_progress("Putting the information together...")

    route = state.get("route")

    # ------------------------------------------------------------------------
    # SQL CONTEXT
    # ------------------------------------------------------------------------

    sql_context = ""

    if route in ("RDBMS", "HYBRID") and state.get("sql_result"):

        sql_context = f"""
==========================
STRUCTURED CUSTOMER DATA
==========================

SQL Query Executed:
{state.get("generated_sql", "")}

SQL Result:
{state.get("sql_result", "")}
""".strip()

    # ------------------------------------------------------------------------
    # VECTOR CONTEXT
    # ------------------------------------------------------------------------

    vector_context = ""

    if route in ("VECTOR_DB", "HYBRID") and state.get("reranked_docs"):

        vector_chunks = []

        for doc in state["reranked_docs"]:

            source = doc.metadata.get(
                "source",
                "Unknown Document",
            )

            page = doc.metadata.get("page")

            page_no = page + 1 if page is not None else "Unknown"

            vector_chunks.append(f"""
Source : {source}
Page   : {page_no}

{doc.page_content}
""".strip())

        vector_context = (
            "==========================\n"
            "KNOWLEDGE BASE\n"
            "==========================\n\n"
            + "\n\n----------------------------------------\n\n".join(vector_chunks)
        )

    contexts = []

    if sql_context:

        contexts.append(sql_context)

    if vector_context:

        contexts.append(vector_context)

    final_context = "\n\n".join(contexts)

    print(f"Route                  : {route}")

    print("SQL Context Present    : " f"{bool(sql_context)}")

    print("Vector Context Present : " f"{bool(vector_context)}")

    return {
        "sql_context": sql_context,
        "vector_context": vector_context,
        "final_context": final_context,
    }


def generate_answer_node(state: RAGState, config=None) -> RAGState:
    _check_cancelled(state)

    print("========= INSIDE GENERATE ANSWER NODE =========")

    llm = _get_llm()

    structured_llm = llm.with_structured_output(AIResponse)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the NorthStar Credit Card Assistant.

Answer the user's question using ONLY the supplied context.

The context may contain:

1. STRUCTURED CUSTOMER DATA

This may contain:

- customer information
- card information
- transactions
- spending
- payments
- balances
- reward points
- transaction history
- other customer-specific data

2. KNOWLEDGE BASE

This may contain:

- card benefits
- reward rules
- fees
- eligibility
- policies
- billing rules
- terms and conditions
- product information

Use the source that is relevant to the user's question.

If both structured customer data and knowledge-base information
are present and both are relevant, combine them.

Do not invent information.

Answer using only the supplied context.

Preserve the exact meaning and scope of the source.

Do not generalize, infer, or expand policy statements.

If the source says an exclusion applies to a specific portion,
condition, transaction type, or circumstance, apply the exclusion
only to that scope.

Do not convert:

"X portion is excluded"

into:

"X is excluded."

Do not convert:

"X transactions are excluded"

into:

"all X spending is excluded."

When a table and an explanatory note appear together, interpret them
together and preserve both the earning rule and its exceptions.

Use conversation history to resolve references such as:

- he
- she
- they
- this customer
- this card
- this transaction
- the same one
- the previous one
- what about that
- how about that

Do not lose entities established in previous turns.

- Be concise and business-friendly.
- Use bullets when useful.
- Explain numerical results clearly.
- Do not mention SQL, databases, vector search, retrieval, prompts,
  internal systems or internal reasoning.
- Do not expose internal processing details.
- If the context does not contain enough information to answer,
  clearly say that the available information is insufficient.

Populate:

- answer
- document_name
- page_no
- policy_citations
- sql_query_executed

For database-only answers:

- document_name may be empty.
- page_no may be empty.
- policy_citations may be empty.

For knowledge-base answers:

- populate document_name and page_no when available.
- populate policy_citations when appropriate.

For HYBRID answers:

- populate both database-related and document-related fields
  when applicable.
""",
            ),
            (
                "human",
                """
Conversation History:

{history}

Current User Question:

{query}

Retrieved Context:

{context}

User Preference Memory (personalization context only):

{user_preferences}

Important: User Preference Memory describes the user's preferences,
likes, dislikes, habits, or defaults. It may be used to personalize
recommendations and presentation, but it is NOT an authoritative source
for balances, transactions, fees, reward rules, policies, eligibility,
or any other factual product/customer information. Retrieved Context
remains the factual authority for those claims.

Previous Evaluation Feedback (if any):

{feedback}
""",
            ),
        ]
    )

    # ------------------------------------------------------------------------
    # STREAMING ANSWER PATH
    #
    # The normal path below remains unchanged and uses structured output.
    # For the streaming endpoint, the answer itself must come from the
    # underlying ChatOpenAI stream.  A structured-output wrapper is useful for
    # correctness/metadata, but it does not expose the answer text as token
    # events through our current graph stream.
    #
    # We therefore:
    #   1. stream the natural-language answer from the base LLM internally;
    #   2. collect the complete generated answer without exposing it to the UI;
    #   3. run the existing structured-output call to populate metadata;
    #   4. overwrite the structured response's answer with the exact generated
    #      text so evaluation sees exactly what was generated;
    #   5. only after the graph reaches its final evaluated state does the
    #      streaming entry point expose the approved answer to the UI.
    #
    # The metadata call is intentionally kept because document/page/citation
    # and SQL fields are part of the existing functionality.
    # ------------------------------------------------------------------------

    is_streaming = bool(
        config and config.get("configurable", {}).get("streaming", False)
    )

    history = state.get("messages", [])

    if is_streaming:
        _check_cancelled(state)
        _emit_progress("Preparing your answer...")

        # Use the same source-accuracy instructions as the existing prompt,
        # but explicitly request answer text only for the live stream.
        stream_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the NorthStar Credit Card Assistant.

Answer the user's question using ONLY the supplied context.

Preserve the exact meaning and scope of the source. Do not invent,
generalize, infer, or expand policy statements. Preserve exceptions,
qualifiers, transaction scopes, dates, amounts, and other constraints.

Use conversation history to resolve references such as "this customer",
"this card", "the previous one", or similar follow-ups.

Be concise and business-friendly. Use bullets when useful. Explain
numerical results clearly.

Do not mention SQL, databases, vector search, retrieval, prompts,
internal systems, or internal reasoning.

If the supplied context is insufficient, clearly say that the available
information is insufficient.

IMPORTANT:
Return ONLY the natural-language answer to the user's question.
Do not return JSON. Do not return field names such as answer,
document_name, page_no, policy_citations, or sql_query_executed.
""",
                ),
                (
                    "human",
                    """
Conversation History:

{history}

Current User Question:

{query}

Retrieved Context:

{context}

Previous Evaluation Feedback (if any):

{feedback}
""",
                ),
            ]
        )

        stream_chain = stream_prompt | llm

        streamed_answer_parts = []

        for chunk in stream_chain.stream(
            {
                "query": state["query"],
                "context": state.get("final_context", ""),
                "history": history,
                "feedback": state.get(
                    "evaluation_feedback",
                    "",
                ),
            }
        ):
            _check_cancelled(state)

            content = getattr(chunk, "content", "")

            # ChatOpenAI normally returns a string here. Be defensive about
            # providers/chunk formats that may return a list of content blocks.
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(
                    item.get("text", "") for item in content if isinstance(item, dict)
                )
            else:
                text = str(content) if content else ""

            if text:
                # Buffer the generated answer internally.
                # Do NOT expose these chunks to the UI yet because this
                # answer still has to pass through evaluate_answer_node().
                streamed_answer_parts.append(text)

        streamed_answer = "".join(streamed_answer_parts).strip()

        _check_cancelled(state)

        # Populate the existing structured response fields without allowing
        # the second call to replace the exact text already streamed to UI.
        metadata_prompt = ChatPromptTemplate.from_messages(
            [
                *prompt.messages,
                (
                    "human",
                    """
The generated answer is:

{streamed_answer}

Use this exact text as the `answer` field. Do not rewrite it.
Populate the remaining response fields from the supplied context.
""",
                ),
            ]
        )

        metadata_chain = metadata_prompt | structured_llm

        metadata_result = metadata_chain.invoke(
            {
                "query": state["query"],
                "context": state.get("final_context", ""),
                "history": history,
                "feedback": state.get(
                    "evaluation_feedback",
                    "",
                ),
                "streamed_answer": streamed_answer,
            }
        )

        _check_cancelled(state)

        response = metadata_result.model_dump()
        response["answer"] = streamed_answer

        if state.get("generated_sql"):
            response["sql_query_executed"] = state["generated_sql"]

        print("[generate_answer_node] Streaming answer generated.")

        return {
            "response": response,
        }

    chain = prompt | structured_llm

    history = state.get("messages", [])

    result = chain.invoke(
        {
            "query": state["query"],
            "context": state.get(
                "final_context",
                "",
            ),
            "history": history,
            "user_preferences": state.get(
                "UserPreference",
                "",
            ),
            "feedback": state.get(
                "evaluation_feedback",
                "",
            ),
        }
    )

    _check_cancelled(state)

    response = result.model_dump()

    # Preserve the SQL query generated during this turn.
    #
    # This is useful for RDBMS and HYBRID responses.

    if state.get("generated_sql"):

        response["sql_query_executed"] = state["generated_sql"]

    print("[generate_answer_node] Answer generated.")

    return {
        "response": response,
    }


# ============================================================================
# EVALUATE ANSWER
# ============================================================================


def evaluate_answer_node(
    state: RAGState,
) -> RAGState:

    _check_cancelled(state)

    print("========= INSIDE EVALUATE ANSWER NODE =========")
    # if state.get("evaluate_count", 0) == 0:

    #     _emit_progress("Preparing your answer...")

    print("========= ANSWER BEING EVALUATED =========")

    print(state["response"])

    evaluate_count = (
        state.get(
            "evaluate_count",
            0,
        )
        + 1
    )

    print(f"Evaluation attempt number: " f"{evaluate_count}")

    llm = _get_evaluator_llm()

    structured_llm = llm.with_structured_output(EvaluationDecision)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the answer evaluator for a credit-card RAG system.

Evaluate the generated answer against the retrieved context.

PASS if:

1. The answer is supported by the retrieved context.
2. The answer does not contradict the retrieved context.
3. Important qualifiers, conditions, exceptions and scope are preserved.
4. The answer directly addresses the user's question.

REGENERATE if:

1. The answer contradicts the retrieved context.
2. The answer makes an unsupported claim.
3. The answer omits a condition that materially changes the meaning.
4. The answer fails to answer the question.

IMPORTANT:

Use only the retrieved context as the factual authority.

Do not substitute general knowledge for retrieved information.

Do not broaden or narrow the scope of a policy statement.

Evaluate strictly against the retrieved context and preserve the exact
scope of qualifiers, conditions and exceptions.

Do not broaden a specific exclusion into a broader exclusion.

The evaluator must not blindly trust numerical results produced by SQL.
Check for obvious aggregation inconsistencies, duplicated counts,
conflicting metrics, or misuse of fields when the retrieved context
contains enough information to identify the correct metric.

Preserve qualifiers such as:

- only
- except
- portion
- subject to
- up to
- minimum
- maximum
- per transaction
- per statement
- per month
- per year

When the source contains a specific exception, preserve that exception
rather than generalizing it to the entire category.
""",
            ),
            (
                "human",
                """
Conversation History:

{history}

Current Question:

{query}

Retrieved Context:

{context}

User Preference Memory (personalization context only):

{user_preferences}

Important: User Preference Memory may explain the user's preferences
or desired personalization. It must NOT be used as factual evidence
for balances, transactions, fees, reward rules, policies, eligibility,
or other product/customer facts. Evaluate factual claims against the
Retrieved Context. Use preferences only to determine whether the answer
appropriately respects the user's stated preferences when relevant.

Generated Answer:

{answer}

Previous Evaluation Feedback:

{feedback}
""",
            ),
        ]
    )

    chain = prompt | structured_llm

    history = state.get("messages", [])

    answer = state.get(
        "response",
        {},
    ).get(
        "answer",
        "",
    )

    print("========= ANSWER SENT TO EVALUATOR =========")

    print(answer)

    result = chain.invoke(
        {
            "query": state["query"],
            "context": state.get(
                "final_context",
                "",
            ),
            "history": history,
            "user_preferences": state.get(
                "UserPreference",
                "",
            ),
            "answer": answer,
            "feedback": state.get(
                "evaluation_feedback",
                "",
            ),
        }
    )

    _check_cancelled(state)

    print("========= EVALUATOR RESULT =========")

    print(f"[evaluate_answer_node] " f"{result.evaluation}")

    print(result)

    # ----------------------------------------------------------
    # Only persist the assistant answer when evaluation is final.
    #
    # If evaluation #1 says REGENERATE, do NOT add the answer
    # to conversation history because it will be replaced.
    #
    # If evaluation PASSes, or evaluation #2 fails and we stop,
    # the current answer becomes the final answer for this turn.
    # ----------------------------------------------------------

    update = {
        "evaluation": result.evaluation,
        "evaluation_feedback": result.feedback,
        "evaluate_count": evaluate_count,
    }

    is_final_answer = result.evaluation == "PASS" or evaluate_count >= 2

    if is_final_answer:

        answer = state.get(
            "response",
            {},
        ).get(
            "answer",
            "",
        )

        if answer:

            update["messages"] = [AIMessage(content=answer)]

    return update


# ============================================================================
# RETRIEVAL QUALITY
# ============================================================================


def should_reformulate_after_search(
    state: RAGState,
) -> bool:

    docs = state.get(
        "retrieved_docs",
        [],
    )

    attempt = state.get(
        "retrieval_attempt",
        0,
    )

    strategy = state.get(
        "knowledge_strategy",
        "VECTOR",
    )

    # ------------------------------------------------------------------
    # Never reformulate more than once
    # ------------------------------------------------------------------

    if attempt >= 2:

        return False

    # ------------------------------------------------------------------
    # No results
    # ------------------------------------------------------------------

    if not docs:

        print(
            "[retrieval_quality] "
            f"No documents found for strategy {strategy}. "
            "Reformulating query."
        )

        return True

    # ------------------------------------------------------------------
    # Strategy-specific scoring
    # ------------------------------------------------------------------

    if strategy == "VECTOR":

        score_key = "similarity"

        threshold = 0.50

    elif strategy == "FTS":

        score_key = "fts_rank"

        # FTS scores are much smaller than vector similarity scores.
        # Tune this based on your observed PostgreSQL ts_rank values.

        threshold = 0.05

    elif strategy == "VECTOR_FTS":

        # RRF scores are lower magnitude than vector similarity.
        # Tune threshold based on observed reciprocal rank values.
        score_key = "rrf_score"
        threshold = 0.01

    else:

        print(
            "[retrieval_quality] "
            f"Unknown strategy {strategy}. "
            "Defaulting to VECTOR scoring."
        )

        score_key = "similarity"

        threshold = 0.50

    # ------------------------------------------------------------------
    # Calculate relevant documents
    # ------------------------------------------------------------------

    relevant_docs = [
        doc
        for doc in docs
        if (
            doc.metadata.get(score_key) is not None
            and doc.metadata.get(score_key) >= threshold
        )
    ]

    top_score = docs[0].metadata.get(score_key)

    print(
        "[retrieval_quality] "
        f"Strategy: {strategy} | "
        f"Top {score_key}: {top_score} | "
        f"Relevant docs: {len(relevant_docs)} | "
        f"Attempt: {attempt}"
    )

    # ------------------------------------------------------------------
    # Quality decision
    # ------------------------------------------------------------------

    if len(relevant_docs) < 3:

        print(
            "[retrieval_quality] " "Retrieval quality is low. " "Reformulating query."
        )

        return True

    print(
        "[retrieval_quality] "
        "Retrieval quality is sufficient. "
        "Skipping reformulation."
    )

    return False


def route_after_retrieval(
    state: RAGState,
) -> str:

    if should_reformulate_after_search(state):

        return "OPTIMIZE"

    return "CONTINUE"


# ============================================================================
# ROUTING HELPERS
# ============================================================================


def route_after_evaluation(
    state: RAGState,
) -> str:

    if state.get("evaluation") == "PASS":

        return "PASS"

    if (
        state.get(
            "evaluate_count",
            0,
        )
        == 1
    ):

        return "REGENERATE"

    return "END"


def route_knowledge_strategy(state: RAGState):

    strategy = state.get("knowledge_strategy")

    if strategy == "VECTOR":

        return "VECTOR"

    if strategy == "FTS":

        return "FTS"

    if strategy == "VECTOR_FTS":

        return "VECTOR_FTS"

    return "VECTOR"


def route_after_reformulation(
    state: RAGState,
):

    strategy = state.get("knowledge_strategy")

    print(f"[route_after_reformulation] " f"Strategy: {strategy}")

    if strategy == "FTS":

        return "fts_search"

    if strategy == "VECTOR_FTS":

        return "vector_fts_search"

    return "vector_search"


def increment_retrieval_attempt_node(
    state: RAGState,
) -> RAGState:

    attempt = (
        state.get(
            "retrieval_attempt",
            0,
        )
        + 1
    )

    print(f"====== RETRIEVAL ATTEMPT {attempt} ======")

    return {"retrieval_attempt": attempt}


# ============================================================================
# HYBRID COMPLETION NODES
# ============================================================================


def hybrid_vector_done_node(
    state: RAGState,
) -> RAGState:

    print("========= HYBRID VECTOR BRANCH COMPLETE =========")

    return {}


def hybrid_sql_done_node(
    state: RAGState,
) -> RAGState:

    print("========= HYBRID SQL BRANCH COMPLETE =========")

    return {}


# ============================================================================
# BUILD RAG GRAPH
# ============================================================================


def build_rag_graph():

    workflow = StateGraph(RAGState)

    # ------------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------------

    # Memory / personalization

    workflow.add_node(
        "memory",
        memory_node,
    )

    # Primary routing

    workflow.add_node(
        "router",
        router_node,
    )

    # ------------------------------------------------------------------------
    # Existing Hybrid (Vector + RDBMS)
    # ------------------------------------------------------------------------

    workflow.add_node(
        "hybrid_start",
        hybrid_start_node,
    )

    workflow.add_node(
        "hybrid_join",
        hybrid_join_node,
    )

    workflow.add_node(
        "hybrid_vector_done",
        hybrid_vector_done_node,
    )

    workflow.add_node(
        "hybrid_sql_done",
        hybrid_sql_done_node,
    )

    # ------------------------------------------------------------------------
    # RDBMS
    # ------------------------------------------------------------------------

    workflow.add_node(
        "nl2sql",
        nl2sql_node,
    )

    # ------------------------------------------------------------------------
    # Knowledge Retrieval Layer
    # ------------------------------------------------------------------------

    workflow.add_node(
        "knowledge_strategy_router",
        knowledge_strategy_router_node,
    )

    workflow.add_node(
        "vector_search",
        vector_search_node,
    )

    workflow.add_node(
        "fts_search",
        fts_search_node,
    )

    workflow.add_node(
        "vector_fts_search",
        vector_fts_search_node,
    )

    workflow.add_node(
        "increment_retrieval_attempt",
        increment_retrieval_attempt_node,
    )

    # ------------------------------------------------------------------------
    # Retrieval Improvement
    # ------------------------------------------------------------------------

    workflow.add_node(
        "query_optimization",
        query_optimization_node,
    )

    workflow.add_node(
        "rerank",
        rerank_node,
    )

    # ------------------------------------------------------------------------
    # Answer Generation
    # ------------------------------------------------------------------------

    workflow.add_node(
        "merge_context",
        merge_context_node,
    )

    workflow.add_node(
        "generate_answer",
        generate_answer_node,
    )

    workflow.add_node(
        "evaluate_answer",
        evaluate_answer_node,
    )

    # ------------------------------------------------------------------------
    # Entry Point
    # ------------------------------------------------------------------------

    workflow.set_entry_point("memory")

    # Memory always runs before routing.
    workflow.add_edge("memory", "router")

    # ------------------------------------------------------------------------
    # PRIMARY ROUTER
    #
    # Decides:
    # VECTOR_DB
    # RDBMS
    # HYBRID
    # DIRECT
    # ------------------------------------------------------------------------

    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "VECTOR_DB": "knowledge_strategy_router",
            "RDBMS": "nl2sql",
            "HYBRID": "hybrid_start",
            "DIRECT": END,
        },
    )

    # ------------------------------------------------------------------------
    # KNOWLEDGE RETRIEVAL STRATEGY ROUTER
    #
    # Decides:
    # VECTOR
    # FTS
    # VECTOR_FTS
    # ------------------------------------------------------------------------

    workflow.add_conditional_edges(
        "knowledge_strategy_router",
        route_knowledge_strategy,
        {
            "VECTOR": "vector_search",
            "FTS": "fts_search",
            "VECTOR_FTS": "vector_fts_search",
        },
    )

    # ------------------------------------------------------------------------
    # EXISTING VECTOR + RDBMS HYBRID FLOW
    # ------------------------------------------------------------------------

    workflow.add_edge(
        "hybrid_start",
        "knowledge_strategy_router",
    )

    workflow.add_edge(
        "hybrid_start",
        "nl2sql",
    )

    # ------------------------------------------------------------------------
    # NL2SQL ROUTING
    # ------------------------------------------------------------------------

    workflow.add_conditional_edges(
        "nl2sql",
        route_after_nl2sql,
        {
            "RDBMS": "merge_context",
            "HYBRID": "hybrid_sql_done",
        },
    )

    # ------------------------------------------------------------------------
    # RETRIEVAL FLOW
    #
    # Search
    #    |
    # Increment attempt
    #    |
    # Decide:
    #       Reformulate
    #       Continue
    #
    # ------------------------------------------------------------------------

    workflow.add_edge(
        "vector_search",
        "increment_retrieval_attempt",
    )

    workflow.add_edge(
        "fts_search",
        "increment_retrieval_attempt",
    )

    workflow.add_edge(
        "vector_fts_search",
        "increment_retrieval_attempt",
    )

    workflow.add_conditional_edges(
        "increment_retrieval_attempt",
        route_after_retrieval,
        {
            "OPTIMIZE": "query_optimization",
            "CONTINUE": "rerank",
        },
    )

    # ------------------------------------------------------------------------
    # QUERY REFORMULATION RETRY
    #
    # Important:
    # Retry the SAME retrieval strategy
    #
    # VECTOR  -> vector_search
    # FTS     -> fts_search
    # VECTOR_FTS -> vector_fts_search
    #
    # ------------------------------------------------------------------------

    workflow.add_conditional_edges(
        "query_optimization",
        route_after_reformulation,
        {
            "vector_search": "vector_search",
            "fts_search": "fts_search",
            "vector_fts_search": "vector_fts_search",
        },
    )

    # ------------------------------------------------------------------------
    # RERANKING
    # ------------------------------------------------------------------------

    workflow.add_conditional_edges(
        "rerank",
        route_after_rerank,
        {
            "VECTOR_DB": "merge_context",
            "HYBRID": "hybrid_vector_done",
        },
    )

    # ------------------------------------------------------------------------
    # EXISTING HYBRID JOIN
    # ------------------------------------------------------------------------

    workflow.add_edge(
        [
            "hybrid_vector_done",
            "hybrid_sql_done",
        ],
        "hybrid_join",
    )

    workflow.add_edge(
        "hybrid_join",
        "merge_context",
    )

    # ------------------------------------------------------------------------
    # RESPONSE GENERATION
    # ------------------------------------------------------------------------

    workflow.add_edge(
        "merge_context",
        "generate_answer",
    )

    workflow.add_edge(
        "generate_answer",
        "evaluate_answer",
    )

    # ------------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------------

    workflow.add_conditional_edges(
        "evaluate_answer",
        route_after_evaluation,
        {
            "PASS": END,
            "REGENERATE": "generate_answer",
            "END": END,
        },
    )

    # ------------------------------------------------------------------------
    # CHECKPOINT
    # ------------------------------------------------------------------------

    memory = InMemorySaver()

    search_agent = workflow.compile(
        checkpointer=memory,
    )

    # ------------------------------------------------------------------------
    # GRAPH VISUALIZATION
    # ------------------------------------------------------------------------

    try:

        graph_image = search_agent.get_graph().draw_mermaid_png()

        with open(
            "credit_card_spend_summarization_agent.png",
            "wb",
        ) as f:

            f.write(graph_image)

        print(
            "Graph visualization saved to " "credit_card_spend_summarization_agent.png"
        )

    except Exception as exc:

        print("[build_rag_graph] " f"Could not generate graph image: {exc}")

    return search_agent


# ============================================================================
# CREATE GRAPH
# ============================================================================


rag_graph = build_rag_graph()


# ============================================================================
# PUBLIC ENTRY POINT
# ============================================================================


def run_search_agent(
    query: str,
    thread_id: str,
    user_id: str,
):

    print("============ INSIDE run_search_agent ============")

    clear_query_cancellation(thread_id)

    initial_state = {
        "thread_id": thread_id,
        "user_id": user_id,
        "query": query,
        "messages": [HumanMessage(content=query)],
        "route": "",
        "retrieval_query": "",
        "fts_query": "",
        "knowledge_strategy": "VECTOR",
        "retrieval_attempt": 0,
        "retrieved_docs": [],
        "reranked_docs": [],
        "generated_sql": "",
        "sql_result": "",
        "sql_context": "",
        "vector_context": "",
        "final_context": "",
        "response": {},
        "evaluation": "",
        "evaluation_feedback": "",
        "evaluate_count": 0,
        "UserPreference": "",
    }

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    try:

        final_state = rag_graph.invoke(
            initial_state,
            config=config,
        )

        # ====================================================================
        # FINAL RESPONSE GUARDRAIL
        # ====================================================================

        response = final_state["response"]

        return response

    finally:

        clear_query_cancellation(thread_id)


# ============================================================================
# STREAMING ENTRY POINT
# ============================================================================


async def run_search_agent_stream(
    query: str,
    thread_id: str,
    user_id: str,
):
    """
    Execute the RAG graph while exposing:

    - intermediate progress events from _emit_progress()
    - approved final answer as token-like UI chunks
    - final structured response
    - cancellation events
    - errors

    Memory lookup runs first for the authenticated user.

    DIRECT queries terminate at the router node.
    Retrieval queries continue through the normal RAG flow.
    """

    print("============ INSIDE run_search_agent_stream ============")

    clear_query_cancellation(thread_id)

    initial_state = {
        "thread_id": thread_id,
        "user_id": user_id,
        "query": query,
        "messages": [HumanMessage(content=query)],
        "route": "",
        "retrieval_query": "",
        "fts_query": "",
        "knowledge_strategy": "VECTOR",
        "retrieval_attempt": 0,
        "retrieved_docs": [],
        "reranked_docs": [],
        "generated_sql": "",
        "sql_result": "",
        "sql_context": "",
        "vector_context": "",
        "final_context": "",
        "response": {},
        "evaluation": "",
        "evaluation_feedback": "",
        "evaluate_count": 0,
        "UserPreference": "",
    }

    config = {
        "configurable": {
            "thread_id": thread_id,
            "streaming": True,
        }
    }

    final_response = None

    selected_route = None

    try:

        async for chunk in rag_graph.astream(
            initial_state,
            config=config,
            stream_mode=[
                "custom",
                "updates",
            ],
            version="v2",
        ):

            # ================================================================
            # CUSTOM EVENTS
            #
            # These come directly from _emit_progress()
            # ================================================================

            if chunk["type"] == "custom":

                data = chunk["data"]

                if isinstance(data, dict):
                    event_type = data.get("event")

                    if event_type == "progress":
                        message = data.get(
                            "message",
                            "",
                        )

                        if message:
                            print("[run_search_agent_stream] " f"Progress: {message}")
                            yield {
                                "event": "progress",
                                "message": message,
                            }

                    elif event_type == "token":
                        content = data.get("content", "")

                        if content:
                            yield {
                                "event": "token",
                                "content": content,
                            }

                    elif event_type == "reset":
                        yield {
                            "event": "reset",
                        }

                continue

            # ================================================================
            # NODE UPDATES
            # ================================================================

            if chunk["type"] != "updates":

                continue

            updates = chunk["data"]

            if not isinstance(
                updates,
                dict,
            ):

                continue

            for (
                node_name,
                node_update,
            ) in updates.items():

                if not isinstance(
                    node_update,
                    dict,
                ):

                    continue

                # ============================================================
                # ROUTER
                #
                # DIRECT queries end here.
                # ============================================================

                if node_name == "router":

                    selected_route = node_update.get("route")

                    print("[run_search_agent_stream] " f"Route: {selected_route}")

                    if selected_route == "DIRECT":

                        response = node_update.get("response")

                        if response:

                            final_response = response

                            direct_answer = response.get("answer", "")

                            if direct_answer:
                                # DIRECT answers terminate at the router and do
                                # not pass through evaluation. They are therefore
                                # safe to stream immediately.
                                for start in range(0, len(direct_answer), 12):
                                    raise_if_query_cancelled(thread_id)
                                    yield {
                                        "event": "token",
                                        "content": direct_answer[start : start + 12],
                                    }
                                    await asyncio.sleep(0.02)

                            print(
                                "[run_search_agent_stream] " "DIRECT response captured."
                            )

                # ============================================================
                # GENERATE ANSWER
                #
                # RDBMS / VECTOR / HYBRID queries eventually
                # produce their response here.
                # ============================================================

                elif node_name == "generate_answer":

                    response = node_update.get("response")

                    if response:

                        final_response = response

                        print("[run_search_agent_stream] " "Answer generated.")

                # ============================================================
                # OTHER NODE RESPONSES
                #
                # If a future node directly produces the final response,
                # this allows us to capture it without changing the design.
                # ============================================================

                elif "response" in node_update:

                    response = node_update.get("response")

                    if response:

                        final_response = response

            # ================================================================
            # CANCELLATION
            # ================================================================

            raise_if_query_cancelled(thread_id)

        # ====================================================================
        # FINAL RESPONSE
        # ====================================================================

        if final_response is None:

            raise RuntimeError(
                "The RAG pipeline completed " "without producing a response."
            )

        print("[run_search_agent_stream] " f"Final route: {selected_route}")

        # --------------------------------------------------------------------
        # APPROVED ANSWER STREAM
        # --------------------------------------------------------------------
        #
        # For RDBMS / VECTOR / HYBRID queries, generate_answer_node() buffers
        # the answer and evaluate_answer_node() decides whether it is accepted
        # or needs regeneration. Therefore, nothing from the generated answer
        # is sent to the UI until the graph has reached its final state.
        #
        # At this point final_response is the authoritative response: either
        # evaluation passed or the second evaluation attempt was reached.
        # Stream that approved answer to the existing UI in small chunks.
        # The short delay is intentional: without it all chunks can arrive in
        # the same event-loop turn, causing the Streamlit queue to receive the
        # whole answer before its 200ms fragment gets a chance to render it.
        # --------------------------------------------------------------------

        approved_answer = final_response.get("answer", "")

        if selected_route != "DIRECT" and approved_answer:
            for start in range(0, len(approved_answer), 12):
                raise_if_query_cancelled(thread_id)

                yield {
                    "event": "token",
                    "content": approved_answer[start : start + 12],
                }

                await asyncio.sleep(0.02)

        yield {
            "event": "final",
            "data": final_response,
        }

    except QueryCancelled:

        print("[run_search_agent_stream] " f"Query cancelled: {thread_id}")

        yield {
            "event": "cancelled",
            "message": "The query was stopped.",
        }

    except Exception as exc:

        print("[run_search_agent_stream] " f"Streaming query failed: {exc}")

        yield {
            "event": "error",
            "status_code": 500,
            "message": ("Unable to process your question. " "Please try again."),
        }

    finally:

        clear_query_cancellation(thread_id)
