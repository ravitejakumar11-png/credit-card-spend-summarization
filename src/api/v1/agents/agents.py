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


# ============================================================================
# CANCELLATION
# ============================================================================


def _check_cancelled(state: RAGState) -> None:
    thread_id = state.get("thread_id")
    raise_if_query_cancelled(thread_id)


# ============================================================================
# USER-FACING STREAMING
# ============================================================================


def _emit_progress(message: str) -> None:
    """
    Emit a small user-facing progress update.

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
        # Streaming is optional.
        # Never allow progress streaming to break the RAG pipeline.
        pass


# ============================================================================
# LLM FACTORIES
# ============================================================================


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


# ============================================================================
# MEM0
# ============================================================================


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
    evaluation: Literal[
        "PASS",
        "REGENERATE",
    ]

    feedback: str


class UserPreferenceDecision(BaseModel):
    is_preference: bool
    preference: str


# ============================================================================
# MEM0 SAVE
# ============================================================================


def save_user_preference_to_mem0(
    preference: str,
    user_id: str,
) -> None:
    """
    Persist a detected user preference in Mem0.

    Memory failures are intentionally non-fatal.
    """

    if not preference:
        return

    if not user_id:
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
        print("[Mem0] Failed to save preference: " f"{exc}")


# ============================================================================
# MEM0 RETRIEVE
# ============================================================================


def retrieve_user_preferences_from_mem0(
    query: str,
    user_id: str,
    top_k: int = 5,
) -> str:
    """
    Retrieve user-specific preferences from Mem0.

    Search is scoped to user_id.

    Returns a compact newline-separated string.
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
            filters={
                "user_id": user_id,
            },
            top_k=top_k,
        )

        if isinstance(results, dict):
            memories = results.get(
                "results",
                [],
            )
        else:
            memories = results or []

        preferences = []

        seen = set()

        for item in memories:

            if isinstance(item, dict):

                memory = item.get(
                    "memory",
                    "",
                )

            else:

                memory = getattr(
                    item,
                    "memory",
                    "",
                )

            if not memory:
                continue

            memory = str(memory).strip()

            if not memory:
                continue

            if memory in seen:
                continue

            seen.add(memory)

            preferences.append(memory)

        if not preferences:

            print("[Mem0] No stored preferences found for " f"user_id={user_id}")

            return ""

        formatted = "\n".join(f"- {preference}" for preference in preferences)

        print(
            "[Mem0] Retrieved "
            f"{len(preferences)} preference(s) "
            f"for user_id={user_id}"
        )

        return formatted

    except Exception as exc:

        print("[Mem0] Failed to retrieve preferences: " f"{exc}")

        return ""


# ============================================================================
# DETECT USER PREFERENCE
# ============================================================================


def detect_user_preference(
    query: str,
    history: list,
) -> UserPreferenceDecision:
    """
    Detect whether the current user message contains a long-term
    preference, like, dislike, habit, choice, or default.

    This function does NOT answer the user's question.
    """

    llm = _get_router_llm()

    structured_llm = llm.with_structured_output(UserPreferenceDecision)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a user-preference detection component for the
NorthStar Credit Card Assistant.

Determine whether the CURRENT USER MESSAGE contains information
that should be remembered as a long-term user preference, like,
dislike, habit, choice, or personal preference.

A preference should be detected when the user explicitly or
implicitly tells the assistant something they:

- like
- dislike
- prefer
- usually do
- want remembered
- want as a default

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
- "I like NorthStar Gold Card."

Examples that SHOULD NOT be remembered:

- "What are the benefits of this card?"
- "How many points did I earn?"
- "What is my balance?"
- "What is the annual fee?"
- "Tell me about lounge access."
- "Hello."
- "What can you do?"
- Temporary instructions that only apply to the current request.

IMPORTANT:

A statement such as:

"I like NorthStar Gold Card."

is a preference even though it contains a card/product name.

Only extract information actually stated or clearly implied.

Do not invent preferences.

Do not store ordinary questions as preferences.

Do not store sensitive personal information.

Keep the preference concise.

Write the preference as a statement about the user.

If multiple preferences are expressed, combine them into one
concise preference string.

If there is no preference:

is_preference = false
preference = ""

Do not answer the user's question.

Return only:

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


# ============================================================================
# MEMORY NODE
# ============================================================================


def memory_node(state: RAGState) -> RAGState:
    """
    Manage long-term user memory before route selection.

    Flow:

        1. Detect a preference in the current message.
        2. Keep the detected preference in RAGState.
        3. If authenticated, save it to Mem0.
        4. If authenticated, retrieve existing Mem0 preferences.
        5. Combine current and stored preferences.
        6. Pass UserPreference to the router.

    IMPORTANT:

    Preference detection happens even for guests.

    Mem0 persistence and retrieval require user_id.
    """

    _check_cancelled(state)

    print("========= INSIDE MEMORY NODE =========")

    query = state["query"]

    history = state.get(
        "messages",
        [],
    )

    user_id = (state.get("user_id") or "").strip()

    # ------------------------------------------------------------------------
    # 1. Detect current preference
    # ------------------------------------------------------------------------

    preference_decision = detect_user_preference(
        query=query,
        history=history,
    )

    _check_cancelled(state)

    current_preference = ""

    if (
        preference_decision.is_preference
        and preference_decision.preference
        and preference_decision.preference.strip()
    ):

        current_preference = preference_decision.preference.strip()

        print("[memory_node] Current preference detected: " f"{current_preference}")

    else:

        print("[memory_node] No new user preference detected.")

    # ------------------------------------------------------------------------
    # 2. Guest user
    #
    # Keep the current preference in RAGState so the router can
    # recognize the message as a preference-only statement.
    #
    # Do not use Mem0 for guests.
    # ------------------------------------------------------------------------

    if not user_id:

        print(
            "[memory_node] No authenticated user_id. "
            "Mem0 persistence/retrieval skipped."
        )

        return {
            "user_preferences": current_preference,
        }

    # ------------------------------------------------------------------------
    # 3. Save current preference
    # ------------------------------------------------------------------------

    if current_preference:

        save_user_preference_to_mem0(
            preference=current_preference,
            user_id=user_id,
        )

    _check_cancelled(state)

    # ------------------------------------------------------------------------
    # 4. Retrieve existing preferences
    # ------------------------------------------------------------------------

    stored_preferences = retrieve_user_preferences_from_mem0(
        query=query,
        user_id=user_id,
        top_k=5,
    )

    _check_cancelled(state)

    # ------------------------------------------------------------------------
    # 5. Combine current preference + stored preferences
    # ------------------------------------------------------------------------

    preference_parts = []

    if current_preference:

        preference_parts.append(f"- {current_preference}")

    if stored_preferences:

        for line in stored_preferences.splitlines():

            line = line.strip()

            if not line:
                continue

            normalized_line = line.lstrip("- ").strip()

            if current_preference and normalized_line == current_preference:
                continue

            preference_parts.append(line)

    user_preferences = "\n".join(preference_parts)

    print("[memory_node] UserPreference context:\n" f"{user_preferences}")

    return {
        "user_preferences": user_preferences,
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

    user_preferences = state.get(
        "user_preferences",
        "",
    )

    print("[router_node] UserPreference context:\n" f"{user_preferences}")

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

3. HYBRID

Use when BOTH the relational database and knowledge base are required.

Use HYBRID only when the answer genuinely requires BOTH:

1. actual customer/transaction data
2. knowledge-base rules or policies

4. DIRECT

Use when no database or knowledge-base retrieval is required.

This includes:

- Greetings
- Simple conversation
- Questions about the assistant's capabilities
- General chit-chat
- Questions unrelated to banking, credit cards, transactions or rewards
- Questions about the user's previously stored preferences, likes,
  dislikes, habits, choices or defaults

IMPORTANT USER PREFERENCE RULE:

The system provides a "User Preference Memory" section below.

This memory contains preferences that were previously stored for
the authenticated user.

Examples:

- User prefers travel by bus rather than train.
- User likes the NorthStar Gold card.
- User prefers concise answers.
- User prefers cashback rewards.

If the user asks about their own preferences, likes, dislikes,
choices, habits or remembered preferences, use the supplied
User Preference Memory to answer the question.

Examples:

User:
"What do I prefer?"

If memory contains:
- User prefers travel by bus rather than train.
- User likes the NorthStar Gold card.

Then the DIRECT response should say something like:

"You prefer traveling by bus rather than train, and you like the
NorthStar Gold card."

User:
"What cards do I like?"

If memory contains:
- User likes the NorthStar Gold card.

Then answer:

"You like the NorthStar Gold card."

User:
"What are my preferences?"

Use all relevant preferences from the supplied memory.

If no relevant preference memory exists, say that no stored preference
was found rather than pretending to remember something.

IMPORTANT:

- Never invent a preference.
- Never infer a preference that is not present in User Preference Memory.
- Never use another user's memory.
- User Preference Memory is only personalization information.
- It is NOT authoritative for balances, transactions, fees, rewards,
  policies, eligibility, or other factual product/customer information.

For DIRECT:

- Provide a brief natural response in `direct_response`.
- If the question is about user preferences, answer using ONLY the
  supplied User Preference Memory.
- Do not retrieve from RDBMS or VECTOR_DB.
- For unrelated questions, politely explain that the assistant is
  designed for NorthStar credit-card and related topics.

Routing rules:

- Return exactly ONE route.
- Use HYBRID whenever both structured data and knowledge-base information
  are required.
- Use retrieval routes only when retrieval is necessary.
- Use conversation history to resolve references.
- For VECTOR_DB, RDBMS and HYBRID, set `direct_response` to an empty string.
- For DIRECT, always provide the actual response in `direct_response`.

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

User Preference Memory:

{user_preferences}

IMPORTANT:

If the current question is asking about the user's preferences,
likes, dislikes, habits, choices or defaults, use the User Preference
Memory above to formulate the DIRECT response.

If the User Preference Memory is empty or does not contain a relevant
preference, do not invent one. Say that no stored preference was found.
""",
            ),
        ]
    )

    chain = prompt | structured_llm

    decision = chain.invoke(
        {
            "query": query,
            "history": history,
            "user_preferences": user_preferences,
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

    elif decision.route == "DIRECT":

        _emit_progress("Preparing a response...")

    print(f"[router_node] Route : {decision.route}")
    print(f"[router_node] Reason: {decision.reason}")
    print(f"[router_node] Direct Response: {decision.direct_response}")

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


def hybrid_start_node(
    state: RAGState,
) -> RAGState:

    print("========= INSIDE HYBRID START NODE =========")

    return {}


def hybrid_join_node(
    state: RAGState,
) -> RAGState:

    print("========= INSIDE HYBRID JOIN NODE =========")

    return {}


# ============================================================================
# KNOWLEDGE STRATEGY ROUTER
# ============================================================================


def knowledge_strategy_router_node(
    state: RAGState,
) -> RAGState:

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
You are a retrieval strategy classifier for a banking
knowledge base.

Your task is to decide the best document retrieval strategy.

Available strategies:

1. VECTOR

Use VECTOR when:

- The user is asking for explanation, reasoning, summary,
  or conceptual understanding.
- The answer requires semantic understanding.
- The wording may not exactly match the document.

Examples:

"What are the benefits of this credit card?"

"Explain reward redemption rules"

"How does billing work?"


2. FTS

Use FTS when:

- Exact words, phrases, identifiers, names, codes, or specific
  terms matter.
- The user is looking for an exact mention in documents.

Examples:

"Find the section mentioning NEFT"

"Where is Statement Credit mentioned?"

"Find Platinum card"


3. VECTOR_FTS

Use VECTOR_FTS when:

- The query needs both semantic understanding and exact terminology.
- The question is complex or policy/rule based.
- Missing either semantic or keyword retrieval may reduce recall.

Examples:

"What are the eligibility rules for accelerated dining rewards?"

"Explain reward points conversion and redemption options"

"What is MCC category and how does NorthStar classify transactions?"

"Explain the dining reward eligibility rules for accelerated points"

"What are the redemption options and their point conversion values?"

"Tell me about Platinum card lounge benefits"


Create appropriate retrieval queries.

For VECTOR:

Generate retrieval_query.

Set fts_query empty.

For FTS:

Generate fts_query.

Set retrieval_query empty.

For VECTOR_FTS:

Generate both.

Rules:

- Do not answer the question.
- Do not invent facts.
- Return structured output only.
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
            "history": state.get(
                "messages",
                [],
            ),
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


def query_optimization_node(
    state: RAGState,
) -> RAGState:

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

VECTOR:

Create a semantic retrieval query.

FTS:

Create a lexical search query.

VECTOR_FTS:

Create both.

Return only structured output.
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
            "history": state.get(
                "messages",
                [],
            ),
            "query": state["query"],
            "knowledge_strategy": state.get(
                "knowledge_strategy",
                "VECTOR",
            ),
        }
    )

    _check_cancelled(state)

    print("========= QUERY REFORMULATION =========")

    print(f"Original Query  : {state['query']}")

    print(f"Retrieval Query : {result.retrieval_query}")

    print(f"FTS Query       : {result.fts_query}")

    return {
        "retrieval_query": result.retrieval_query,
        "fts_query": result.fts_query,
    }


# ============================================================================
# NL2SQL
# ============================================================================


def nl2sql_node(
    state: RAGState,
) -> RAGState:

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

Generate ONE valid PostgreSQL SELECT query that answers the
current user question using the database schema, business rules
and conversation history.

Rules:

- Use only tables and columns present in schema.
- SELECT only.
- Never generate INSERT, UPDATE, DELETE, DROP or DDL.
- Return ONLY raw SQL.
- Add LIMIT 50 unless aggregate.
- Resolve references using conversation history.
- Preserve entities and filters established in previous turns.
- Do not broaden customer-specific queries.

Reward point fields:

credit_cards.reward_points:
Current reward point balance.

card_transactions.reward_pts_earned:
Points earned by individual transactions.

reward_transactions.points_earned:
Points posted through reward ledger.

Do not treat these as interchangeable.

Avoid double-counting when joining multiple detail tables.

Before returning SQL verify:

1. It answers the current question.
2. References are resolved.
3. Required filters are present.
4. Dataset is not unnecessarily broadened.
5. SQL is valid PostgreSQL.

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

    raw_sql = sql_chain.invoke(
        {
            "schema": schema_info,
            "history": state.get(
                "messages",
                [],
            ),
            "question": state["query"],
        }
    )

    _check_cancelled(state)

    generated_sql = raw_sql.content.strip()

    print("======== GENERATED SQL QUERY ========")

    print(generated_sql)

    try:

        sql_result = db.run(generated_sql)

        _check_cancelled(state)

    except Exception as err:

        sql_result = f"Generated SQL execution error: {err}"

    print("========= NL2SQL NODE OUTPUT =========")

    print("\nSQL Result:")

    print(str(sql_result))

    print("======================================")

    return {
        "generated_sql": generated_sql,
        "sql_result": str(sql_result),
    }


def route_after_nl2sql(
    state: RAGState,
) -> str:

    if state["route"] == "HYBRID":

        return "HYBRID"

    return "RDBMS"


# ============================================================================
# RERANK
# ============================================================================


def rerank_node(
    state: RAGState,
) -> RAGState:

    _check_cancelled(state)

    print("========= INSIDE RERANK NODE =========")

    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    docs = state.get(
        "retrieved_docs",
        [],
    )

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


def route_after_rerank(
    state: RAGState,
) -> str:

    if state["route"] == "HYBRID":

        return "HYBRID"

    return "VECTOR_DB"


# ============================================================================
# MERGE CONTEXT
# ============================================================================


def merge_context_node(
    state: RAGState,
) -> RAGState:

    _check_cancelled(state)

    print("========= INSIDE MERGE CONTEXT NODE =========")

    _emit_progress("Putting the information together...")

    route = state.get("route")

    # ------------------------------------------------------------------------
    # SQL CONTEXT
    # ------------------------------------------------------------------------

    sql_context = ""

    if route in (
        "RDBMS",
        "HYBRID",
    ) and state.get("sql_result"):

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

    if route in (
        "VECTOR_DB",
        "HYBRID",
    ) and state.get("reranked_docs"):

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


# ============================================================================
# GENERATE ANSWER
# ============================================================================


def generate_answer_node(
    state: RAGState,
) -> RAGState:

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

Answer the user's question using ONLY the supplied retrieved context.

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

Use the source relevant to the question.

If both sources are relevant, combine them.

Do not invent information.

Do not mention:

- SQL
- databases
- vector search
- retrieval
- prompts
- internal systems
- internal reasoning

Use conversation history to resolve references.

Be concise and business-friendly.

USER PREFERENCE MEMORY:

User Preference Memory contains long-term preferences,
likes, dislikes, habits, choices, or defaults.

It MAY be used to personalize recommendations or presentation.

It is NOT authoritative factual information.

Never use User Preference Memory as evidence for:

- balances
- transactions
- spending
- fees
- reward rules
- policies
- eligibility
- account information
- customer-specific facts

Retrieved Context is the factual authority.

Populate:

- answer
- document_name
- page_no
- policy_citations
- sql_query_executed

For database-only answers, document fields may be empty.

For knowledge-base answers, populate document information
when available.

For HYBRID answers, populate both when applicable.
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

User Preference Memory:

{user_preferences}

Previous Evaluation Feedback:

{feedback}
""",
            ),
        ]
    )

    chain = prompt | structured_llm

    result = chain.invoke(
        {
            "query": state["query"],
            "context": state.get(
                "final_context",
                "",
            ),
            "history": state.get(
                "messages",
                [],
            ),
            "user_preferences": state.get(
                "user_preferences",
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

    if (
        state.get(
            "evaluate_count",
            0,
        )
        == 0
    ):

        _emit_progress("Preparing your answer...")

    evaluate_count = (
        state.get(
            "evaluate_count",
            0,
        )
        + 1
    )

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

Use only the retrieved context as factual authority.

Do not substitute general knowledge.

Do not broaden or narrow policy statements.

Do not blindly trust numerical results produced by SQL.

Check for obvious aggregation inconsistencies, duplicated counts,
conflicting metrics, or misuse of fields when enough context exists.

USER PREFERENCE MEMORY:

User Preference Memory may explain user preferences,
likes, dislikes, habits, choices or desired personalization.

It is NOT factual evidence.

Do not use it to validate:

- balances
- transactions
- spending
- fees
- reward rules
- policies
- eligibility
- customer facts

Use User Preference Memory only to determine whether the answer
appropriately respects the user's stated preferences when relevant.
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

User Preference Memory:

{user_preferences}

Generated Answer:

{answer}

Previous Evaluation Feedback:

{feedback}
""",
            ),
        ]
    )

    chain = prompt | structured_llm

    answer = state.get(
        "response",
        {},
    ).get(
        "answer",
        "",
    )

    result = chain.invoke(
        {
            "query": state["query"],
            "context": state.get(
                "final_context",
                "",
            ),
            "history": state.get(
                "messages",
                [],
            ),
            "user_preferences": state.get(
                "user_preferences",
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

    if attempt >= 2:
        return False

    if not docs:

        print(
            "[retrieval_quality] "
            f"No documents found for strategy {strategy}. "
            "Reformulating query."
        )

        return True

    if strategy == "VECTOR":

        score_key = "similarity"
        threshold = 0.50

    elif strategy == "FTS":

        score_key = "fts_rank"
        threshold = 0.05

    elif strategy == "VECTOR_FTS":

        score_key = "rrf_score"
        threshold = 0.01

    else:

        score_key = "similarity"
        threshold = 0.50

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

    if len(relevant_docs) < 3:

        print(
            "[retrieval_quality] " "Retrieval quality is low. " "Reformulating query."
        )

        return True

    print("[retrieval_quality] " "Retrieval quality is sufficient.")

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


def route_knowledge_strategy(
    state: RAGState,
):

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
    # MEMORY
    # ------------------------------------------------------------------------

    workflow.add_node(
        "memory",
        memory_node,
    )

    # ------------------------------------------------------------------------
    # ROUTER
    # ------------------------------------------------------------------------

    workflow.add_node(
        "router",
        router_node,
    )

    # ------------------------------------------------------------------------
    # HYBRID
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
    # KNOWLEDGE RETRIEVAL
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
    # RETRIEVAL IMPROVEMENT
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
    # ANSWER
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
    # START
    # ------------------------------------------------------------------------

    workflow.set_entry_point("memory")

    # START → MEMORY → ROUTER

    workflow.add_edge(
        "memory",
        "router",
    )

    # ------------------------------------------------------------------------
    # PRIMARY ROUTER
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
    # KNOWLEDGE STRATEGY
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
    # HYBRID
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
    # NL2SQL
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
    # SEARCH → ATTEMPT
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
    # REFORMULATION
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
    # RERANK
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
    # HYBRID JOIN
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
    # ANSWER GENERATION
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
        "user_preferences": "",
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

    - progress events
    - final response
    - cancellation events
    - errors

    Memory lookup runs first.

    DIRECT queries terminate at router.
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
        "user_preferences": "",
    }

    config = {
        "configurable": {
            "thread_id": thread_id,
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

            # ----------------------------------------------------------------
            # CUSTOM EVENTS
            # ----------------------------------------------------------------

            if chunk["type"] == "custom":

                data = chunk["data"]

                if isinstance(data, dict) and data.get("event") == "progress":

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

                continue

            # ----------------------------------------------------------------
            # NODE UPDATES
            # ----------------------------------------------------------------

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

                # ------------------------------------------------------------
                # ROUTER
                # ------------------------------------------------------------

                if node_name == "router":

                    selected_route = node_update.get("route")

                    print("[run_search_agent_stream] " f"Route: {selected_route}")

                    if selected_route == "DIRECT":

                        response = node_update.get("response")

                        if response:

                            final_response = response

                            print(
                                "[run_search_agent_stream] " "DIRECT response captured."
                            )

                # ------------------------------------------------------------
                # GENERATE ANSWER
                # ------------------------------------------------------------

                elif node_name == "generate_answer":

                    response = node_update.get("response")

                    if response:

                        final_response = response

                        print("[run_search_agent_stream] " "Answer generated.")

                # ------------------------------------------------------------
                # OTHER RESPONSE-PRODUCING NODES
                # ------------------------------------------------------------

                elif "response" in node_update:

                    response = node_update.get("response")

                    if response:

                        final_response = response

            # ----------------------------------------------------------------
            # CANCELLATION
            # ----------------------------------------------------------------

            raise_if_query_cancelled(thread_id)

        # --------------------------------------------------------------------
        # FINAL RESPONSE
        # --------------------------------------------------------------------

        if final_response is None:

            raise RuntimeError(
                "The RAG pipeline completed " "without producing a response."
            )

        print("[run_search_agent_stream] " f"Final route: {selected_route}")

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
