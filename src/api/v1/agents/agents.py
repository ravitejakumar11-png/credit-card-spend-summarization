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
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from src.api.v1.states.rag_state import RAGState
from src.api.v1.tools.vector_search_tool import vector_search_node
from src.api.v1.schemas.query_schema import AIResponse
from src.core.db import get_sql_database
from src.core.db import get_cached_schema

load_dotenv()


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


class RouteDecision(BaseModel):
    route: Literal[
        "VECTOR_DB",
        "RDBMS",
        "HYBRID",
        "DIRECT",
    ]
    reason: str
    direct_response: str


class EvaluationDecision(BaseModel):
    evaluation: Literal["PASS", "REGENERATE"]
    feedback: str


def router_node(state: RAGState) -> RAGState:

    print("========= INSIDE ROUTER NODE =========")

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
1. actual customer/transaction data, and
2. knowledge-base rules or policies.

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
        {"query": state["query"], "history": state.get("messages", [])}
    )

    print(f"[router_node] Route : {decision.route}")
    print(f"[router_node] Reason: {decision.reason}")

    return {
        **state,
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


def query_reformulation_node(state: RAGState) -> RAGState:
    print("========= INSIDE QUERY REFORMULATION NODE =========")

    llm = _get_router_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You rewrite user questions into effective search queries for the
NorthStar Credit Card knowledge base.

Create ONE standalone retrieval query.

Use conversation history to resolve references such as:
he, she, this, that, previous, same, what about, how about.

Also expand important domain terms with closely related terms when useful
for document retrieval.

Rules:
- Preserve the user's intent.
- Resolve conversational references using history.
- Add useful credit-card terminology or synonyms when they improve retrieval.
- Do not answer the question.
- Do not invent facts, policies, names, amounts, dates, or rules.
- Do not introduce information that is not present in the question or history.
- If the question is already clear, make only useful retrieval improvements.
- Return only the search query.
""",
            ),
            (
                "human",
                """
Conversation History:
{history}

Current Question:
{query}
""",
            ),
        ]
    )

    chain = prompt | llm

    result = chain.invoke(
        {
            "history": state.get("messages", []),
            "query": state["query"],
        }
    )

    retrieval_query = result.content.strip()

    print("========= QUERY REFORMULATION =========")
    print(f"Original Query   : {state['query']}")
    print(f"Retrieval Query  : {retrieval_query}")

    return {
        **state,
        "retrieval_query": retrieval_query,
    }


def nl2sql_node(state: RAGState) -> RAGState:
    print("About to generate nl2sql")
    # connect to LLM
    llm = _get_llm()
    # connect to rdbms
    db = get_sql_database()
    # get the tables' live schema
    # schema_info = db.get_table_info()
    schema_info = get_cached_schema()
    # write the system prompt and pass on the schema to get only sql query
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
- Do not ask for clarification when the target can be resolved from
  the available history.

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
    # preprare the chain and invoke with a query
    sql_chain = sql_prompt | llm
    # look for sql query only
    history = state.get("messages", [])
    raw_sql = sql_chain.invoke(
        {"schema": schema_info, "history": history, "question": state["query"]}
    )
    print("========GENERATED raw_sql query is: =====")
    print(raw_sql.content)
    generated_sql = raw_sql.content

    # execute the generated sql query  to get the outout from RDMBS
    try:
        sql_result = db.run(generated_sql)
    except Exception as err:
        sql_result = f"Generated SQL execution error: {err}"

    # # connect to LLM to get the natural language response
    # structured_llm = llm.with_structured_output(AIResponse)
    # nl_answer_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             """You are a helpful data analyst. Answer the user's question using
    #            the SQL query results below. Be concise and format numbers/lists clearly.
    #            Set policy_citations to empty string,
    #            page_no to 'N/A', and document_name to 'agentic_rag_db'.
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
    # # return the sql query is RAGState
    # # and also the output in sql_result of RAGState
    print("========= NL2SQL NODE OUTPUT =========")
    print("\nSQL Result:")
    print(str(sql_result))
    print("======================================")
    return {
        **state,
        "generated_sql": generated_sql,
        "sql_result": str(sql_result),
        #  "response": response,
    }


def rerank_node(state: RAGState):
    # establish connection with the cohere reranking model
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    # send the query and the retrieved_docs to the reranking model

    docs = state["retrieved_docs"]
    search_query = state.get("retrieval_query") or state["query"]

    print("=======3. INSIDE rerank_node. Before calling reranker =========")
    rerank_response = co.rerank(
        model="rerank-v3.5",
        query=search_query,
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

    return {**state, "reranked_docs": reranked_docs}


def merge_context_node(state: RAGState) -> RAGState:
    """
    Merge the SQL results and Vector search results into a single context
    for the answer generation node.

    This node does NOT call an LLM.
    """

    print("========= INSIDE MERGE CONTEXT NODE =========")

    route = state.get("route")

    # ----------------------------------------------------------
    # Prepare SQL Context
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # Prepare Vector Context
    # ----------------------------------------------------------
    vector_context = ""

    if route in ("VECTOR_DB", "HYBRID") and state.get("reranked_docs"):

        vector_chunks = []

        for doc in state["reranked_docs"]:

            source = doc.metadata.get("source", "Unknown Document")
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

    # ----------------------------------------------------------
    # Merge contexts
    # ----------------------------------------------------------
    contexts = []

    if sql_context:
        contexts.append(sql_context)

    if vector_context:
        contexts.append(vector_context)

    final_context = "\n\n".join(contexts)

    print(f"Route                  : {route}")
    print(f"SQL Context Present    : {bool(sql_context)}")
    print(f"Vector Context Present : {bool(vector_context)}")

    return {
        **state,
        "sql_context": sql_context,
        "vector_context": vector_context,
        "final_context": final_context,
    }


def generate_answer_node(state: RAGState) -> RAGState:

    print("========= INSIDE GENERATE ANSWER NODE =========")

    llm = _get_llm()
    structured_llm = llm.with_structured_output(AIResponse)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
SOURCE ACCURACY

Answer using only the supplied context.

Preserve the exact meaning and scope of the source.

Do not generalize, infer, or expand policy statements.

If the source says an exclusion applies to a specific portion,
condition, transaction type, or circumstance, apply the exclusion only
to that scope.

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

## Response Style

- Be concise and business-friendly.
- Use bullets when useful.
- Explain numerical results clearly.
- Do not mention SQL, databases, vector search, retrieval, prompts,
  internal systems or internal reasoning.

Populate:
- answer
- document_name
- page_no
- policy_citations
- sql_query_executed
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

    chain = prompt | structured_llm

    history = state.get("messages", [])

    result = chain.invoke(
        {
            "query": state["query"],
            "context": state.get("final_context", ""),
            "history": history,
            "feedback": state.get("evaluation_feedback", ""),
        }
    )

    print("[generate_answer_node] Answer generated.")

    return {
        **state,
        "response": result.model_dump(),
        #  "retry_count": state.get("retry_count", 0) + 1,
    }


def evaluate_answer_node(state: RAGState) -> RAGState:

    print("========= INSIDE EVALUATE ANSWER NODE =========")
    print("========= ANSWER BEING EVALUATED =========")
    print(state["response"])

    evaluate_count = state.get("evaluate_count", 0) + 1

    print(f"Evaluation attempt number: {evaluate_count}")

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

Evaluate strictly against the retrieved context and preserve the exact scope of qualifiers, 
conditions and exceptions. Do not broaden a specific exclusion into a broader exclusion.

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
rather than generalizing it to the entire category..
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

    answer = state.get("response", {}).get("answer", "")

    print("========= ANSWER SENT TO EVALUATOR =========")
    print(answer)

    result = chain.invoke(
        {
            "query": state["query"],
            "context": state.get("final_context", ""),
            "history": history,
            "answer": answer,
            "feedback": state.get("evaluation_feedback", ""),
        }
    )

    print("========= EVALUATOR RESULT =========")
    print(f"[evaluate_answer_node] {result.evaluation}")
    print(result)

    return {
        **state,
        "evaluation": result.evaluation,
        "evaluation_feedback": result.feedback,
        "evaluate_count": evaluate_count,
    }


def build_rag_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("router", router_node)
    workflow.add_node("nl2sql", nl2sql_node)
    workflow.add_node("vector_search", vector_search_node)
    workflow.add_node("query_reformulation", query_reformulation_node)
    workflow.add_node("rerank", rerank_node)

    workflow.add_node("merge_context", merge_context_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("evaluate_answer", evaluate_answer_node)

    # the following is the starting point
    workflow.set_entry_point("router")

    # conditional routing: "vectordb" -> vector_search (or) "rdbms" -> nl2sql
    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "VECTOR_DB": "query_reformulation",
            "RDBMS": "nl2sql",
            "HYBRID": "query_reformulation",
            "DIRECT": END,
        },
    )

    workflow.add_edge("query_reformulation", "vector_search")
    workflow.add_edge("vector_search", "rerank")

    workflow.add_conditional_edges(
        "rerank",
        lambda state: state["route"],
        {
            "VECTOR_DB": "merge_context",
            "HYBRID": "nl2sql",
        },
    )

    workflow.add_edge("nl2sql", "merge_context")
    workflow.add_edge("merge_context", "generate_answer")
    workflow.add_edge("generate_answer", "evaluate_answer")

    workflow.add_conditional_edges(
        "evaluate_answer",
        lambda state: (
            "PASS"
            if state["evaluation"] == "PASS"
            else ("REGENERATE" if state.get("evaluate_count", 0) == 1 else "END")
        ),
        {
            "PASS": END,
            "REGENERATE": "generate_answer",
            "END": END,
        },
    )

    memory = InMemorySaver()

    search_agent = workflow.compile(checkpointer=memory)

    # generating and saving the graph visualization
    graph_image = search_agent.get_graph().draw_mermaid_png()
    with open("credit_card_spend_summarization_agent.png", "wb") as f:
        f.write(graph_image)

    return search_agent


rag_graph = build_rag_graph()


def run_search_agent(query: str, thread_id: str):
    print("============1. INSIDE run_search_agent")

    initial_state = {
        "query": query,
        "retrieval_query": "",
        "messages": [HumanMessage(content=query)],
        "route": "",
        "retrieved_docs": [],
        "reranked_docs": [],
        "generated_sql": "",
        "sql_result": "",
        "vector_context": "",
        "sql_context": "",
        "final_context": "",
        "response": {},
        "evaluation": "",
        "evaluate_count": 0,
        "evaluation_feedback": "",
    }

    config = {"configurable": {"thread_id": thread_id}}

    final_state = rag_graph.invoke(
        initial_state,
        config=config,
    )

    return final_state["response"]
