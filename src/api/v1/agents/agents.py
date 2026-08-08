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
    route: Literal["VECTOR_DB", "RDBMS", "HYBRID"]
    reason: str


class EvaluationDecision(BaseModel):
    evaluation: Literal["PASS", "REGENERATE"]
    feedback: str


def router_node(state: RAGState) -> RAGState:

    print("========= INSIDE ROUTER NODE =========")

    llm = _get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the Query Router for the NorthStar Credit Card Agentic RAG System.

Your responsibility is to determine where information should be retrieved from
to answer the user's question.

There are ONLY three possible routes.

--------------------------------------------------
1. VECTOR_DB
--------------------------------------------------

Choose VECTOR_DB when the question is about:

- Credit card features
- Rewards program
- Reward calculation rules
- Cashback rules
- Lounge access
- Fees and charges
- Interest calculation
- Billing cycle
- EMI conversion
- Card eligibility
- Credit card benefits
- Policies
- FAQs
- Terms & Conditions
- Product documentation
- Bank rules
- Any information contained in the Credit Card Knowledge Base.

--------------------------------------------------
2. RDBMS
--------------------------------------------------

Choose RDBMS when the answer depends ONLY on structured customer data stored in
the relational database.

Examples:

- Customer profile
- Card information
- Transactions
- Merchant spends
- Billing statements
- Outstanding balance
- Credit limit
- Available credit
- Reward points earned
- Transaction history
- Spend summary
- Payments
- Monthly spending
- Transaction counts

--------------------------------------------------
3. HYBRID
--------------------------------------------------

Choose HYBRID when BOTH sources are required.

Examples:

- Why was I charged a late payment fee?
    (Need transaction history + fee policy)

- Why did I earn only 200 reward points?
    (Need transactions + reward rules)

- Explain my billing statement.
    (Need statement data + billing cycle rules)

- Why was interest charged?
    (Need statement/payment data + interest policy)

- Explain my spending summary.
    (Need SQL spend data + spend categorization rules)

--------------------------------------------------

Important Rules

- Return exactly ONE route.
- Never return more than one route.
- If both structured customer data and product knowledge are required,
  always choose HYBRID.
- If unsure, prefer HYBRID rather than guessing a single source.

Return:

route
reason
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
    }


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
                   You are an expert SQL generator for a conversational database assistant.

Your task is to generate accurate PostgreSQL queries based on:

1. The current user question
2. Previous conversation history
3. Available database schema and business rules


## Conversation Context Resolution Rules

The user may ask follow-up questions that depend on information mentioned earlier in the conversation.

The current question may contain references such as:

- he / she / they
- this customer
- this account
- this card
- this transaction
- the same one
- the previous one
- that record

Before generating SQL, you MUST resolve these references using the conversation history.


## Entity Resolution Guidelines

- Identify any relevant entities already established in the conversation.
- Maintain continuity with previously identified entities.
- If a previous turn established a specific entity, use that entity when generating SQL.
- Do not broaden the query unnecessarily when the context already identifies the target entity.

Examples of incorrect behavior:

- Returning data for all entities when the conversation context identifies a specific one.
- Ignoring previously established identifiers.
- Asking for clarification when the entity can be resolved from conversation history.


## SQL Generation Rules

Before generating the final SQL:

1. Understand the user's intent.
2. Resolve any ambiguous references using conversation history.
3. Identify the required tables and relationships.
4. Apply appropriate filters based on the resolved context.
5. Generate only the SQL required to answer the user's question.


## Validation Before Returning SQL

Check:

- Does the query answer the current user question?
- Did I correctly resolve any references from previous conversation?
- Am I querying a broader dataset than required?
- Are required filtering conditions present?
- Is the SQL syntactically valid PostgreSQL?


If the conversation provides enough context to identify the target entity, the generated SQL should reflect that context.


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

    return {**state, "reranked_docs": reranked_docs}


def merge_context_node(state: RAGState) -> RAGState:
    """
    Merge the SQL results and Vector search results into a single context
    for the answer generation node.

    This node does NOT call an LLM.
    """

    print("========= INSIDE MERGE CONTEXT NODE =========")

    # ----------------------------------------------------------
    # Prepare SQL Context
    # ----------------------------------------------------------
    sql_context = ""

    if state.get("sql_result"):
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

    if state.get("reranked_docs"):

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
    # Merge both contexts
    # ----------------------------------------------------------
    contexts = []

    if sql_context:
        contexts.append(sql_context)

    if vector_context:
        contexts.append(vector_context)

    final_context = "\n\n".join(contexts)

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
You are an expert Credit Card Customer Support Assistant for NorthStar Bank.

You are provided with:

1. Structured customer/account information retrieved from the relational database.
2. Credit card product knowledge retrieved from the Knowledge Base.

Instructions:

- Answer ONLY using the provided context.
- Never make up information.
- If the answer requires both SQL data and Knowledge Base information,
  combine them into one natural response.
- If only one source is available, answer using only that source.
- If the answer cannot be determined from the provided context,
  politely say that the required information is unavailable.

When answering:

- Be clear and concise.
- Use bullet points wherever appropriate.
- Explain numbers in a business-friendly manner.
- Never mention SQL, database, vector search or internal implementation.
- Never expose internal reasoning.

Populate the structured response fields:

- answer
- document_name
- page_no
- policy_citations
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
            "context": state["final_context"],
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

    llm = _get_llm()
    structured_llm = llm.with_structured_output(EvaluationDecision)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an Answer Quality Evaluator for the NorthStar Credit Card Assistant.

Your job is NOT to answer the question.

Evaluate whether the generated answer:

- Correctly answers the user's question.
- Uses only the retrieved context.
- Does not hallucinate information.
- Includes all important information needed to answer the user's
question, but avoids adding unnecessary details.
- Is clear and complete.

If the answer is satisfactory, return:

PASS

Otherwise return:

REGENERATE

If regeneration is needed, provide a short feedback explaining
what is missing or incorrect.
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
            "context": state["final_context"],
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
            "VECTOR_DB": "vector_search",
            "RDBMS": "nl2sql",
            "HYBRID": "vector_search",
        },
    )

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


def run_search_agent(query: str):
    print("============1. INSIDE run_search_agent ")
    initial_state = {
        "query": query,
        "messages": [],
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
        "evaluate_count": 0,
    }

    final_state = rag_graph.invoke(initial_state)

    return final_state["response"]
