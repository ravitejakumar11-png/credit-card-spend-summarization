from src.api.v1.agents.agents import (
    run_search_agent,
    run_search_agent_stream,
)
from src.core.guardrails import guard_input, guard_output


def query_documents(
    query: str,
    thread_id: str,
):
    # Input guardrail
    guard_input(query)

    print(query)
    print(f"thread_id: {thread_id}")

    result = run_search_agent(
        query=query,
        thread_id=thread_id,
    )

    # run_search_agent() returns final_state["response"].
    # The response structure contains the final answer under "answer".
    if isinstance(result, dict) and result.get("answer"):
        result["answer"] = guard_output(result["answer"])

    return result


def query_documents_stream(
    query: str,
    thread_id: str | None = None,
):
    # Input guardrail runs before the StreamingResponse is opened.
    # Output guardrail is intentionally NOT applied to the stream.
    guard_input(query)

    print(query)
    print(f"thread_id: {thread_id}")

    async def event_generator():
        async for event in run_search_agent_stream(
            query=query,
            thread_id=thread_id,
        ):
            yield event

    return event_generator()
