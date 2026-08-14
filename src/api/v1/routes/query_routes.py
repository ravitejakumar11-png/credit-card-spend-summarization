import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.v1.schemas.query_schema import QueryRequest, QueryResponse
from src.api.v1.services.query_cancellation import cancel_query
from src.api.v1.services.query_service import (
    query_documents,
    query_documents_stream,
)
from src.core.db import EmbeddingServiceError
from src.core.guardrails import GuardrailViolation

router = APIRouter(prefix="/api/v1/query")


# ============================================================================
# NORMAL QUERY ENDPOINT
# ============================================================================


@router.post("/")
def query_endpoint(
    request: QueryRequest,
) -> QueryResponse:
    try:
        return query_documents(
            query=request.query,
            thread_id=request.thread_id,
            user_id=request.user_id,
        )

    except GuardrailViolation as violation:
        raise HTTPException(
            status_code=400,
            detail={
                "guardrail": violation.guard,
                "message": violation.message,
            },
        ) from violation

    except EmbeddingServiceError as exc:
        print("[query_route] " f"Embedding service unavailable: {exc}")

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        print("[query_route] " f"Query processing failed: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Unable to process the query.",
        ) from exc


# ============================================================================
# STREAMING QUERY ENDPOINT
# ============================================================================


@router.post("/stream")
async def stream_query_endpoint(
    request: QueryRequest,
):
    """
    Stream progress updates and final/token events from the RAG agent
    using Server-Sent Events (SSE).

    The input guardrail is applied before the SSE stream is opened.
    This allows GuardrailViolation to be returned as a normal HTTP 400
    response instead of being embedded inside an already-open SSE stream.

    Output guardrail processing is intentionally left to the downstream
    service/agent so that the existing streaming response behavior is
    preserved.
    """

    # ------------------------------------------------------------------------
    # Apply/trigger the query service before opening the SSE response.
    #
    # This preserves the GuardrailViolation handling from the guarded version
    # while keeping the actual async event iteration from the streaming
    # version.
    # ------------------------------------------------------------------------
    try:
        event_stream = query_documents_stream(
            query=request.query,
            thread_id=request.thread_id,
            user_id=request.user_id,
        )

    except GuardrailViolation as violation:
        raise HTTPException(
            status_code=400,
            detail={
                "guardrail": violation.guard,
                "message": violation.message,
            },
        ) from violation

    async def event_generator():
        try:
            async for event in event_stream:
                yield (
                    "data: "
                    + json.dumps(
                        event,
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )

        except GuardrailViolation as violation:
            # Defensive handling in case the streaming service is an async
            # generator and raises the guardrail only when iteration begins.
            print("[query_stream_route] " f"Guardrail violation: {violation}")

            yield (
                "data: "
                + json.dumps(
                    {
                        "event": "error",
                        "status_code": 400,
                        "guardrail": violation.guard,
                        "message": violation.message,
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )

        except EmbeddingServiceError as exc:
            print("[query_stream_route] " f"Embedding service unavailable: {exc}")

            yield (
                "data: "
                + json.dumps(
                    {
                        "event": "error",
                        "status_code": 503,
                        "message": (
                            "The embedding service is "
                            "currently unavailable. "
                            "Please try again later."
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )

        except Exception as exc:
            print("[query_stream_route] " f"Streaming query failed: {exc}")

            yield (
                "data: "
                + json.dumps(
                    {
                        "event": "error",
                        "status_code": 500,
                        "message": (
                            "Unable to process your question. " "Please try again."
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# CANCELLATION
# ============================================================================


@router.post("/cancel/{thread_id}")
def cancel_query_endpoint(
    thread_id: str,
):
    try:
        cancel_query(thread_id)

        return {
            "status": "success",
            "message": "Query cancellation requested.",
            "thread_id": thread_id,
        }

    except Exception as exc:
        print("[query_route] " f"Failed to cancel query: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Unable to cancel query.",
        ) from exc
