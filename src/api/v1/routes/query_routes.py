import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.v1.schemas.query_schema import (
    QueryRequest,
    QueryResponse,
)

from src.api.v1.services.query_service import (
    query_documents,
    query_documents_stream,
)

from src.api.v1.services.query_cancellation import (
    cancel_query,
)

from src.core.db import EmbeddingServiceError

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
        )

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
def query_stream_endpoint(
    request: QueryRequest,
):

    def event_generator():

        try:

            for event in query_documents_stream(
                query=request.query,
                thread_id=request.thread_id,
            ):

                yield (
                    json.dumps(
                        event,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        except EmbeddingServiceError as exc:

            print("[query_stream_route] " f"Embedding service unavailable: {exc}")

            yield (
                json.dumps(
                    {
                        "event": "error",
                        "status_code": 503,
                        "message": (
                            "The embedding service is "
                            "currently unavailable. "
                            "Please try again later."
                        ),
                    }
                )
                + "\n"
            )

        except Exception as exc:

            print("[query_stream_route] " f"Streaming query failed: {exc}")

            yield (
                json.dumps(
                    {
                        "event": "error",
                        "status_code": 500,
                        "message": (
                            "Unable to process your question. " "Please try again."
                        ),
                    }
                )
                + "\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
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
