from fastapi import APIRouter, HTTPException

from src.api.v1.schemas.query_schema import QueryRequest, QueryResponse
from src.api.v1.services.query_service import query_documents
from src.core.db import EmbeddingServiceError

router = APIRouter(prefix="/api/v1/query")


@router.post("/")
def query_endpoint(request: QueryRequest) -> QueryResponse:

    try:

        docs = query_documents(
            query=request.query,
            thread_id=request.thread_id,
        )

        return docs

    except EmbeddingServiceError as exc:

        print("[query_route] Embedding service unavailable: " f"{exc}")

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        print("[query_route] Query processing failed: " f"{exc}")

        raise HTTPException(
            status_code=500,
            detail="Unable to process the query.",
        ) from exc
