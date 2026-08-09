from fastapi import APIRouter
from src.api.v1.schemas.query_schema import QueryRequest, QueryResponse
from src.api.v1.services.query_service import query_documents

router = APIRouter(prefix="/api/v1/query")


@router.post("/")
def query_endpoint(request: QueryRequest) -> QueryResponse:
    docs = query_documents(request.query)
    return docs
