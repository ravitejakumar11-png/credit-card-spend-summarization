from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.v1.routes import query_routes
from src.api.v1.routes import upload_routes
from src.core.db import validate_embedding_service


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("========== APPLICATION STARTUP ==========")

    validate_embedding_service()

    print("========== APPLICATION STARTUP COMPLETE ==========")

    yield


app = FastAPI(
    title="NorthStar Credit Card Assistant",
    lifespan=lifespan,
)


@app.get("/")
async def root():

    return {"message": "NorthStar Credit Card Assistant"}


@app.get("/health")
def health_check():

    return {"status": "ok"}


app.include_router(query_routes.router)

app.include_router(upload_routes.router)
