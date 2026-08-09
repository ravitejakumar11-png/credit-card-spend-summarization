from fastapi import FastAPI

from src.api.v1.routes import query
from src.api.v1.routes import upload_routes

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello from credit-card-spend-summarization!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(query.router)
app.include_router(upload_routes.router)
