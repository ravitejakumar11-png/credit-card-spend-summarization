from fastapi import FastAPI
from src.api.v1.routes import query

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(query.router)
