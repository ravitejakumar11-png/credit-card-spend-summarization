from fastapi import FastAPI
from src.api.v1.routes import query_routes
from src.api.v1.routes import upload_routes

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(query_routes.router)
app.include_router(upload_routes.router)
