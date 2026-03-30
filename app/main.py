from fastapi import FastAPI
from api.routes.query_routes import router
from vectorstore.chroma_store import collection
from retrieval.keyword_retriever import build_index
from optimization.experiment_db import init_db

import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite schema
    init_db()

    results = collection.get()

    docs = results["documents"]

    if docs:
        build_index(docs)

    print("BM25 index built with", len(docs), "documents")
    try:
        yield
    except asyncio.CancelledError:
        pass

app = FastAPI(title="Self Optimizing RAG", lifespan=lifespan)

app.include_router(router)


@app.get("/")
def health():
    return {"status": "running"}