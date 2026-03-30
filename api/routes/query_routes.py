from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from retrieval.pipeline import rag_pipeline
from retrieval.keyword_retriever import rebuild_from_chroma
from cache.memory_store import verify_memory
from control_plane.config_manager import config_manager   # ✅ NEW

import json
import time

router = APIRouter()


# ----------------------------------
# REQUEST MODELS
# ----------------------------------

class QueryRequest(BaseModel):
    question: str


class FeedbackRequest(BaseModel):
    question: str
    is_correct: bool


# ----------------------------------
# STREAMING RESPONSE
# ----------------------------------

def stream_response(answer, observability):

    words = answer.split()

    for word in words:
        yield word + " "
        time.sleep(0.02)

    # ----------------------------------
    # 🔥 Attach observability
    # ----------------------------------
    yield "\n\n__OBSERVABILITY_START__\n"
    yield json.dumps(observability)


# ----------------------------------
# QUERY ENDPOINT
# ----------------------------------

@router.post("/query")
def query(request: QueryRequest):

    answer, observability = rag_pipeline(request.question)

    # 🔥 Attach control-plane snapshot (NEW)
    observability["active_config"] = config_manager.dump_config()

    return StreamingResponse(
        stream_response(answer, observability),
        media_type="text/plain"
    )


# ----------------------------------
# INDEX REBUILD
# ----------------------------------

@router.post("/index/rebuild")
def rebuild_index():
    rebuild_from_chroma()
    return {
        "status": "success",
        "message": "BM25 index rebuilt from Chroma DB."
    }


# ----------------------------------
# FEEDBACK ENDPOINT
# ----------------------------------

@router.post("/feedback")
def feedback(request: FeedbackRequest):

    success = verify_memory(request.question, request.is_correct)

    return {
        "status": "success",
        "memory_verified": success,
        "message": "Feedback processed"
    }


# ----------------------------------
# 🔥 OPTIONAL: DEBUG CONFIG ENDPOINT
# ----------------------------------

@router.get("/config")
def get_config():
    """
    View current control-plane config (VERY USEFUL)
    """
    return config_manager.dump_config()