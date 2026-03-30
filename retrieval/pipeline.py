import time

from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import rerank

from llm.llm_service import generate_answer

from optimization.optimizer import choose_config
from query_processing.query_analyzer import analyze_query
from query_processing.query_planner import plan_query

from control_plane.model_router import route_model_with_exploration
from control_plane.knowledge_router import route_knowledge
from control_plane.config_manager import config_manager

from cache.query_cache import get_cached, set_cache
from cache.chroma_memory_store import retrieve_memory, store_memory

from workers.evaluation_worker import run_evaluation_async
from observability.latency_tracker import LatencyTracker

from evaluation.confidence_model import compute_confidence
from evaluation.rag_evaluator import evaluate_rag


# ----------------------------------
# UTILITY
# ----------------------------------

def normalize_documents(retrieved_docs):

    clean_docs = []

    for doc in retrieved_docs:
        if isinstance(doc, tuple):
            clean_docs.append(str(doc[0]))
        elif isinstance(doc, dict):
            clean_docs.append(str(doc.get("document", "")))
        else:
            clean_docs.append(str(doc))

    return clean_docs


# ----------------------------------
# MAIN PIPELINE
# ----------------------------------

def rag_pipeline(question):

    tracker = LatencyTracker()
    start_time = time.time()

    print("\n==============================")
    print("Incoming Query:", question)

    config_cp = config_manager.get_config()

    # ----------------------------------
    # QUERY ANALYSIS
    # ----------------------------------

    query_analysis = analyze_query(question)

    # ----------------------------------
    # CACHE
    # ----------------------------------

    if query_analysis.get("type") != "coding" and config_cp.get("enable_query_cache", True):
        cached = get_cached(question)
        if cached:
            print("⚡ Cache Hit → returning directly")
            return cached

    # ----------------------------------
    # MODEL ROUTING
    # ----------------------------------

    model = route_model_with_exploration(query_analysis, epsilon=0.1)
    print("Model selected:", model)

    # ----------------------------------
    # KNOWLEDGE ROUTING
    # ----------------------------------

    mode = route_knowledge(query_analysis)
    print("Knowledge Mode:", mode)

    # ----------------------------------
    # 🔥 LLM ONLY MODE (FIXED)
    # ----------------------------------

    if mode == "LLM_ONLY":

        tracker.start("llm")

        answer = generate_answer(
            context="",
            question=question,
            model=model,
            query_type=query_analysis.get("type", "general"),
            query_analysis=query_analysis
        )

        tracker.end("llm")

        # ✅ FIX: compute real confidence
        confidence = compute_confidence(
            clean_reranked=[],
            top_k=0,
            memory_context=[],
            query_analysis=query_analysis,
            evaluation_scores=None,
            answer=answer
        )

        total_latency = time.time() - start_time

        observability = {
            "mode": "LLM_ONLY",
            "model_used": model,
            "query_analysis": query_analysis,
            "latency": {
                **tracker.get_all(),
                "total_seconds": total_latency
            },
            "confidence": confidence
        }

        final_response = (answer, observability)

        if query_analysis.get("type") != "coding":
            set_cache(question, final_response)

        return final_response

    # ----------------------------------
    # CONFIG
    # ----------------------------------

    config = choose_config()
    top_k = config_cp.get("top_k", config.get("top_k", 3))

    print("Optimizer Config:", config)
    print("Control Plane Config:", config_cp)

    # ----------------------------------
    # QUERY PLANNING
    # ----------------------------------

    sub_queries = plan_query(question, query_analysis)

    if len(sub_queries) > 1:
        print("🔀 Multi-hop queries:", sub_queries)

    # ----------------------------------
    # RETRIEVAL
    # ----------------------------------

    tracker.start("retrieval")

    all_docs = []

    for sub_q in sub_queries:
        docs = hybrid_search(
            sub_q,
            query_analysis=query_analysis,
            top_k=top_k
        )
        docs = normalize_documents(docs)
        all_docs.extend(docs)

    tracker.end("retrieval")

    if not all_docs:
        return "No relevant documents found.", {}

    # ----------------------------------
    # DEDUP
    # ----------------------------------

    retrieved_docs = list(dict.fromkeys(all_docs))

    # ----------------------------------
    # RERANK
    # ----------------------------------

    tracker.start("rerank")

    rerank_query = " ".join(sub_queries) if len(sub_queries) > 1 else question

    reranked = rerank(rerank_query, retrieved_docs)
    clean_reranked = [(doc, float(score)) for doc, score in reranked]

    reranked_docs = [doc for doc, _ in clean_reranked]

    tracker.end("rerank")

    final_docs = reranked_docs[:top_k]

    # ----------------------------------
    # MEMORY
    # ----------------------------------

    memory_context = retrieve_memory(question)

    # ----------------------------------
    # CONTEXT
    # ----------------------------------

    context_parts = []
    context_parts.extend(memory_context[:2])
    context_parts.extend(final_docs)

    context = "\n".join(context_parts)

    print(
        "Retrieved Docs:", len(final_docs),
        "| Memory Used:", len(memory_context)
    )

    # ----------------------------------
    # LLM
    # ----------------------------------

    tracker.start("llm")

    answer = generate_answer(
        context=context,
        question=question,
        model=model,
        query_type=query_analysis.get("type", "general"),
        query_analysis=query_analysis
    )

    tracker.end("llm")

    # ----------------------------------
    # EVALUATION + CONFIDENCE
    # ----------------------------------

    evaluation_scores = evaluate_rag(question, answer, context)

    confidence = compute_confidence(
        clean_reranked,
        top_k,
        memory_context,
        query_analysis,
        evaluation_scores,
        answer=answer   # 🔥 already correct
    )

    # ----------------------------------
    # MEMORY STORE
    # ----------------------------------

    store_memory(question, answer, confidence)

    # ----------------------------------
    # ASYNC LOGGING
    # ----------------------------------

    run_evaluation_async(
        question,
        answer,
        context,
        config,
        confidence
    )

    # ----------------------------------
    # OBSERVABILITY
    # ----------------------------------

    total_latency = time.time() - start_time

    observability = {
        "mode": mode,
        "model_used": model,
        "query_analysis": query_analysis,
        "optimizer_config": config,
        "control_plane_config": config_cp,
        "retrieval": {
            "total_docs": len(retrieved_docs),
            "top_k": top_k,
            "sub_queries": sub_queries
        },
        "memory_used": len(memory_context),
        "confidence": confidence,
        "evaluation_scores": evaluation_scores,
        "latency": {
            **tracker.get_all(),
            "total_seconds": total_latency
        }
    }

    final_response = (answer, observability)

    # ----------------------------------
    # CACHE
    # ----------------------------------

    if query_analysis.get("type") != "coding":
        set_cache(question, final_response)

    return final_response