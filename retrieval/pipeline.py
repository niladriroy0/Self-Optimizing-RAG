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
    """Extract clean text from (doc, score) tuples or raw strings."""
    clean_docs = []

    for doc in retrieved_docs:
        if isinstance(doc, tuple):
            clean_docs.append(str(doc[0]))
        elif isinstance(doc, dict):
            clean_docs.append(str(doc.get("document", "")))
        else:
            clean_docs.append(str(doc))

    return clean_docs


def get_min_distance(scored_docs):
    """Return the minimum (best) L2 distance from a list of (doc, distance) tuples."""
    distances = [dist for _, dist in scored_docs if isinstance(dist, (int, float))]
    return min(distances) if distances else float("inf")


# ----------------------------------
# MAIN PIPELINE
# ----------------------------------

def rag_pipeline(question):

    tracker = LatencyTracker()
    start_time = time.time()

    print("\n" + "="*50)
    print(f"🚀 [NEW PIPELINE RUN]")
    print(f"User Query: '{question}'")
    print("="*50 + "\n")

    config_cp = config_manager.get_config()

    # ----------------------------------
    # QUERY ANALYSIS
    # ----------------------------------
    print("🔍 [STEP 1] Executing Query Analysis...")
    query_analysis = analyze_query(question)

    # ----------------------------------
    # CACHE
    # ----------------------------------
    print("🗂️  [STEP 2] Checking Exact Match Cache...")
    if query_analysis.get("type") != "coding" and config_cp.get("enable_query_cache", True):
        cached = get_cached(question)
        if cached:
            print("   ⚡ CACHE HIT! Short-circuiting pipeline.")
            return cached
        print("   ❌ Cache Miss.")

    # ----------------------------------
    # MODEL ROUTING
    # ----------------------------------
    print("🚦 [STEP 3] Resolving Dynamic Model Routes...")
    model = route_model_with_exploration(query_analysis, epsilon=0.1)
    print(f"   🤖 Selected Primary Model: {model}")

    # ----------------------------------
    # KNOWLEDGE ROUTING
    # ----------------------------------
    mode = route_knowledge(query_analysis)
    print(f"   🛤️ Selected Knowledge Pathway: {mode}")

    # ----------------------------------
    # 🔥 LLM ONLY MODE (FIXED)
    # ----------------------------------

    if mode == "LLM_ONLY":
        print("\n🧠 [FLOW: LLM-ONLY TRIGGERED]")
        print("   Fetching semantic memories to augment parametric knowledge...")
        
        tracker.start("llm")

        memory_context = retrieve_memory(question)
        context = "\n".join(memory_context)
        
        print(f"   Generating Answer (Context size: {len(memory_context)} memories)...")
        answer = generate_answer(
            context=context,
            question=question,
            model=model,
            query_type=query_analysis.get("type", "general"),
            query_analysis=query_analysis
        )

        tracker.end("llm")

        print("📊 [STEP 4] Computing LLM-Only Confidence...")
        confidence = compute_confidence(
            clean_reranked=[],
            top_k=0,
            memory_context=[],
            query_analysis=query_analysis,
            evaluation_scores=None,
            answer=answer
        )
        print(f"   Final Confidence Source: {confidence:.2f}")

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

        print("💾 [STEP 5] Updating Sub-Systems & Terminating Flow...")
        if query_analysis.get("type") != "coding":
            set_cache(question, final_response)
            
        store_memory(question, answer, confidence)

        print("✅ Pipeline Complete.\n")
        return final_response

    # ----------------------------------
    # CONFIG
    # ----------------------------------
    print("\n📚 [FLOW: HYBRID/RAG TRIGGERED]")
    print("⚙️  Applying Optimizer Settings...")

    config = choose_config()
    top_k = config_cp.get("top_k", config.get("top_k", 3))

    # ----------------------------------
    # QUERY PLANNING
    # ----------------------------------
    print("🗺️  Executing Query Planning & Sub-Queries...")
    sub_queries = plan_query(question, query_analysis)

    if len(sub_queries) > 1:
        print(f"   🔀 Strategy: Multi-hop reasoning ({len(sub_queries)} queries)")

    # ----------------------------------
    # RETRIEVAL
    # ----------------------------------
    print("🔎 [STEP 4] Executing Hybrid Search...")
    tracker.start("retrieval")

    all_docs = []         # plain text docs
    all_scored_raw = []   # (doc, distance) tuples for quality checks

    for sub_q in sub_queries:
        scored_results = hybrid_search(
            sub_q,
            query_analysis=query_analysis,
            top_k=top_k
        )
        all_scored_raw.extend(scored_results)
        all_docs.extend(normalize_documents(scored_results))

    tracker.end("retrieval")
    print(f"   📥 Retrieved {len(all_docs)} raw documents from Vector store.")

    # --------------------------
    # FALLBACK ROUTE
    # --------------------------
    if not all_docs:
        print("\n⚠️  [CRITICAL] Vector Search returned 0 documents.")
        print("   🔄 Executing Smart Fallback: Promoting to Parametric Intelligence...")
        
        tracker.start("llm")
        memory_context = retrieve_memory(question)
        context = "\n".join(memory_context)
        
        answer = generate_answer(
            context=context,
            question=question,
            model=model,
            query_type=query_analysis.get("type", "general"),
            query_analysis=query_analysis
        )
        tracker.end("llm")
        
        confidence = compute_confidence(
            clean_reranked=[],
            top_k=0,
            memory_context=memory_context,
            query_analysis=query_analysis,
            evaluation_scores=None,
            answer=answer
        )
        total_latency = time.time() - start_time
        
        observability = {
            "mode": "FALLBACK_LLM",
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
        store_memory(question, answer, confidence)
        
        print("✅ Pipeline Complete (Via Fallback).\n")
        return final_response

    # ----------------------------------
    # DEDUP & RERANK
    # ----------------------------------
    print("🧹 [STEP 5] Deduplicating & Cross-Encoder Reranking...")
    retrieved_docs = list(dict.fromkeys(all_docs))

    tracker.start("rerank")
    rerank_query = " ".join(sub_queries) if len(sub_queries) > 1 else question

    reranked = rerank(rerank_query, retrieved_docs)
    clean_reranked = [(doc, float(score)) for doc, score in reranked]
    reranked_docs = [doc for doc, _ in clean_reranked]
    tracker.end("rerank")

    # ----------------------------------
    # 🔥 EARLY FALLBACK: Score Quality Gate
    # ----------------------------------
    min_relevance = config_cp.get("min_relevance_threshold", 0.15)
    top_rerank_score = clean_reranked[0][1] if clean_reranked else 0.0

    if top_rerank_score < min_relevance:
        print(f"\n⚡ [EARLY FALLBACK] Top reranker score ({top_rerank_score:.3f}) "
              f"< threshold ({min_relevance}). Docs are irrelevant — skipping RAG LLM call.")
        print("   🔄 Jumping directly to Parametric Intelligence...")

        tracker.start("llm")
        memory_context = retrieve_memory(question)
        context = "\n".join(memory_context)

        answer = generate_answer(
            context=context,
            question=question,
            model=model,
            query_type=query_analysis.get("type", "general"),
            query_analysis=query_analysis
        )
        tracker.end("llm")

        confidence = compute_confidence(
            clean_reranked=[],
            top_k=0,
            memory_context=memory_context,
            query_analysis=query_analysis,
            evaluation_scores=None,
            answer=answer
        )
        total_latency = time.time() - start_time

        observability = {
            "mode": "EARLY_FALLBACK_LLM",
            "model_used": model,
            "query_analysis": query_analysis,
            "early_fallback_reason": f"Reranker score {top_rerank_score:.3f} < threshold {min_relevance}",
            "latency": {
                **tracker.get_all(),
                "total_seconds": total_latency
            },
            "confidence": confidence
        }

        final_response = (answer, observability)
        if query_analysis.get("type") != "coding":
            set_cache(question, final_response)
        store_memory(question, answer, confidence)

        print("✅ Pipeline Complete (Early Fallback).\n")
        return final_response

    final_docs = reranked_docs[:top_k]
    print(f"   🎯 Highest quality chunks isolated (Top K = {top_k})")

    # ----------------------------------
    # MEMORY & CONTEXT MESHING
    # ----------------------------------
    print("🧠 [STEP 6] Syncing Semantic Memory Store...")
    memory_context = retrieve_memory(question)

    print("🧩 Meshing Memories with Vector DB Chunks...")
    context_parts = []
    context_parts.extend(memory_context[:2])
    context_parts.extend(final_docs)
    context = "\n".join(context_parts)

    # ----------------------------------
    # LLM
    # ----------------------------------
    print("💬 [STEP 7] Executing Primary Generation...")
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
    print("⚖️  [STEP 8] Synchronous Answer Evaluation...")
    evaluation_scores = evaluate_rag(question, answer, context)

    print("📊 Computing Final Global Confidence Score...")
    confidence = compute_confidence(
        clean_reranked,
        top_k,
        memory_context,
        query_analysis,
        evaluation_scores,
        answer=answer
    )
    print(f"   Final Confidence Set: {confidence:.2f}")

    # ----------------------------------
    # MEMORY STORE & ASYNC LOGGING
    # ----------------------------------
    print("💾 [STEP 9] Updating Systems & Terminating Flow...")
    store_memory(question, answer, confidence)

    print("   📡 Firing Async Evaluation Worker Hook...")
    run_evaluation_async(
        question,
        answer,
        context,
        config,
        confidence
    )

    # ----------------------------------
    # FINAL METRICS
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

    if query_analysis.get("type") != "coding":
        set_cache(question, final_response)

    print(f"⏱️  Total Processing Time: {total_latency:.2f}s")
    print("✅ Pipeline Complete.\n")

    return final_response