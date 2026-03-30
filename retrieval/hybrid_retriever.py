# retrieval/hybrid_retriever.py

from vectorstore.chroma_store import search_documents
from retrieval.keyword_retriever import keyword_search
from control_plane.config_manager import config_manager


# ----------------------------------
# 🔥 HYBRID SEARCH (ANALYSIS-DRIVEN)
# ----------------------------------

def hybrid_search(query, query_analysis=None, top_k=3):

    query_analysis = query_analysis or {}

    use_hybrid = config_manager.get_param("enable_hybrid", True)

    keyword_weight = config_manager.get_param("keyword_weight", 0.5)
    semantic_weight = config_manager.get_param("semantic_weight", 0.5)

    needs_keyword = query_analysis.get("needs_keyword", False)
    needs_semantic = query_analysis.get("needs_semantic", True)

    results = []

    # ----------------------------------
    # VECTOR SEARCH
    # ----------------------------------

    if use_hybrid and needs_semantic:
        vector_k = max(1, int(top_k * semantic_weight * 3))
        vector_results = search_documents(query, k=vector_k)
    else:
        vector_results = []

    # ----------------------------------
    # KEYWORD SEARCH
    # ----------------------------------

    if use_hybrid and needs_keyword:
        keyword_k = max(1, int(top_k * keyword_weight * 3))
        keyword_results = keyword_search(query, k=keyword_k)
    else:
        keyword_results = []

    # ----------------------------------
    # FALLBACK (if nothing triggered)
    # ----------------------------------

    if not vector_results and not keyword_results:
        vector_results = search_documents(query, k=top_k * 2)

    # ----------------------------------
    # MERGE (DEDUP SAFE)
    # ----------------------------------

    seen = set()

    def add_unique(docs):
        for d in docs:
            key = str(d)  # safe fallback
            if key not in seen:
                seen.add(key)
                results.append(d)

    add_unique(vector_results)
    add_unique(keyword_results)

    # ----------------------------------
    # FINAL TRIM (NO RERANK HERE)
    # ----------------------------------

    return results[: max(top_k * 5, 20)]