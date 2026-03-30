from control_plane.model_router import extract_features

def route_knowledge(query_analysis):

    features = extract_features(query_analysis)

    # -------------------------------
    # 1. Pure Coding → No Retrieval
    # -------------------------------
    if features["requires_coding"] or features["has_code"]:
        return "LLM_ONLY"

    # -------------------------------
    # 2. High Ambiguity → Need RAG
    # -------------------------------
    if features["ambiguity"] > 0.6:
        return "RAG"

    # -------------------------------
    # 3. General Knowledge → Hybrid
    # -------------------------------
    if features["type"] == "general":
        return "HYBRID"

    # -------------------------------
    # 4. Complex Reasoning → RAG
    # -------------------------------
    if features["complexity_score"] > 0.7:
        return "RAG"

    # -------------------------------
    # 5. Short Simple Queries → LLM
    # -------------------------------
    if features["length"] < 8:
        return "LLM_ONLY"

    # -------------------------------
    # Default
    # -------------------------------
    return "HYBRID"