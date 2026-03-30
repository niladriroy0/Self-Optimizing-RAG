def analyze_query(query: str):

    q = query.lower()
    words = q.split()

    analysis = {
        "length": len(words),
        "type": "general",
        "complexity": "medium",          # ✅ NEW
        "is_multi_hop": False         # ✅ NEW
    }

    # ----------------------------------
    # TYPE DETECTION
    # ----------------------------------

    if any(w in q for w in ["code", "python", "bug", "algorithm"]):
        analysis["type"] = "coding"

    elif any(w in q for w in ["explain", "why", "how", "reason"]):
        analysis["type"] = "reasoning"

    # ----------------------------------
    # COMPLEXITY DETECTION
    # ----------------------------------

    if len(words) > 12 or "compare" in q or "difference" in q:
        analysis["complexity"] = "high"

    # ----------------------------------
    # MULTI-HOP DETECTION
    # ----------------------------------

    if any(k in q for k in [" and ", " then ", "compare", "difference"]):
        analysis["is_multi_hop"] = True

    return analysis